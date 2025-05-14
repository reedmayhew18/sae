import argparse
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

def build_activations(
        dataset, 
        tokenizer, 
        model, 
        device, 
        layer_index=15,    # which layer to hook
        batch_size=8,      # number of samples per batch
        file_size=10*8192, # max examples per output file
        vector_size=4096,  # hidden-size of that layer
        data_len=30000, 
        output_dir="activations_data"
    ):
    # 1) Hook to capture the raw (batch, seq_len, vector_size) activations
    activation_cache = []
    def save_activations_hook(module, input, output):
        # input[0] has shape (batch, seq_len, vector_size)
        activation_cache.append(input[0].cpu())
    handle = model.model.layers[layer_index].register_forward_hook(save_activations_hook)

    # 2) Prepare accumulator & counters
    files_saved = 0
    batch_texts = []
    all_data = np.empty((0, vector_size), dtype=np.float16)

    print("Processing dataset and saving activations in batches…")
    for i, example in enumerate(dataset):
        batch_texts.append(example["text"])
        is_end_of_batch = ((i + 1) % batch_size == 0) or (i + 1 >= data_len)

        if is_end_of_batch:
            # Tokenize + pad/truncate to fixed length
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=513,
                padding="max_length",
            ).to(device)

            # Run forward, our hook will fill activation_cache
            with torch.no_grad():
                model(**inputs)

            # Stack the single-element list into a tensor of shape (batch, seq_len, vector)
            acts = torch.cat(activation_cache, dim=0)  # -> (batch, seq_len, vector_size)
            # Mean-pool across the token dimension → (batch, vector_size)
            pooled = acts.mean(dim=1)                 # -> (batch, vector_size)
            batch_activations = pooled.cpu().numpy().astype(np.float16)

            # Append to the master array
            all_data = np.vstack((all_data, batch_activations))
            print(f"  collected {all_data.shape[0]} total rows so far")

            # Save out a file if we’ve crossed the file_size threshold
            while all_data.shape[0] >= file_size:
                out_path = os.path.join(output_dir, f"activations_{files_saved:04d}.npy")
                np.save(out_path, all_data[:file_size])
                print(f"  → saved {out_path} ({file_size} rows)")
                files_saved += 1
                all_data = all_data[file_size:]  # keep the remainder

            # reset for the next batch
            batch_texts = []
            activation_cache.clear()

        if (i + 1) >= data_len:
            break

    # Save any leftover rows
    if all_data.shape[0] > 0:
        out_path = os.path.join(output_dir, f"activations_{files_saved:04d}.npy")
        np.save(out_path, all_data)
        print(f"  → saved remainder {out_path} ({all_data.shape[0]} rows)")

    handle.remove()
    print("Done extracting all activations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract mean-pooled activations from a LLaMA model"
    )
    home = os.path.expanduser("~")
    default_cache = os.path.join(home, ".cache", "huggingface", "hub")

    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Llama-3.2-3B",
                        help="HuggingFace model ID")
    parser.add_argument("--cache_dir", type=str,
                        default=default_cache,
                        help="HF cache directory")
    parser.add_argument("--layer_index", type=int, default=15,
                        help="Which transformer layer to hook")
    parser.add_argument("--data_len", type=int, default=30000,
                        help="Total examples to process")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--file_size", type=int, default=81920,
                        help="Max rows to store per .npy file")
    parser.add_argument("--vector_size", type=int, default=3072,
                        help="Dimensionality of the hidden layer")
    parser.add_argument("--output_dir", type=str, default="activations_data",
                        help="Where to write .npy files")
    args = parser.parse_args()

    # device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, cache_dir=args.cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # make sure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    # streaming load of dataset
    dataset = load_dataset(
        "JeanKaddour/minipile",
        split="train",
        streaming=True
    )

    # run it
    build_activations(
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        device=device,
        layer_index=args.layer_index,
        batch_size=args.batch_size,
        file_size=args.file_size,
        vector_size=args.vector_size,
        data_len=args.data_len,
        output_dir=args.output_dir
    )
