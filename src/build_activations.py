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
        layer_index=15,  # layer index to extract activations from
        batch_size=8,  # number of sentences in a batch
        file_size=10*8192,  # number of examples in a file
        vector_size=3072, 
        data_len=30_000, 
        output_dir="activations_data"
    ):
    # Register hook on the 16th layer
    activation_cache = [] # cache of activations for a batch
    def save_activations_hook(module, input, output):
        activation_cache.append(input[0].cpu().detach().numpy())
    hook_handle = model.model.layers[layer_index].register_forward_hook(save_activations_hook) 

    # Initialize accumulators and parameters
    files_saved = 0
    batch_texts = []
    all_data = np.empty((0, vector_size+3), dtype=np.float16)  # 3072 + 3 (sent_idx, token_idx, token)
    all_last3 = np.empty((0, 3), dtype=np.float32)  # last3: sent_idx, token_idx, token

    # Create batches from the dataset
    print("Processing dataset and saving activations in batches...")
    for i, example in enumerate(dataset):
        batch_texts.append(example['text'])
        
        if (i + 1) % batch_size == 0 or i + 1 >= data_len:
            # Process full batch or final partial batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=513, # 512 + 1 for <begin_of_text> token, which will be removed later
                padding="max_length",
            ).to(device)
            
            with torch.no_grad():
                model(**inputs)
            
            # Convert activation_cache to numpy array and reshape
            batch_activations = np.array(activation_cache)
            
            # Reshape batch_activations from (1, 8, 513, 3072) to (8*513, 3072)
            batch_activations = batch_activations.reshape(batch_activations.shape[1] * batch_activations.shape[2], -1)

            # Create sentence index array (sent_idx) and token index array (token_idx)
            # sent_idx = [1 1 1 1 1; 2 2 2 2 2; 3 3 3 3 3; ...]
            # token_idx = [1 2 3 4 5; 1 2 3 4 5; 1 2 3 4 5; ...]
            num_sentences, num_tokens = inputs['attention_mask'].shape # (8, 513)
            sent_idx = np.repeat(np.arange(1, num_sentences + 1), num_tokens).reshape(-1, 1)  # Shape: (8*513, 1)
            sent_idx = sent_idx + (i - batch_size) # offset by batch index
            token_idx = np.tile(np.arange(1, num_tokens + 1), num_sentences).reshape(-1, 1)    # Shape: (8*513, 1)
            token_idx = token_idx - 1 # begin with 0
            # also save tokens id from tokenizer
            tokens = inputs['input_ids'].cpu().numpy().reshape(-1, 1)
            tokens_offset = tokens - 64000  # offset by 64000 to not overflow float16
                                    # 128000 is the vocab size of Llama 3.2 3B, float16 is [-65504, 65504]
                    
            # Stack activations, sent_idx, and token_idx
            batch_activations = np.hstack((batch_activations, sent_idx, token_idx, tokens_offset)).astype(np.float16)
            batch_last3 = np.hstack((sent_idx, token_idx, tokens)).astype(np.float32)

            # Remove rows where attention mask is 0 (padding tokens)
            attention_mask = inputs['attention_mask'].cpu().numpy().reshape(-1)
            attention_mask[token_idx.ravel() == 0] = 0 # Remove <begin_of_text> token (token_idx = 0)
            batch_activations = batch_activations[attention_mask != 0]
            batch_last3 = batch_last3[attention_mask != 0]

            # Stack to all_data
            all_data = np.vstack((all_data, batch_activations))
            all_last3 = np.vstack((all_last3, batch_last3))
            print(f"all_data shape: {all_data.shape}")
            print(f"all_last3 shape: {all_last3.shape}")

            # Save to file if file_size limit is reached
            if all_data.shape[0] >= file_size:
                activations_file = os.path.join(output_dir, f"activations_batch_{files_saved:04d}.npy")
                last3_file = os.path.join(output_dir, f"last3_batch_{files_saved:04d}.npy")

                np.save(activations_file, all_data[:file_size, :])
                np.save(last3_file, all_last3[:file_size, :])

                files_saved += 1
                print(f"Saved file {files_saved} with {file_size} examples")
                all_data = all_data[file_size:, :]  # Retain any remaining rows
                all_last3 = all_last3[file_size:, :]  # Retain any remaining rows
                
            # Reset for next batch
            batch_texts = []
            activation_cache = []

        if i + 1 >= data_len:
            break

    # Save any remaining data
    if all_data.shape[0] > 0:
        activations_file = os.path.join(output_dir, f"activations_batch_{files_saved:04d}.npy")
        last3_file = os.path.join(output_dir, f"last3_batch_{files_saved:04d}.npy")

        np.save(activations_file, all_data)
        np.save(last3_file, all_last3)
        del all_data
        del all_last3

    print("Finished processing and saving all batches")
    hook_handle.remove()

if __name__ == "__main__":
    # model_name = "meta-llama/Llama-3.2-3B"
    # layer_index = 15
    # data_len = 3_000_000
    # batch_size = 8
    # file_size = 10*8192
    # vector_size = 3072
    # output_dir = "activations_data"

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract activations from LLaMA model and save to disk")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B", help="Model name")
    # <home>/.cache/huggingface/hub/<model_name>
    home_dir = os.path.expanduser("~")
    c_dir = os.path.join(home_dir, ".cache/huggingface/hub")
    parser.add_argument("--cache_dir", type=str, default=c_dir, help="Cache directory")
    parser.add_argument("--layer_index", type=int, default=15, help="Layer index to extract activations from")
    parser.add_argument("--data_len", type=int, default=30000, help="Number of dataset samples to process")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--file_size", type=int, default=81920, help="Max number of samples per file")
    parser.add_argument("--vector_size", type=int, default=3072, help="Size of activation vectors")
    parser.add_argument("--output_dir", type=str, default="activations_data", help="Output directory for activations")
    args = parser.parse_args()

    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (consider running on a GPU node)")

    # Load the LLaMA 3.2B model without quantization (for now)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare 4-bit quantization configuration (optional)
    # Uncomment the following lines if you wish to use quantization
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        # quantization_config=bnb_config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load the first 3 million examples from 'The Pile' dataset
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

    # Set up processing parameters
    os.makedirs(args.output_dir, exist_ok=True)

    # Build activations
    build_activations(
        dataset,
        tokenizer,
        model,
        device,
        layer_index=args.layer_index,
        batch_size=args.batch_size,
        file_size=args.file_size,
        vector_size=args.vector_size,
        data_len=args.data_len,
        output_dir=args.output_dir
    )



