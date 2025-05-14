#!/usr/bin/env python3
"""
Extract hidden-layer activations from a Llama-3 model using the existing SAE pipeline.
This script reuses src/build_activations.py's build_activations() function.
"""
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from src.build_activations import build_activations

def main():
    parser = argparse.ArgumentParser(
        description="Extract activations from a Llama-3 model and save to disk"
    )
    parser.add_argument(
        "--model_name", type=str,
        default="unsloth/Meta-Llama-3.1-8B-Instruct",
        help="HuggingFace model identifier for Llama-3"
    )
    home = os.path.expanduser("~")
    default_cache = os.path.join(home, ".cache/huggingface/hub")
    parser.add_argument(
        "--cache_dir", type=str,
        default=default_cache,
        help="Transformers cache directory"
    )
    parser.add_argument("--layer_index", type=int, default=15,
                        help="Index of transformer layer to hook")
    parser.add_argument("--data_len", type=int, default=30000,
                        help="Number of examples to process from dataset")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of sequences per forward pass")
    parser.add_argument("--file_size", type=int, default=10*8192,
                        help="Max number of token-activations per output file")
    parser.add_argument("--vector_size", type=int, default=4096,
                        help="Dimension of hidden-state vectors")
    parser.add_argument("--output_dir", type=str, default="activations_data",
                        help="Directory to write activation .npy files")
    parser.add_argument(
        "--quant4bit", action="store_true",
        help="Enable 4-bit quantized model load via bitsandbytes"
    )
    args = parser.parse_args()

    # select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token

    # load model, optionally in 4-bit
    if args.quant4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    # streaming dataset
    dataset = load_dataset(
        "JeanKaddour/minipile",
        split="train",
        streaming=True
    )

    # ensure output exists
    os.makedirs(args.output_dir, exist_ok=True)

    # extract and save activations
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

if __name__ == "__main__":
    main()
