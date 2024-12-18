# Scaling Monosemanticity with LLaMA

This repository reproduces and extends the work from the article [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html), applying the methods to LLaMA 3.2-3B. The project explores monosemantic neurons in large language models, investigates their scaling behavior, and implements sparse autoencoders to extract interpretable features.

---

## Features

- **Extracting Activations**: Analyze token-level activations from the 16th layer of LLaMA 3.2 (3B, 16-bit quantized) using the Pile dataset.
- **Sparse Autoencoder (SAE)**: Train an overcomplete SAE (3072-65536-3072) to uncover interpretable features in the latent space.
- **Feature Search**: Identify features relevant to specific topics using multi-prompt inputs and activation metrics.
- **Influencing Outputs**: Use steering vectors derived from the SAE to influence LLaMA’s output during inference.
