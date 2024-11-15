# Scaling Monosemanticity with LLaMA

This repository reproduces and extends the work from the article [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html), applying the methods to LLaMA 3.2-3B. The project explores monosemantic neurons in large language models, investigates their scaling behavior, and implements sparse autoencoders to extract interpretable features.

---

## Features

- **Model Exploration**: Uses LLaMA 3.2-3B to examine activations and extract monosemantic features.
- **Sparse Autoencoders**: Implementation of sparse autoencoders to reduce dimensionality and identify interpretable latent features.
- **Feature Influence**: Modify specific features in the latent space to influence the model's output.
- **Analysis Tools**: Includes tools for sparsity analysis, UMAP visualization, and feature specificity.
- **Dataset Integration**: Processes large datasets like "The Pile" to study activations at scale.

---

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/scaling-monosemanticity-llama.git
   cd scaling-monosemanticity-llama
   ```

2. Install required dependencies:
   ```bash
   pip install transformers datasets keras huggingface_hub tensorflow python-dotenv zstandard umap-learn
   ```

3. Set up your Hugging Face token:
   - Add your token to a `.env` file:
     ```
     HF_TOKEN=your_huggingface_token
     ```

4. Download the model:
   - LLaMA 3.2-3B is used as the base model. You can adjust to other LLaMA variants if desired.

---

## Usage

### 1. Model Activations
Run the script to save residual stream activations for specific layers of the LLaMA model. Activations are stored in `.npy` files for further processing.

### 2. Sparse Autoencoder Training
Train a sparse autoencoder to identify and encode significant features:
```bash
python train_autoencoder.py
```

### 3. Feature Analysis
Analyze the extracted latent vectors for sparsity, specificity, and feature completeness using the included visualization tools.

### 4. Influence LLM
Manipulate specific features in the latent space to influence the model's outputs:
```bash
python influence_llm.py
```

---

## Results

1. **Sparsity in Latent Representations**: Sparse autoencoders effectively reduce the dimensionality of activations while maintaining interpretability.
2. **Feature Influence**: Specific latent features were shown to impact the output text in predictable ways.
3. **UMAP Visualizations**: Dimensionality reduction techniques like UMAP reveal clusters of semantically meaningful features.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Based on the original work in [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html).
- Special thanks to Hugging Face for providing open-access models and datasets.
