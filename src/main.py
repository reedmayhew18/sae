# %%
from analysis_utils import top_activations, prompt_search_mean_local, prompt_search_rank, influence, cosine_similarity_search
from plot_utils import examples_plot_df, examples_plot_df_horizontal, examples_plot_anthropic, plot_UMAP
from autoencoder import SparseAutoencoder

# %%
import json
import glob
import torch
import numpy as np

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# %%
model_name = "meta-llama/Llama-3.2-3B"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# %%
indicies = [32026, 57660, 45783, 41517, 64668, 29073, 14701, 6527, 49447, 52757, 9720, 50479, 6949, 17260, 64775,  25398]
top_examples = top_activations(
    feat_idx_list=indicies,
    k=20,
    context_len=15,
    latent_dataset_path="../sparse_latent_vectors/latent_vectors_batch_*.pt",
    dataset_slice=(5,50),
    act_id_dataset="../activations_data/last3_batch_*.npy",
    tokenizer=tokenizer,
    verbose=True
)

# %%
print("Top 1 example for feat 45783")
print(json.dumps(top_examples[45783][0], indent=2))

# %%
examples_plot_df(top_examples[25398][0]) # Plots the top example for feature 25398

# %%
examples_plot_df_horizontal(top_examples[45783][0])

# %%
feat_idx = 45783
top5_examples = {}
top5_examples[feat_idx] = {}
for i in range(5): # slice top 5
    top5_examples[feat_idx][i] = top_examples[feat_idx][i]

examples_plot_anthropic(feat_idx, top5_examples[feat_idx], save_path=None)

# %%
# Model and tokenizer setup
model_name = "meta-llama/Llama-3.2-3B"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

# Load the sparse autoencoder model and weights
input_dim = 3072
hidden_dim = 2 ** 16 # 65536
model_sae = SparseAutoencoder(input_dim, hidden_dim)
# model_sae.load_state_dict(torch.load("sparse_autoencoder.pth"))
checkpoint = torch.load("../models/checkpoint", weights_only=True)
model_sae.load_state_dict(checkpoint['state_dict'])
# model_sae.eval()


# %%
pos_prompts = [
    "program on aerobic capacity and muscle strength of adults with hearing loss. Twenty-three adults with hearing loss were separated into 2 groups. Thirteen subjects",
    "the effect of a traditional dance training program on aerobic capacity and muscle strength of adults with hearing loss. Twenty-three adults with hearing loss were separated into",
    "been examined comprehensively. Peritoneal lavage was performed in 351 patients before curative resection of a gastric carcinoma between 1987 and"
]
neg_prompts = []

top_k_indices, top_k_values = prompt_search_mean_local(pos_prompts, neg_prompts, 15, tokenizer, model, model_sae, verbose=True)
top_k_indices, top_k_values = prompt_search_rank(pos_prompts, neg_prompts, 15, tokenizer, model, model_sae, verbose=True)

# %%
gen_txt = influence(45783, 0.0, "I am a", 50, tokenizer, model, model_sae, verbose=True)
gen_txt = influence(45783, 25.0, "I am a", 50, tokenizer, model, model_sae, verbose=True)

# %% [markdown]
# Lying / misinformation feature

# %%
# gen_txt = influence(25398, 35.0, "Q: Which planet is the closest to the Sun? A:", 10, tokenizer, model, model_sae, verbose=True)
gen_txt = influence(25398, 0.0, "Q: Which city is the capital of Switzerland? A:", 1, tokenizer, model, model_sae, verbose=True)
gen_txt = influence(25398, 60.0, "Q: Which city is the capital of Switzerland? A:", 7, tokenizer, model, model_sae, verbose=True)

# %% [markdown]
# Business entities feature

# %%

latent_vector_files = sorted(glob.glob("../sparse_latent_vectors/latent_vectors_batch_*.pt"))
top_k_features = cosine_similarity_search(12219, 25, latent_vector_files, verbose=True)


# %%

ixs = [12219, 33348, 56081, 48378, 49406, 60636, 50974, 17505, 1594, 57862, 7914, 40773, 60113,
 36097, 61739, 40515, 23242, 19980, 102, 44440, 31970, 53915, 50355, 42403, 41106, 2812, 36770, 35616,
 27833, 35098, 48360, 11369, 35009, 40721, 33551, 10336, 27528, 18094, 64382, 14609, 59227, 33009,
 22958, 21864, 26801, 60492, 33656, 48507, 12114, 8488, 1032, 25570, 5518, 41465, 58452, 11799,
 53527, 26536, 48007, 22649, 12428, 43027, 49251, 44218, 8752, 30040, 63443, 285, 39139, 36785,
 15941, 23520, 14523, 21303, 45157, 55401, 30525, 53495, 53931, 4823, 55882, 64993, 43260,
 15610, 54948, 40124, 56572, 26693, 44618, 12067, 41415, 24785, 5712, 23393, 54304, 10463,
 13651, 31034, 32965, 55900]
feature_names = {
    50974: 'Detailed descriptions',
    23393: 'Public infrastructure',
    33348: 'Institution names',
    12219: 'Corporate entities',
    36770: 'Professional societal analysis',
    26536: 'Product advertising tone',
    45157: 'Institutional titles',
    56081: 'Major tech brands',
    48378: 'Notable names',
    49406: 'Technical descriptions',
    60636: 'Family references',
    17505: 'Formal writing style',
    1594: 'Names in legal contexts',
    57862: 'Regulatory terms',
    40773: 'Formal suffix terms',
    60113: 'Corporate references'
}
plot_UMAP(ixs, feature_names, latent_vector_files, save_path=None)


