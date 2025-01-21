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

examples_dict = top_activations(
    feat_idx=53475,
    k=20,
    context_len=15,
    latent_dataset_path="../sparse_latent_vectors/latent_vectors_batch_*.pt",
    dataset_slice=(0,10),
    act_id_dataset="../activations_data/last3_batch_*.npy",
    tokenizer=tokenizer,
    verbose=False
)
print(json.dumps(examples_dict[0], indent=2))#, sort_keys=True))

indicies = [58517, 8306, 26908, 50988, 53475]
top_examples = {}
for idx in indicies:
    top_examples[idx] = top_activations(idx, 20, 15, "../sparse_latent_vectors/latent_vectors_batch_*.pt", (0,10), "../activations_data/last3_batch_*.npy", tokenizer, verbose=False)

print("Top 1 example for feat 53475")
print(json.dumps(top_examples[53475][0], indent=2))


# %%
examples_plot_df(top_examples[53475][0]) # Plots the top example for feature 58517

# %%
examples_plot_df_horizontal(top_examples[53475][0])

# %%
feat_idx = 53475
top5_examples = {}
top5_examples[feat_idx] = {}
for i in range(5):
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

# %%
gen_txt = influence(45783, 25.0, "I am a", 50, tokenizer, model, model_sae, verbose=True)

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


