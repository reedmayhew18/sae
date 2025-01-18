import torch
import torch.nn as nn

#TODO: SAE will be moved to a separate file
scale_factor = 11.888623072966611
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SparseAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim)
        # Decoder
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded

import glob
import torch
import numpy as np
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.2-3B"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def _get_topk_activations(feat_idx, k, latent_vector_files):
    #TODO: change to min-heap (sorting is expensive)
    # First loop - get top k activations across all files
    top_k = []
    for latent_file in latent_vector_files:
        loaded_vectors = torch.load(latent_file)
        values, ind = torch.topk(loaded_vectors[:, feat_idx], k)
        
        # Add all values from this file
        for idx, val in zip(ind, values):
            top_k.append((idx.item(), val.item(), latent_file))
        
        # Keep only top k overall after processing each file
        top_k.sort(key=lambda x: x[1], reverse=True)
        if len(top_k) > k:
            top_k = top_k[:k]

    minibatch_size = loaded_vectors.shape[0]

    return top_k, minibatch_size

def _get_topk_context(feat_idx, k, top_k, context_len, minibatch_size, act_id_dataset, tokenizer, verbose=True):
    # Second loop - get context for top k activations
    if verbose:
        print(f"Top {k} activations for feature index {feat_idx} across all files:")
    
    examples_dict = {}
    for i, (idx, val, latent_fname) in enumerate(top_k):
        # Determine the corresponding last3 file
        batch_number = int(latent_fname.split('_batch_')[1].split('_')[0])
        minibatch_number = int(latent_fname.split('minibatch_')[1].split('.')[0])
        idx = idx + minibatch_number * minibatch_size # adjusting the index to a global offset
        last3_fname = act_id_dataset.replace('*', f"{batch_number:04d}")

        # Load the last3 file
        last3_data = np.load(last3_fname)

        # Get sentence index and token index for target token
        sent_idx, tok_idx, token = last3_data[idx]  # sent_idx, token_idx, token

        # Get context window indices, staying within the same sentence
        start_idx = max(0, idx - context_len)
        end_idx = min(len(last3_data), idx + context_len + 1)

        # Adjust window to stay within the same sentence
        context_window = last3_data[start_idx:end_idx]
        context_window = context_window[context_window[:, 0] == sent_idx]  # Match sent_idx

        # Extract tokens and activations
        lt = torch.load(latent_fname)
        min_tk_i = context_window[0, 1]
        max_tk_i = context_window[-1, 1]
        context = lt[(lt[:, -2] >= min_tk_i) & (lt[:, -2] <= max_tk_i) & (lt[:, -3] == sent_idx)]
        context_activations = context[:, feat_idx]

        # Decode tokens
        context_tokens = context_window[:, 2]
        # text = tokenizer.decode(context_tokens.tolist())
        decoded_tokens = tokenizer.convert_ids_to_tokens(context_tokens.tolist())
        target_token_txt = tokenizer.decode([int(token)])

        # Print details
        clean_tokens_list = [token.replace('Ġ', ' ') for token in decoded_tokens]
        np_activations = context_activations.cpu().numpy()
        text = ' '.join(clean_tokens_list)

        if verbose:
            print(f"File: {latent_fname}")
            print(f"Activation value: {val}") # {val:.4f}")
            print(f"Target token: '{target_token_txt}'")# (index {rel_idx})")
            text1 = text.replace('\n', ' ')
            print(f"Context: {text1}")
            print("")
        
        examples_dict[i] = {
            "file": latent_fname,
            "target_token": target_token_txt,
            "target_token_value": val,
            "context": text,
            "token_list": clean_tokens_list,
            "activations": np_activations.tolist()
        }
    
    return examples_dict

def top_activations(feat_idx, k, context_len, latent_dataset_path, dataset_slice, act_id_dataset, tokenizer, verbose=True):
    
    a,b = dataset_slice
    latent_vector_files = sorted(glob.glob(latent_dataset_path))[a:b]

    top_k, minibatch_size = _get_topk_activations(feat_idx, k, latent_vector_files)

    examples_dict = _get_topk_context(feat_idx, k, top_k, context_len, minibatch_size, act_id_dataset, tokenizer, verbose)

    return examples_dict


examples_dict = top_activations(
    feat_idx=53475,
    k=20,
    context_len=15,
    latent_dataset_path="sparse_latent_vectors/latent_vectors_batch_*.pt",
    dataset_slice=(0,10),
    act_id_dataset="activations_data/last3_batch_*.npy",
    tokenizer=tokenizer,
    verbose=False
)

indicies = [58517, 8306, 26908, 50988, 53475]
top_examples = {}
for idx in indicies:
    top_examples[idx] = top_activations(idx, 20, 15, "sparse_latent_vectors/latent_vectors_batch_*.pt", (0,10), "activations_data/last3_batch_*.npy", tokenizer, verbose=False)

def examples_plot_df(one_example_dict):
    clean_tokens_list = one_example_dict['token_list']
    np_activations = np.array(one_example_dict['activations'])
    
    import pandas as pd
    df = pd.DataFrame({
        'Token': clean_tokens_list,
        'Activation': np_activations
    })

    v_max = np.max(np_activations)
    # Display DataFrame with color gradient based on activation values
    from IPython.display import display
    display(df.style.background_gradient("coolwarm", subset='Activation', vmin=-v_max, vmax=v_max))

examples_plot_df(top_examples[58517][0]) # Plots the top example for feature 58517


def examples_plot_df_horizontal(one_example_dict):
    clean_tokens_list = one_example_dict['token_list']
    np_activations = np.array(one_example_dict['activations'])

    # Create a DataFrame for tokens and their activations
    import pandas as pd
    data = pd.DataFrame({
        "Token": clean_tokens_list,
        "Activation": np_activations
    })

    # Transpose the DataFrame
    data = data.T
    data.columns = data.loc["Token"]  # Use tokens as the column headers
    data = data.drop("Token")  # Drop the redundant 'Token' row

    # Generate HTML with colored backgrounds
    def color_tokens(val):
        from matplotlib import cm
        cmap = cm.get_cmap("coolwarm")
        rgba_color = cmap(val)
        background_color = f"rgba({int(rgba_color[0] * 255)}, {int(rgba_color[1] * 255)}, {int(rgba_color[2] * 255)}, 1)"
        return f"background-color: {background_color}; color: black;"

    # Apply styling
    styled_data = data.style.applymap(color_tokens)

    # Display the styled DataFrame
    from IPython.display import display, HTML
    display(HTML(styled_data.to_html()))

examples_plot_df(top_examples[58517][0])


def examples_plot_anthropic(feat_idx, examples_dict, save_path=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.cm as cm

    topkexamples = [(ex["token_list"], ex["activations"]) for ex in examples_dict.values()] # TODO: maybe activation to numpy
    topk = len(topkexamples)

    # Plotting top k examples
    fig, ax = plt.subplots(figsize=(15, topk))  # Adjust height dynamically
    plt.title(f"Token Activations for Top {topk} Examples of Feature {feat_idx}")

    max_val = np.max(topkexamples[0][1])
    min_val = 0.0

    for example_idx, (tokens, activations) in enumerate(topkexamples):
        # Normalize activations for color mapping
        norm = Normalize(vmin=min_val, vmax=max_val)
        cmap = cm.Reds  # Use a red colormap

        x_pos = 0.0  # Initialize horizontal position for tokens
        renderer = fig.canvas.get_renderer()  # Renderer for accurate text measurements

        for token, activation in zip(tokens, activations):
            # Map activation value to a color
            color = cmap(norm(activation))

            # Add the token to the plot with a colored background
            text_obj = ax.text(
                x_pos, -example_idx, token,
                fontsize=12, color="black", ha="left", va="center",
                bbox=dict(facecolor=color, edgecolor="none", boxstyle="round,pad=0.3")
            )

            # Use the text object's bounding box to calculate its width
            text_bbox = text_obj.get_window_extent(renderer=renderer)
            text_width = text_bbox.width / fig.dpi  # Convert from pixels to inches
            text_width /= 11.5  # Normalize to reduce excessive spacing

            # Update x-position for the next token, with a small gap
            x_pos += text_width + 0.013  # Add a small gap between tokens

    # Add a horizontal colorbar at the top right
    cbar_ax = fig.add_axes([0.98, 0.92, 0.2, 0.03])  # [left, bottom, width, height]
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Activation Value", labelpad=0)

    # Adjust plot aesthetics
    ax.axis('off')
    ax.set_xlim(0, x_pos)  # Set x-limits to prevent overflow
    ax.set_ylim(-topk, 0.5)
    fig.subplots_adjust(top=0.85, bottom=0.1, left=0.05, right=1.34)

    # save png
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    # Show the plot
    plt.show()

examples_plot_anthropic(58517, top_examples[58517], save_path=None)


import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Model and tokenizer setup
model_name = "meta-llama/Llama-3.2-3B"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config).to(device)

activation_cache = []
def save_activations_hook(module, input, output):
    # input is a tuple; input[0] is the tensor we need
    activation_cache.append(input[0].cpu().detach().numpy())

# Register hook on the 16th layer
layer_index = 15  # Zero-based index; 15 corresponds to the 16th layer
hook_handle = model.model.layers[layer_index].register_forward_hook(save_activations_hook) 

# Load the sparse autoencoder model and weights
input_dim = 3072  
hidden_dim = 2 ** 16 # 65536
model_sae = SparseAutoencoder(input_dim, hidden_dim)
# model_sae.load_state_dict(torch.load("sparse_autoencoder.pth"))
checkpoint = torch.load("models/checkpoint")
model_sae.load_state_dict(checkpoint['state_dict'])
# model_sae.eval()


def extract_features(positive_prompts, negative_prompts, tokenizer, llm_model, sae_model):
    prompts = positive_prompts + negative_prompts

    # Tokenize sentences
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=35, # can equal to the length (in tokens) of the longest sentence
                    # limit suggested due to VRAM constraints
        padding="max_length",
    ).to(device)

    activation_cache = []
    # Forward pass
    with torch.no_grad():
        llm_model(**inputs)

    activations = np.array(activation_cache)
    print(f"Activations shape: {activations.shape}") # (1, num_sent, seq_len, 3072)

    # Reshape activations to (num_sent*seq_len, 3072)
    activations = activations.squeeze(0)  # Remove batch dimension
    num_sent, seq_len, hidden_dim = activations.shape
    activations = activations.reshape(-1, hidden_dim)

    # Build index tensor for sentence separation
    sent_idx_tensor = torch.tensor([[i] * seq_len for i in range(num_sent)]).flatten()

    # Get attention mask and remove padding tokens
    attention_mask = inputs['attention_mask'].view(-1).cpu().numpy()
    activations = activations[attention_mask == 1]
    sent_idx_tensor = sent_idx_tensor[attention_mask == 1]

    activations = torch.tensor(activations, dtype=torch.float32)  # Convert to torch tensor

    # Get latent vectors for sentences
    with torch.no_grad():
        _, encoded = sae_model(activations)
        latent_vector = encoded.cpu().numpy()
    
    return latent_vector, sent_idx_tensor


def prompt_search_mean_local(positive_prompts, negative_prompts, top_k, tokenizer, llm_model, sae_model, verbose=True):
    latent_vector, sent_idx_tensor = extract_features(positive_prompts, negative_prompts, tokenizer, llm_model, sae_model)

    # Positive prompts
    pos_ix = np.where(sent_idx_tensor < len(positive_prompts))[0]
    pos_latent = latent_vector[pos_ix]
    # Apply min-max normalization to latent vectors
    latent_vector_normalized = (pos_latent - np.min(pos_latent, axis=0)) / (np.max(pos_latent, axis=0) - np.min(pos_latent, axis=0) + 1e-8)
    # Get mean activation for each feature across tokens
    mean_activation = np.mean(latent_vector_normalized, axis=0)

    # Negative prompts
    if len(negative_prompts) > 0:
        neg_ix = np.where(sent_idx_tensor >= len(positive_prompts))[0]
        neg_latent = latent_vector[neg_ix]
        # Apply min-max normalization to latent vectors
        neg_latent_normalized = (neg_latent - np.min(neg_latent, axis=0)) / (np.max(neg_latent, axis=0) - np.min(neg_latent, axis=0) + 1e-8)
        # Get mean activation for each feature across tokens
        neg_mean_activation = np.mean(neg_latent_normalized, axis=0)
        # Subtract negative mean activation
        mean_activation -= neg_mean_activation

    # Get top k features and their mean activation values
    top_k = 15
    top_k_indices = np.argsort(mean_activation)[-top_k:][::-1]
    top_k_values = mean_activation[top_k_indices]

    if verbose:
        print(f"Top {top_k} features and their mean activation values:")
        for idx, val in zip(top_k_indices, top_k_values):
            print(f"Feature {idx}: {val:.4f}")
    
    return top_k_indices, top_k_values

def prompt_search_rank(positive_prompts, negative_prompts, top_k, tokenizer, llm_model, sae_model, vebose=True):
    
    latent_vector, sent_idx_tensor = extract_features(positive_prompts, negative_prompts, tokenizer, llm_model, sae_model)

    pos_ix = np.where(sent_idx_tensor < len(positive_prompts))[0]
    pos_latent = latent_vector[pos_ix]
    neg_ix = np.where(sent_idx_tensor >= len(positive_prompts))[0]
    neg_latent = latent_vector[neg_ix]

    # 1. Choose how large the "top slice" is. 
    #    For example, top_frac = 0.05 means top 5% of features for each token.
    #    Or use top_k = 100 to pick top 100 features per token.
    top_frac = 0.05 # at 65k features, this is 3250 features
    num_features = pos_latent.shape[1]  # same as neg_latent.shape[1]

    top_k = int(num_features * top_frac)  # number of features to pick per token
    top_k = max(top_k, 1)                # ensure at least 1

    # 2. Count how often each feature appears in that top slice for positive vs. negative.

    pos_count = np.zeros(num_features, dtype=np.int64)
    for row in pos_latent:
        # row shape: (num_features,)
        # get indices of top_k features in that row
        # argsort returns ascending, so we take last top_k
        top_indices = np.argpartition(row, -top_k)[-top_k:]
        # increment counts
        pos_count[top_indices] += 1

    neg_count = np.zeros(num_features, dtype=np.int64)
    for row in neg_latent:
        top_indices = np.argpartition(row, -top_k)[-top_k:]
        neg_count[top_indices] += 1

    # 3. Convert counts to frequencies (how often a feature is in top_k%).
    pos_freq = pos_count / pos_latent.shape[0]
    neg_freq = neg_count / neg_latent.shape[0] if len(negative_prompts) > 0 else neg_count / 1.0

    # 4. Score = difference or ratio, e.g. difference:
    score = pos_freq - neg_freq

    # 5. Sort by score descending
    sorted_features = np.argsort(score)[::-1]
    topN = 15
    for i in range(topN):
        f_idx = sorted_features[i]
        print(f"Feature {f_idx} => pos_freq={pos_freq[f_idx]:.4f}, neg_freq={neg_freq[f_idx]:.4f}, score={score[f_idx]:.4f}")
    
    return sorted_features[:topN], score[sorted_features[:topN]]


def influence(feat_idx, multiplier, start_sent, num_tok, tokenizer, llm_model, sae_model, verbose=True):
    # Generate artificial latent vector, pass through SAE decoder, and boost it

    # Boosted activations setup
    artificial_latent_vector = np.zeros(hidden_dim)
    artificial_latent_vector[feat_idx] = 1

    # if not 0 print
    if verbose:
        if multiplier != 0:
            print(f"{multiplier} boost on feature {feat_idx}")
        else:
            print("No boost")

    # Decode to get activations
    with torch.no_grad():
        artificial_latent_vector_tensor = torch.tensor(artificial_latent_vector, dtype=torch.float32).unsqueeze(0)
        reconstructed_activations = sae_model.decoder(artificial_latent_vector_tensor)
        boosted_activations = reconstructed_activations.to(torch.float16) * multiplier

    # Hook function to add boosted activations to all tokens
    def influence_hook(module, input, output):
        if isinstance(output, tuple):  # Handle tuple output
            boosted_output = output[0] + boosted_activations.to(device)
            return (boosted_output,) + output[1:]
        else:
            return output + boosted_activations.to(device)

    layer_index = 15  # Target layer for injection

    try:
        # Register the hook
        hook_handle = llm_model.model.layers[layer_index].register_forward_hook(influence_hook)

        # Text generation loop
        inputs = tokenizer(start_sent, return_tensors="pt", add_special_tokens=False).to(device)
        input_ids = inputs["input_ids"]

        generated_ids = input_ids.clone()

        for _ in range(num_tok):
            with torch.no_grad():
                outputs = llm_model(input_ids=generated_ids)

                # Maximum likelihood sampling
                next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0) 

                # Stop generation if EOS or padding token is predicted
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
                
                # Append the predicted token ID
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        # Decode the generated sequence
        generated_text = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)
    finally:
        # Remove the hook to prevent side effects
        hook_handle.remove()
    
    if verbose:
        print(f"Generated text with influence: '{generated_text}'")
    return generated_text

gen_txt = influence(58517, 25.0, "I am a", 50, tokenizer, model, model_sae, verbose=True)

import heapq
def cosine_similarity_search(feat_idx, top_k, latent_vector_files, verbose=True):
    # Dictionary to accumulate global similarity scores for each feature
    global_similarities = {}

    # Process each chunk
    for latent_file in latent_vector_files:
        loaded_vectors = torch.load(latent_file)[:,:-3]  # Load current chunk

        # Get the column corresponding to the target feature index
        target_feature = loaded_vectors[:, feat_idx]

        # Compute similarity with all other features (columns)
        similarities = torch.nn.functional.cosine_similarity(
            target_feature.unsqueeze(0),  # Make it 2D for broadcasting
            loaded_vectors.T,  # Transpose for feature-wise comparison
            dim=1
        )

        # Accumulate similarity scores globally
        for i, similarity in enumerate(similarities):
            if i not in global_similarities:
                global_similarities[i] = 0.0
            global_similarities[i] += similarity.item()

    # Identify the top-k features globally
    top_k_features = heapq.nlargest(top_k, global_similarities.items(), key=lambda x: x[1])

    if verbose:
        # Display results
        for idx, agg_similarity in top_k_features[:25]:
            print(f"Feature {idx} has aggregated similarity {agg_similarity:.4f}")

    return top_k_features  # Returns a list of (feature_index, aggregated_similarity) tuples

latent_vector_files = sorted(glob.glob("sparse_latent_vectors/latent_vectors_batch_*.pt"))
top_k_features = cosine_similarity_search(12219, 25, latent_vector_files, verbose=True)

import umap
import matplotlib.pyplot as plt
def plot_UMAP(feat_ixs, feature_names, latent_vector_files, save_path=None):
    # Load and concatenate batches
    latent_vectors = []
    for batch_file in latent_vector_files:
        batch_vectors = torch.load(batch_file).cpu().numpy()
        batch_vectors = batch_vectors[:, feat_ixs]
        latent_vectors.append(batch_vectors)
    latent_vectors = np.concatenate(latent_vectors, axis=0)

    # Remove last 3 columns (sent_idx, tok_idx, token)
    latent_vectors = latent_vectors[:, :-3]

    print(f"Loaded {len(latent_vector_files)} batches, total shape: {latent_vectors.shape}")

    # UMAP dimensionality reduction
    umap_embedder = umap.UMAP(n_components=2, metric='cosine')
    latent_vectors_2d = umap_embedder.fit_transform(latent_vectors.T)

    # Plot
    plt.figure(figsize=(15, 15))
    plt.scatter(latent_vectors_2d[:, 0], latent_vectors_2d[:, 1], s=10, alpha=0.5)

    # Add labels
    for i, (x, y) in enumerate(latent_vectors_2d):
        feature_idx = feat_ixs[i]
        label = feature_names.get(feature_idx, f"{feature_idx}") # f"Feature: {feature_idx}"
        plt.text(x, y, label, fontsize=10, alpha=0.7, rotation=0)

    plt.title(f'UMAP projection of {len(latent_vector_files)} batches of latent vectors', fontsize=14)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

ixs = [12219, 33348, 56081, 48378, 49406, 60636, 50974, 17505, 1594, 57862]
feature_names = {
    12219: 'Corporate entities',
    33348: 'Institution names',
    56081: 'Major tech brands',
    48378: 'Notable names',
    49406: 'Technical descriptions'
}
plot_UMAP(ixs, feature_names, latent_vector_files, save_path=None)