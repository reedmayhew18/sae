import pandas as pd
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from IPython.display import display, HTML

def examples_plot_df(one_example_dict):
    clean_tokens_list = one_example_dict['token_list']
    np_activations = np.array(one_example_dict['activations'])
    
    df = pd.DataFrame({
        'Token': clean_tokens_list,
        'Activation': np_activations
    })

    v_max = np.max(np_activations)
    # Display DataFrame with color gradient based on activation values
    display(df.style.background_gradient("coolwarm", subset='Activation', vmin=-v_max, vmax=v_max))


def examples_plot_df_horizontal(one_example_dict):
    tokens = one_example_dict['token_list']
    activations = np.array(one_example_dict['activations'])

    # Build dataframe, then transpose so each token is a column
    df = pd.DataFrame({"Token": tokens, "Activation": activations}).T
    # Make columns unique
    df.columns = [f"{col}_{i}" for i, col in enumerate(df.loc["Token"])]
    # Drop the "Token" row
    df = df.drop("Token")

    # Determine symmetric range for colormap
    max_val = activations.max()
    vmin, vmax = -max_val, max_val

    # Define the coloring function
    def color_tokens(val):
        # Use reversed coolwarm so that negative -> red, zero -> white, positive -> blue
        cmap = cm.get_cmap("coolwarm")
        norm = Normalize(vmin=vmin, vmax=vmax)
        rgba = cmap(norm(val))
        return f"background-color: rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, 1); color: black;"

    styled_df = df.style.applymap(color_tokens)
    display(HTML(styled_df.to_html()))


def examples_plot_anthropic(feat_idx, examples_dict, save_path=None):
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