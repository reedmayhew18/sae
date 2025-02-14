import logging
import torch
from autoencoder import SparseAutoencoder
from utils import get_file_names
from evaluate import eval_model
from build_latents import build_latents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = 3072  
    hidden_dim = 65536
    scale_factor = 11.888623072966611
    l1_lambda = 0.00597965

    # Initialize model and load checkpoint.
    model = SparseAutoencoder(input_dim, hidden_dim)
    # checkpoint_path = "models/checkpoint"  # Adjust this path as needed.
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # logger.info("Loaded checkpoint hyperparameters: %s", checkpoint.get("hyper_parameters", "N/A"))
    # model.load_state_dict(checkpoint["state_dict"])
    # model_path = "models/sparse_autoencoder_496.3666.pth"
    model_path = "models/sparse_autoencoder_65kbest.pth"
    weights = torch.load(model_path, weights_only=True)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    # Retrieve file names.
    file_names = get_file_names(filter_prefix="activations", root_dir="activations_data")

    # Evaluate the model.
    eval_model(
        model=model,
        l1_lambda=l1_lambda,
        file_names=file_names,
        scale_factor=scale_factor,
        batch_size=2048,
        test_fraction=0.0,
        device=device,
    )

    # Extract latent vectors.
    build_latents(
        model=model,
        scale_factor=scale_factor,
        file_names=file_names,
        output_dir="sparse_latent_vectors",
        batch_skip=0,
        num_batches=1,
        process_batch_size=1024,
        num_minibatches=19,
        device=device,
    )

if __name__ == "__main__":
    main()
