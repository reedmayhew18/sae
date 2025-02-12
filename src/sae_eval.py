import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from autoencoder import SparseAutoencoder
from dataset import ActivationDataset

# Helper to get your file list
def get_file_names(filter_prefix="activations", root_dir="/kaggle/input"):
    file_names = []
    for dirname, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith(filter_prefix):
                file_names.append(os.path.join(dirname, filename))
    return sorted(file_names)

# Evaluation example
def eval_model(model, l1_lambda, file_names, scale_factor):
    test_dataset = ActivationDataset(
        file_names, 
        batch_size=4096,
        f_type="train", 
        # test_fraction=0.01, # last batch file
        test_fraction=0.0, # 12 files == cca 10mil tokens
        scale_factor=scale_factor, 
        seed=123 # different seed that in actual training
    ) # this outputs batches of size 49k - uses 7820MiB VRAM = 95% of GPU
    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # take 1 batch at a time

    criterion = nn.MSELoss().to(device)

    # Run and compute reconstruction error, l1 loss, and total loss
    total_loss = 0; total_mse_loss = 0; total_l1_loss = 0; num_batches = 0
    global_active_mask = torch.zeros((hidden_dim), dtype=torch.bool, device=device)
    for batch in data_loader:
        batch = batch.to(device)
        
        outputs, encoded = model(batch)

        # percent of active features
        # print(encoded.min().item(), encoded.max().item())
        global_active_mask |= torch.any(encoded > 0, dim=1).squeeze(0)
        active_features = torch.any(encoded != 0, dim=1).sum().item()  # Count active features
        total_features = encoded.shape[2]  # Total number of latent features (4096)
        percent_active_features = active_features / total_features
        print(f"Percent Active Features: {percent_active_features * 100:.2f}%")

        mse_loss = criterion(outputs, batch)
        decoder_weight_norms = torch.norm(model.decoder.weight, p=2, dim=0)  # Shape: [num_features]
        l1_terms = encoded * decoder_weight_norms.unsqueeze(0)  # Shape: [batch_size, num_features]
        l1_loss = torch.mean(l1_terms)  # Normalize across both batch size and features
        loss = mse_loss + l1_lambda * l1_loss

        total_loss += loss.item()
        total_mse_loss += mse_loss.item()
        total_l1_loss += l1_loss.item()

        explained_variance = 1 - mse_loss / torch.var(batch)
        # Print batch-level metrics
        print(f"MSE Loss: {mse_loss.item():.4f}, L1 Loss: {l1_loss.item():.4f}, Explained Var: {explained_variance.item():.4f}")
        num_batches += 1

    # Print final metrics
    print(f"Total Test Loss: {total_loss/num_batches:.4f}")
    print(f"Total MSE Loss: {total_mse_loss/num_batches:.4f}")
    print(f"Total L1 Loss: {total_l1_loss/num_batches:.4f}")

    active_features = global_active_mask.sum().item()
    total_features = global_active_mask.numel()
    global_sparsity = (1 - active_features / total_features) * 100
    print(f"Global Sparsity Across All Batches: {global_sparsity:.2f}%")
    print(f"Percent of Active Features: {active_features / total_features * 100:.2f}%")


# Latent extraction example
def build_dataset(model, scale_factor, file_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    dataset = ActivationDataset(
        file_names, 
        batch_size=0, # not subsampled
        f_type="all", 
        scale_factor=scale_factor,
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False) # take 1 batch at a time

    # Extract and save latent vectors
    batch_skip = 4 # 20GB limit on kaggle output
    num_batches = 1
    batch_size = 4096  # Size we can fit in VRAM
    num_minibatches = 19  # 81920/8192 = 10 minibatches per batch
    with torch.no_grad():
        for idx, batch_data in enumerate(data_loader):
            if idx < batch_skip:
                continue
            if idx >= batch_skip+num_batches :
                break
            batch, sent_idx, token_idx, token = batch_data
            sent_idx = sent_idx.to(device)
            token_idx = token_idx.to(device)
            token = token.to(device)
            batch = batch.squeeze(0)  # Remove batch dimension of 1
            
            # Process minibatches and save immediately
            for i in range(num_minibatches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                
                # Get minibatch slice
                minibatch = batch[start_idx:end_idx]
                _, encoded = model(minibatch)
                
                # Stack with metadata
                # Reshape metadata tensors to match batch size
                sent_idx_batch = sent_idx[:,start_idx:end_idx].T
                token_idx_batch = token_idx[:,start_idx:end_idx].T
                token_batch = token[:,start_idx:end_idx].T
                
                output_vectors = torch.cat((encoded, sent_idx_batch, token_idx_batch, token_batch), dim=1)

                # To sparse
                output_vectors = output_vectors.to_sparse_csr()
                
                # Save each minibatch immediately as a PyTorch tensor
                torch.save(output_vectors, f"sparse_latent_vectors/latent_vectors_batch_{idx}_minibatch_{i}.pt")

                print(f"Saved minibatch {i+1} of {num_minibatches} for batch {idx}")


if __name__ == "__main__":
    data_dir = "activations_data"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 3072  
    hidden_dim = 65536

    model = SparseAutoencoder(input_dim, hidden_dim).to(device)
    # model.load_state_dict(torch.load("models/sparse_autoencoder_496.3666.pth"))
    # checkpoint = torch.load("/kaggle/input/checkpoint65k_sae/pytorch/default/1/checkpoint")
    checkpoint = torch.load("models/checkpoint")
    print(checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint['state_dict'])

    # Set model to evaluation mode
    model.eval()

    l1_lambda = 0.00597965  # Regularization strength for sparsity
    scale_factor = 11.888623072966611

    file_names = get_file_names(filter_prefix="activations", root_dir="activations_data")

    # Example usage:
    eval_model(model=model, l1_lambda=l1_lambda, scale_factor=scale_factor, file_names=file_names)
    build_dataset(model=model, scale_factor=scale_factor, file_names=file_names, output_dir="sparse_latent_vectors")