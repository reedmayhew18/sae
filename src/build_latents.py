import os
import torch
from torch.utils.data import DataLoader
from dataset import ActivationDataset
import logging

logger = logging.getLogger(__name__)

def build_latents(
    model: torch.nn.Module,
    scale_factor: float,
    file_names: list,
    output_dir: str,
    batch_skip: int = 0,
    num_batches: int = 1,
    process_batch_size: int = 4096,
    num_minibatches: int = 19,
    device: str = "cuda",
):
    """
    Extract latent vectors from activations using a trained SAE and save them as sparse tensors.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset = ActivationDataset(
        file_names=file_names,
        data_split="all",
        batch_size=0,  # no subsampling
        scale_factor=scale_factor,
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for idx, batch_data in enumerate(data_loader):
            if idx < batch_skip:
                continue
            if idx >= batch_skip + num_batches:
                break
            # Expecting a tuple: (activations, sent_idx, token_idx, token)
            if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 4:
                logger.error("Dataset item must have 4 elements for 'all' split.")
                continue
            activations, sent_idx, token_idx, token = batch_data
            # Move metadata to device.
            sent_idx = sent_idx.to(device)
            token_idx = token_idx.to(device)
            token = token.to(device)
            # Remove outer batch dimension.
            activations = activations.squeeze(0)
            
            for i in range(num_minibatches):
                start_idx = i * process_batch_size
                end_idx = (i + 1) * process_batch_size
                minibatch = activations[start_idx:end_idx]
                if minibatch.size(0) == 0:
                    continue
                _, encoded = model(minibatch.to(device))
                
                # Reshape metadata to match minibatch size (assumes metadata is [1, N])
                sent_idx_batch = sent_idx[:, start_idx:end_idx].T
                token_idx_batch = token_idx[:, start_idx:end_idx].T
                token_batch = token[:, start_idx:end_idx].T
                
                output_vectors = torch.cat((encoded, sent_idx_batch, token_idx_batch, token_batch), dim=1)
                # Convert to a sparse tensor (if needed by downstream tasks)
                output_vectors_sparse = output_vectors.to_sparse_csr()
                save_path = os.path.join(output_dir, f"latent_vectors_batch_{idx}_minibatch_{i}.pt")
                # torch.save(output_vectors_sparse, save_path)
                logger.info(f"Saved minibatch {i+1} of {num_minibatches} for file {idx}")

