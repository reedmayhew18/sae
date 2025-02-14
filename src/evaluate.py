import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ActivationDataset

logger = logging.getLogger(__name__)

def eval_model(
    model: nn.Module,
    l1_lambda: float,
    file_names: list,
    scale_factor: float,
    batch_size: int = 4096,
    test_fraction: float = 0.0,
    device: str = "cuda",
):
    """
    Evaluate the model on a dataset constructed from file_names.
    """
    # Use the "train" split if test_fraction is zero; adjust as needed.
    dataset = ActivationDataset(
        file_names=file_names,
        data_split="train",
        batch_size=batch_size,
        test_fraction=test_fraction,
        scale_factor=scale_factor,
        device=device,
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    criterion = nn.MSELoss().to(device)
    total_loss = 0.0
    total_mse_loss = 0.0
    total_l1_loss = 0.0
    num_batches = 0
    
    # Determine hidden_dim from model
    hidden_dim = model.decoder.weight.shape[1]
    global_active_mask = torch.zeros(hidden_dim, dtype=torch.bool, device=device)
    
    for batch in data_loader:
        # If dataset returns a tuple (when data_split == "all"), take only activations.
        # if isinstance(batch, (tuple, list)):
        #     batch = batch[0]
        batch = batch.to(device)
        outputs, encoded = model(batch)

        global_active_mask |= torch.any(encoded > 0, dim=1).squeeze(0)

        mse_loss = criterion(outputs, batch)
        decoder_weight_norms = torch.norm(model.decoder.weight, p=2, dim=0)
        l1_terms = encoded * decoder_weight_norms.unsqueeze(0)
        l1_loss = torch.mean(l1_terms)

        loss = mse_loss + l1_lambda * l1_loss

        total_loss += loss.item()
        total_mse_loss += mse_loss.item()
        total_l1_loss += l1_loss.item()

        explained_variance = 1 - mse_loss / torch.var(batch)
        logger.info(
            f"MSE Loss: {mse_loss.item():.4f}, L1 Loss: {l1_loss.item():.4f}, Explained Var: {explained_variance.item():.4f}"
        )
        num_batches += 1

    logger.info(f"Total Test Loss: {total_loss/num_batches:.4f}")
    logger.info(f"Total MSE Loss: {total_mse_loss/num_batches:.4f}")
    logger.info(f"Total L1 Loss: {total_l1_loss/num_batches:.4f}")

    active_features = global_active_mask.sum().item()
    total_features = global_active_mask.numel()
    global_sparsity = (1 - active_features / total_features) * 100
    logger.info(f"Global Sparsity Across All Batches: {global_sparsity:.2f}%")
    logger.info(f"Percent of Active Features: {active_features / total_features * 100:.2f}%")

