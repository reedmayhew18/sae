import logging
from typing import List, Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import get_file_names

logger = logging.getLogger(__name__)

class ActivationDataset(Dataset):
    """
    Dataset for loading and processing activation files.
    
    Args:
        file_names (List[str]): List of file paths.
        data_split (str): Which split to use; one of "train", "test", or "all".
        batch_size (Optional[int]): For subsampling in training. If None or 0, no subsampling is done.
        test_fraction (float): Fraction of files to reserve for testing.
        scale_factor (float): Normalization scale factor.
        seed (int): Seed for reproducibility in subsampling.
    """
    def __init__(
        self,
        file_names: List[str],
        data_split: str,
        batch_size: Optional[int] = None,
        test_fraction: float = 0.01,
        scale_factor: float = 1.0,
        device: str = "cuda",
        seed: int = 42,
    ):
        if data_split not in ("train", "test", "all"):
            raise ValueError("data_split must be 'train', 'test', or 'all'")
        if not (0 <= test_fraction <= 1):
            raise ValueError("test_fraction must be between 0 and 1")
        
        self.data_split = data_split
        self.batch_size = batch_size
        self.test_fraction = test_fraction
        self.scale_factor = scale_factor
        self.device = device
        self.seed = seed

        # Split files if needed
        self.full_file_names = file_names
        self.file_names = self._split_file_names(file_names, data_split, test_fraction)
        logger.info(f"Loaded {len(self.file_names)} files for {data_split} split.")

    def _split_file_names(
        self, file_names: List[str], data_split: str, test_fraction: float
    ) -> List[str]:
        if data_split == "all":
            return file_names
        split_idx = int(len(file_names) * (1 - test_fraction))
        if data_split == "train":
            return file_names[:split_idx]
        else:  # data_split == "test"
            return file_names[split_idx:]

    def __len__(self) -> int:
        return len(self.file_names)
    
    def __getitem__(self, idx: int) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        file_path = self.file_names[idx]
        try:
            data = np.load(file_path)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise
        
        # If using "all", extract metadata (last 3 columns)
        if self.data_split == "all":
            sent_idx = data[:, -3]
            token_idx = data[:, -2]
            token = data[:, -1]
        # Remove metadata columns from activations
        activations = data[:, :-3]
        # Normalize activations (scaling by sqrt of feature dimension)
        activations = activations / self.scale_factor * np.sqrt(activations.shape[1])
        activations_tensor = torch.tensor(activations, dtype=torch.float32, device=self.device)
        
        # For training, subsample a random batch if requested.
        if self.data_split == "train" and self.batch_size and self.batch_size > 0:
            n_examples = activations_tensor.shape[0]
            if n_examples < self.batch_size:
                logger.warning(f"Batch size {self.batch_size} exceeds available examples {n_examples}; using all.")
                indices = torch.arange(n_examples)
            else:
                # For reproducibility, you might want to seed per worker instead.
                indices = torch.randperm(n_examples, device=self.device)[:self.batch_size]
            activations_tensor = activations_tensor[indices]
        
        if self.data_split == "all":
            # Return metadata as tensors too.
            return (activations_tensor,
                    torch.tensor(sent_idx, dtype=torch.float32),
                    torch.tensor(token_idx, dtype=torch.float32),
                    torch.tensor(token, dtype=torch.float32))
        else:
            return activations_tensor


# DataLoader Creation
def create_data_loaders(batch_size, scale_factor):
    file_names = get_file_names("activations", "activations_data")

    train_dataset = ActivationDataset(file_names, "train", batch_size, 0.01, scale_factor, 42)
    val_dataset = ActivationDataset(file_names, "test", batch_size, 0.01, scale_factor, 42)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)#, num_workers=3, pin_memory=True, persistent_workers=True)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Keep 1 as outer batch size for per-file sampling
        shuffle=False,
        num_workers=3,  # Adjust based on CPU availability
        pin_memory=False,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,  # Smaller for validation
        pin_memory=False,
    )
    return train_loader, val_loader