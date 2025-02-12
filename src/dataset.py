import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ActivationDataset(Dataset):
    def __init__(self, file_names, batch_size, f_type, test_fraction=0.01, scale_factor=1.0, seed=42):
        self.batch_size = batch_size
        self.seed = seed

        if f_type in ["train", "test", "all"]:
            self.f_type = f_type
        else:
            raise ValueError("f_type must be 'train' or 'test' or 'all'")
        
        if not 0 <= test_fraction <= 1:
            raise ValueError("test_fraction must be between 0 and 1")
        self.test_fraction = test_fraction

        self.scale_factor = scale_factor
        self.file_names = file_names
        
        split_idx = int(len(self.file_names) * (1 - test_fraction))
        if f_type == "train":
            self.file_names = self.file_names[:split_idx]
        elif f_type == "test":
            self.file_names = self.file_names[split_idx:]
        else: # all
            pass

        print(f"Loaded {len(self.file_names)} batches for {f_type} set")

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        activations = np.load(self.file_names[idx])
        if self.f_type == "all":
            sent_idx = activations[:, -3]
            token_idx = activations[:, -2] 
            token = activations[:, -1]
        # remove last 3 columns (sent_idx, token_idx, and token)
        activations = activations[:, :-3]
        # normalize activations
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        activations = torch.tensor(activations, dtype=torch.float32, device=device)
        # print("Activation Range Before Normalization:", torch.min(activations).item(), torch.max(activations).item())
        activations = activations / self.scale_factor * np.sqrt(activations.shape[1])
        # print("Activation Range After Normalization:", torch.min(activations).item(), torch.max(activations).item())

        if self.f_type == "train":
            # Set seed for reproducibility
            np.random.seed(self.seed)
            # random subsample 8192 examples
            indices = torch.randperm(activations.shape[0], device=activations.device)[:self.batch_size]
            activations = activations[indices]
        
        if self.f_type == "all":
            return activations, sent_idx, token_idx, token
        else:
            return activations


# DataLoader Creation
def create_data_loaders(batch_size, scale_factor):
    train_dataset = ActivationDataset("train", 0.01, scale_factor, batch_size, 42)
    val_dataset = ActivationDataset("test", 0.01, scale_factor, batch_size, 42)
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