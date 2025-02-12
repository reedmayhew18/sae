import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from autoencoder import SparseAutoencoder

class LightningSparseAutoencoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, l1_lambda, lr):
        super().__init__()
        # Use the common SAE as a submodule
        self.model = SparseAutoencoder(input_dim, hidden_dim)
        self.l1_lambda = l1_lambda
        self.lr = lr
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, batch, decoded, encoded):
        mse_loss = self.criterion(decoded, batch)
        # Compute L1 loss on the encoded layer using decoder weight norms
        decoder_weight_norms = torch.norm(self.model.decoder.weight, p=2, dim=0)
        l1_terms = encoded * decoder_weight_norms.unsqueeze(0)
        l1_loss = torch.mean(torch.sum(l1_terms, dim=1))
        return mse_loss, l1_loss

    def training_step(self, batch, batch_idx):
        # Assume batch is already on the proper device
        decoded, encoded = self(batch)
        mse_loss, l1_loss = self.compute_loss(batch, decoded, encoded)
        total_loss = mse_loss + self.l1_lambda * l1_loss
        self.log("train_loss", total_loss, on_step=True, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        decoded, encoded = self(batch)
        mse_loss, l1_loss = self.compute_loss(batch, decoded, encoded)
        total_loss = mse_loss + self.l1_lambda * l1_loss
        self.log("val_loss", total_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
