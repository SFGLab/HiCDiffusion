import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from lightning.pytorch.utilities import grad_norm
import math

print_sizes = False
class Interaction3DPredictor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.example_input_array = torch.Tensor(16, 5, int(math.pow(2, 20)))

        self.conv_blocks = nn.ModuleList([])
        self.conv_trans_blocks = nn.ModuleList([])
        channels_in = [5, 32, 32, 32, 64, 64, 64, 128, 128, 256, 256, 256, 256]
        channels_out = [32, 32, 32, 64, 64, 64, 128, 128, 256, 256, 256, 256, 256]
        
        for i in range(0, 12):
                self.conv_blocks.append(nn.Sequential(nn.Conv1d(channels_in[i], channels_out[i], 3, padding=1),
                                                nn.ReLU(),
                                                nn.MaxPool1d(2)))
        decoder_channels_in = [256, 128, 68, 16]
        decoder_channels_out = [128, 68, 16, 8]
        for i in range(0, 4):
            self.conv_trans_blocks.append(nn.Sequential(nn.ConvTranspose2d(decoder_channels_in[i], decoder_channels_out[i], 2, stride=2),
                                                nn.ReLU()))
        self.conv_trans_blocks.append(nn.Sequential(nn.Conv2d(8, 1, 59, padding=1),
                                                nn.ReLU()))
    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = x.view(-1, 256, 16, 16)
        for block in self.conv_trans_blocks:
            x = block(x)
        x = x.view(-1, 200, 200)
        return x

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        mae = torch.nn.L1Loss()
        mse = torch.nn.MSELoss()
        self.log("train_loss", loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_mae", mae(y_hat, y), sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_mse", mse(y_hat, y), sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def on_train_epoch_end(self):
        print('\n')

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        mae = torch.nn.L1Loss()
        mse = torch.nn.MSELoss()
        self.log("val_mae", mae(y_hat, y), sync_dist=True, prog_bar=True)
        self.log("val_mse", mse(y_hat, y), sync_dist=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)