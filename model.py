import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from lightning.pytorch.utilities import grad_norm
import math

class ResidualConv2d(nn.Module):
    def __init__(self, hidden_in, hidden_out, kernel, padding):
        super(ResidualConv2d, self).__init__()
        self.main = nn.Sequential(
                                    nn.Conv1d(hidden_in, hidden_out, kernel, padding=padding),
                                    nn.BatchNorm1d(hidden_out),
                                    nn.ReLU(),
                                    nn.Conv1d(hidden_out, hidden_out, kernel, padding=padding),
                                    nn.BatchNorm1d(hidden_out),
                                    nn.MaxPool1d(2)
                                    )
        self.downscale = nn.Sequential(nn.Conv1d(hidden_in, hidden_out, kernel, padding=padding),
                                            nn.MaxPool1d(2))
        self.relu = nn.ReLU()
    def forward(self, x):
        #return self.relu(self.main(x))
        residual = self.downscale(x)
        output = self.main(x)

        return self.relu(output+residual) 
    

class Interaction3DPredictor(pl.LightningModule):
    def __init__(self, batch_size):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(16, 5, int(math.pow(2, 20)))
        self.batch_size = batch_size

        self.conv_blocks = nn.ModuleList([])
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.relu = nn.ReLU()
        self.conv_trans_blocks = nn.ModuleList([])

        channels_in = [5, 32, 32, 32, 64, 64, 64, 128, 128, 256, 256, 256, 256]
        channels_out = [32, 32, 32, 64, 64, 64, 128, 128, 256, 256, 256, 256, 256]
        
        for i in range(0, 12):
            self.conv_blocks.append(ResidualConv2d(channels_in[i], channels_out[i], 3, 1))
        decoder_channels_in = [256, 128, 64, 16]
        decoder_channels_out = [128, 64, 16, 8]
        for i in range(0, 4):
            self.conv_trans_blocks.append(nn.Sequential(nn.ConvTranspose2d(decoder_channels_in[i], decoder_channels_out[i], 2, stride=2),
                                                nn.BatchNorm2d(decoder_channels_out[i]),
                                                nn.ReLU()))
        self.conv_trans_blocks.append(nn.Sequential(nn.Conv2d(8, 1, 59, padding=1)))
    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)

        res_transformer = x

        x = self.transformer_encoder(x)

        x = self.relu(x+res_transformer)

        x = x.view(-1, 256, 16, 16)
        for block in self.conv_trans_blocks:
            x = block(x)
        x = x.view(-1, 200, 200)
        return x

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = torch.nn.MSELoss()
        mae = torch.nn.L1Loss()
        mse = torch.nn.MSELoss()
        self.log("train_loss", loss(y_hat, y), sync_dist=True, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("train_mae", mae(y_hat, y), sync_dist=True, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("train_mse", mse(y_hat, y), sync_dist=True, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return loss(y_hat, y)
    
    def on_train_epoch_end(self):
        print('\n')

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = torch.nn.MSELoss()
        mae = torch.nn.L1Loss()
        mse = torch.nn.MSELoss()
        self.log("val_loss", loss(y_hat, y), sync_dist=True, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("val_mae", mae(y_hat, y), sync_dist=True, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("val_mse", mse(y_hat, y), sync_dist=True, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)