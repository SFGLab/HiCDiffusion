import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from lightning.pytorch.utilities import grad_norm
import math
import matplotlib.pyplot as plt
import numpy as np
import shutil
import pyranges as pr
import pandas as pd
import matplotlib
import wandb
from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
from torchmetrics import MetricCollection

channels_in = [5, 32, 32, 32, 64, 64, 64, 128, 128, 256, 256, 256, 256]
channels_out = [32, 32, 32, 64, 64, 64, 128, 128, 256, 256, 256, 256, 256]

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList([])
        for i in range(0, 13):
            self.blocks.append(ResidualConv1d(channels_in[i], channels_out[i], 3, 1))

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.final_block = nn.Sequential(nn.Conv1d(256, 1, 1))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.transformer_encoder(x)
        x = self.final_block(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.blocks = nn.ModuleList([])

        self.first_block = nn.Sequential(nn.Conv1d(1, 256, 1))
        for i in range(0, 13):
            self.blocks.append(ResidualTransConv1d(channels_out[-i-1], channels_in[-i-1], 3, 1))

    def forward(self, x):
        x = self.first_block(x)
        for block in self.blocks:
            x = block(x)
        return x

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

class ResidualTransConv1d(nn.Module):
    def __init__(self, hidden_in, hidden_out, kernel, padding):
        super(ResidualTransConv1d, self).__init__()
        self.main = nn.Sequential(
                                    nn.Conv1d(hidden_in, hidden_out, kernel, padding=padding),
                                    nn.BatchNorm1d(hidden_out),
                                    nn.ReLU(),
                                    nn.Conv1d(hidden_out, hidden_out, kernel, padding=padding),
                                    nn.BatchNorm1d(hidden_out),
                                    nn.Upsample(scale_factor=2, mode='linear')
                                    )
        self.upscale = nn.Sequential(nn.Conv1d(hidden_in, hidden_out, kernel, padding=padding),
                                            nn.Upsample(scale_factor=2, mode='linear'))
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = self.upscale(x)
        output = self.main(x)

        return self.relu(output+residual) 

class ResidualConv1d(nn.Module):
    def __init__(self, hidden_in, hidden_out, kernel, padding):
        super(ResidualConv1d, self).__init__()
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
        residual = self.downscale(x)
        output = self.main(x)

        return self.relu(output+residual) 
    
class Interaction3DPredictorSequenceVAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.example_input_array = torch.Tensor(2, 5, int(math.pow(2, 21)))

        self.model = VAE()
        
        metrics = MetricCollection([ MeanAbsoluteError(), MeanAbsolutePercentageError(), MeanSquaredError()
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
    
    def forward(self, x):

        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _, _ = batch

        x_pred = self(x)

        loss = torch.nn.L1Loss()

        self.log("train_loss", loss(x_pred, x), on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)
        self.log_dict(self.train_metrics(x_pred.view(-1), x.view(-1)), sync_dist=True, batch_size=x.shape[0])

        return loss(x_pred, x)
    
    def on_train_epoch_end(self):
        print('\n')

    def validation_step(self, batch, batch_idx):
        x, _, _ = batch

        x_pred = self(x)

        loss = torch.nn.L1Loss()

        self.log("val_loss", loss(x_pred, x), on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)
        self.log_dict(self.valid_metrics(x_pred.view(-1), x.view(-1)), sync_dist=True, batch_size=x.shape[0])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)