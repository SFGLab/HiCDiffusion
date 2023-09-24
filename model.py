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
from torchmetrics.classification import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, R2Score, PearsonCorrCoef
from torchmetrics import MetricCollection
#starts_to_log = {18_100_000, 27_600_000, 36_600_000, 74_520_000, 83_520_000, 97_520_000, 110_020_000, 126_020_000} # HiC

starts_to_log = {18_100_000, 27_600_000, 36_600_000, 74_520_000, 83_520_000, 89_020_000, 97_520_000, 126_020_000} # HiChIP - added one interesting region

def create_image(folder, y_pred, y_real, epoch, chromosome, position):
        color_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red"])
        color_map_diff = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "white","red"])
        file_name = "%s/%s_%s_%s.png" % (folder, epoch, chromosome, str(position))
        plt.figure(figsize=(19,6))
        plt.subplot(1, 3, 1)
        plt.suptitle('Output %s - %s %s' % (epoch, chromosome, str(position)))
        plt.gca().set_title('Predicted')
        plt.imshow(y_pred, cmap=color_map, vmin=0, vmax=1)
        plt.subplot(1, 3, 2)
        plt.gca().set_title('Real')
        plt.imshow(y_real, cmap=color_map, vmin=0, vmax=1)
        plt.subplot(1, 3, 3)
        plt.gca().set_title('Difference')
        plt.imshow(y_real-y_pred, cmap=color_map_diff, vmin=-1, vmax=1)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(file_name, dpi=400)
        plt.cla()
        return file_name


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
class ResidualConv2d(nn.Module):
    def __init__(self, hidden_in, hidden_out, kernel, padding, dilation):
        super(ResidualConv2d, self).__init__()
        self.main = nn.Sequential(
                                    nn.Conv2d(hidden_in, hidden_out, kernel, padding=padding, dilation=dilation),
                                    nn.BatchNorm2d(hidden_out),
                                    nn.ReLU(),
                                    nn.Conv2d(hidden_out, hidden_out, kernel, padding=padding, dilation=dilation),
                                    nn.BatchNorm2d(hidden_out)
                                    )
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        output = self.main(x)

        return self.relu(output+residual)    
    
class Interaction3DPredictor(pl.LightningModule):
    def __init__(self, validation_folder, prediction_folder):
        super().__init__()
        self.save_hyperparameters()

        self.example_input_array = torch.Tensor(2, 5, int(math.pow(2, 21)))
        self.validation_folder = validation_folder
        self.prediction_folder = prediction_folder

        self.conv_blocks = nn.ModuleList([])
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.relu = nn.ReLU()
        self.conv_trans_blocks = nn.ModuleList([])

        channels_in = [5, 32, 32, 32, 64, 64, 64, 128, 128, 256, 256, 256, 256, 256]
        channels_out = [32, 32, 32, 64, 64, 64, 128, 128, 256, 256, 256, 256, 256, 256]

        for i in range(0, 13):
            self.conv_blocks.append(ResidualConv1d(channels_in[i], channels_out[i], 3, 1))

        for i in range(0, 5): #hic
            self.conv_trans_blocks.append(ResidualConv2d(512, 512, 3, 2**(i+1), 2**(i+1)))

        self.conv_trans_blocks.append(nn.Sequential(nn.Conv2d(512, 1, 1)))

        # metrics
        
        metrics = MetricCollection([ MeanAbsoluteError(), MeanAbsolutePercentageError(), MeanSquaredError(), R2Score(), PearsonCorrCoef()
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')

    def repeat_dimension(self, x):
        dim_reapeat = 256
            
        x_i = x.unsqueeze(2).repeat(1, 1, dim_reapeat, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, dim_reapeat)
        return torch.cat([x_i, x_j], dim = 1)
    
    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)

        res_transformer = x

        x = self.transformer_encoder(x)

        x = self.relu(x+res_transformer)

        x = self.repeat_dimension(x)

        for block in self.conv_trans_blocks:
            x = block(x)

        x = x.view(-1, 256, 256)

        return x

    def training_step(self, batch, batch_idx):
        x, y, pos = batch

        y_pred = self(x)
        loss = torch.nn.L1Loss()

        self.log("train_loss", loss(y_pred, y), on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)
        self.log_dict(self.train_metrics(y_pred, y), sync_dist=True, batch_size=x.shape[0])

        return loss(y_pred, y)
    
    def on_train_epoch_end(self):
        print('\n')
    
    def on_validation_epoch_end(self): # upload from previous epoch
        if(self.current_epoch >= 1):
            if(self.global_rank == 0):
                for pos in starts_to_log:
                    example_name = "example_%s_%s" % ("chr9", str(pos))
                    self.logger.log_image(key = example_name, images=["%s/%s_%s_%s.png" % (self.validation_folder, self.current_epoch-1, "chr9", str(pos))])

    def validation_step(self, batch, batch_idx):
        x, y, pos = batch
        y_pred = self(x)

        loss = torch.nn.L1Loss()

        self.log("val_loss", loss(y_pred, y), on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)
        self.log_dict(self.valid_metrics(y_pred, y), sync_dist=True)

        for i in range(0, x.shape[0]):
            if(pos[1][i].item() in starts_to_log):
                create_image(self.validation_folder, y_pred[i].cpu(), y[i].cpu(), self.current_epoch, pos[0][i], pos[1][i].item())


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, pos = batch
        y_pred = self(x)

        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)