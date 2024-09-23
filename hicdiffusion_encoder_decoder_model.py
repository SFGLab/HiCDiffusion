import lightning.pytorch as pl
import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt
import matplotlib
from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, R2Score, PearsonCorrCoef
from torchmetrics import MetricCollection
import os
#starts_to_log = {18_100_000, 27_600_000, 36_600_000, 74_520_000, 83_520_000, 97_520_000, 110_020_000, 126_020_000} # HiC

def ptp(input):
    return input.max() - input.min()

starts_to_log = {18_100_000, 27_600_000, 36_600_000} # HiChIP - added one interesting region
eps = 1e-7

def create_image(folder, y_pred, y_real, epoch, chromosome, position):
        color_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red"])
        color_map_diff = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "white","red"])
        file_name = "%s/%s_%s_%s.png" % (folder, epoch, chromosome, str(position))
        plt.figure(figsize=(19,6))
        plt.subplot(1, 3, 1)
        plt.suptitle('Output %s - %s %s' % (epoch, chromosome, str(position)))
        plt.gca().set_title('Predicted')
        plt.imshow(y_pred, cmap=color_map, vmin=0, vmax=5)
        plt.subplot(1, 3, 2)
        plt.gca().set_title('Real')
        plt.imshow(y_real, cmap=color_map, vmin=0, vmax=5)
        plt.subplot(1, 3, 3)
        pearson = PearsonCorrCoef()
        plt.gca().set_title('Difference (PCC: %s)' % str(round(pearson(y_pred.view(-1), y_real.view(-1)).item(), 4)))
        plt.imshow(y_real-y_pred, cmap=color_map_diff, vmin=-5, vmax=5)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(file_name, dpi=400)
        plt.cla()
        return file_name
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_blocks = nn.ModuleList([])
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.relu = nn.ReLU()

        channels_in = [5, 32, 32, 32, 64, 64, 64, 128, 128, 256, 256, 256, 256, 256]
        channels_out = [32, 32, 32, 64, 64, 64, 128, 128, 256, 256, 256, 256, 256, 256]

        for i in range(0, 13):
            self.conv_blocks.append(ResidualConv1d(channels_in[i], channels_out[i], 3, 1))

    def repeat_dimension(self, x):
        dim_reapeat = 256
            
        x_i = x.unsqueeze(2).repeat(1, 1, dim_reapeat, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, dim_reapeat)
        return torch.cat([x_i, x_j], dim = 1)
    
    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = torch.transpose(x, 1, 2)
        res_transformer = x

        x = self.transformer_encoder(x)

        x = self.relu(x+res_transformer)
        x = torch.transpose(x, 1, 2)

        x = self.repeat_dimension(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_trans_blocks = nn.ModuleList([])

        for i in range(0, 5):
            self.conv_trans_blocks.append(ResidualConv2d(512, 512, 3, 2**(i+1), 2**(i+1)))



    def forward(self, x):
        for block in self.conv_trans_blocks:
            x = block(x)

        return x

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
    
class HiCDiffusionEncoderDecoder(pl.LightningModule):
    def __init__(self, validation_folder, val_chr, test_chr):
        super().__init__()
        self.save_hyperparameters()
        self.val_chr = val_chr
        self.test_chr = test_chr

        self.example_input_array = torch.Tensor(2, 5, int(math.pow(2, 21)))
        self.validation_folder = validation_folder

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.reduce_layer = nn.Sequential(nn.Conv2d(512, 1, 1))

        # metrics
        
        metrics = MetricCollection([ MeanAbsoluteError(), MeanAbsolutePercentageError(), MeanSquaredError(), PearsonCorrCoef()
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')

    def repeat_dimension(self, x):
        dim_reapeat = 256
            
        x_i = x.unsqueeze(2).repeat(1, 1, dim_reapeat, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, dim_reapeat)
        return torch.cat([x_i, x_j], dim = 1)
    
    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
        x = self.reduce_layer(x)
        x = x.view(-1, 256, 256)
        
        return x

    def training_step(self, batch, batch_idx):
        x, y, pos = batch

        y_pred = self(x)
        loss = torch.nn.L1Loss()

        self.log("train_loss", loss(y_pred, y), on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)

        y_pred_flat = y_pred.view(-1)
        y_flat = y.view(-1)
        
        if(ptp(y_pred_flat) == 0.0):
            y_pred_flat[0] += eps
            
        if(ptp(y_flat) == 0.0):
            y_flat[0] += eps
        
        self.log_dict(self.train_metrics(y_pred_flat, y_flat), sync_dist=True, on_epoch=True, batch_size=x.shape[0])

        return loss(y_pred, y)
    
    def on_train_epoch_end(self):
        print('\n')
    
    def on_validation_epoch_end(self): # upload from previous epoch
        if(self.current_epoch >= 1):
            if(self.global_rank == 0):
                for pos in starts_to_log:
                    example_name = "example_%s_%s" % (self.val_chr, str(pos))
                    path_to_img = "%s/%s_%s_%s.png" % (self.validation_folder, self.current_epoch-1, self.val_chr, str(pos))
                    if(os.path.isfile(path_to_img)): # sometimes it might be missing - e.g. is in centromere
                        self.logger.log_image(key = example_name, images=[path_to_img])

    def validation_step(self, batch, batch_idx):
        x, y, pos = batch
        y_pred = self(x)

        loss = torch.nn.L1Loss()

        self.log("val_loss", loss(y_pred, y), on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)
        y_pred_flat = y_pred.view(-1)
        y_flat = y.view(-1)
        
        if(ptp(y_pred_flat) == 0.0):
            y_pred_flat[0] += eps
            
        if(ptp(y_flat) == 0.0):
            y_flat[0] += eps
        
        self.log_dict(self.valid_metrics(y_pred_flat, y_flat), on_epoch=True, sync_dist=True, batch_size=x.shape[0])

        for i in range(0, x.shape[0]):
            if(pos[1][i].item() in starts_to_log):
                create_image(self.validation_folder, y_pred[i].cpu(), y[i].cpu(), self.current_epoch, pos[0][i], pos[1][i].item())


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, pos = batch
        y_pred = self(x)

        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)