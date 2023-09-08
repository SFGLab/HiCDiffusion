import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
print_sizes = False
class Interaction3DPredictor(pl.LightningModule):
    def __init__(self):
        super().__init__()
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
        if(print_sizes):
            print(x.shape)
        for block in self.conv_blocks:
            x = block(x)
            if(print_sizes):
                print(x.shape)
        x = x.view(-1, 256, 16, 16)
        for block in self.conv_trans_blocks:
            x = block(x)
            if(print_sizes):
                print(x.shape)
        x = x.view(-1, 200, 200)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, sync_dist=True)
        mae = torch.nn.L1Loss()
        mse = torch.nn.MSELoss()
        self.log("mae", mae(y_hat, y), sync_dist=True)
        self.log("mse", mse(y_hat, y), sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)