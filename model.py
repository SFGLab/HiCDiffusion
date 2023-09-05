import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
import torch

class Interaction3DPredictor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.ModuleList([])
        self.conv_trans_blocks = nn.ModuleList([])
        channels_in = [5, 32, 32, 32, 64, 64, 64, 64, 128, 128, 256, 256, 256]
        channels_out = [32, 32, 32, 64, 64, 64, 64, 128, 128, 256, 256, 256, 512]
        for i in range(0, 13):
            if(i % 2 == 0):
                self.conv_blocks.append(nn.Sequential(nn.Conv1d(channels_in[i], channels_out[i], 5, padding=2),
                                                nn.ReLU(),
                                                nn.MaxPool1d(2)))
            else:
                self.conv_blocks.append(nn.Sequential(nn.Conv1d(channels_in[i], channels_out[i], 5, padding=2),
                                                nn.ReLU()))
        for i in range(0, 12):
            self.conv_trans_blocks.append(nn.Sequential(nn.ConvTranspose2d(channels_out[-i-1], channels_in[-i-1], 5, padding=2),
                                                nn.ReLU()))
        
        self.conv_trans_blocks.append(nn.Sequential(nn.ConvTranspose2d(5, 1, 5),
                                                nn.ReLU()))
        self.fc1 = nn.Linear(1, 1)
        #self.fc2 = nn.Linear(485376, 160000)
        #self.fc3 = nn.Linear(262144, 160000)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        for block in self.conv_trans_blocks:
            x = block(x)
        x = self.fc1(x)
        #x = self.fc2(x)
        #x = self.fc3(x)
        return torch.reshape(x, (400, 400))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)