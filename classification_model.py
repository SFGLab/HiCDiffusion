import lightning.pytorch as pl
import torch.nn as nn
import torch
from lightning.pytorch.utilities import grad_norm
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryRecall, BinaryPrecision, BinaryF1Score
from torchmetrics import MetricCollection
from hicdiffusion_encoder_decoder_model import HiCDiffusionEncoderDecoder

def normalize(A):
    A = A.view(-1, 256, 256)
    outmap_min, _ = torch.min(A, dim=1, keepdim=True)
    outmap_max, _ = torch.max(A, dim=1, keepdim=True)
    outmap = (A - outmap_min) / (outmap_max - outmap_min)
    return outmap.view(-1, 1, 256, 256)

def ptp(input):
    return input.max() - input.min()

size_img = 256

eps = 1e-7

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
        self.downscale = nn.Sequential(nn.Conv2d(hidden_in, hidden_out, kernel, padding=padding))
    def forward(self, x):
        residual = self.downscale(x)
        output = self.main(x)

        return self.relu(output+residual)    


class ClassificationModel(pl.LightningModule):
    def __init__(self, encoder_decoder_model, val_chr, test_chr):
        super().__init__()
        self.val_chr = val_chr
        self.test_chr = test_chr
        
        self.save_hyperparameters()
        
        self.encoder_decoder = HiCDiffusionEncoderDecoder.load_from_checkpoint(encoder_decoder_model)
        self.encoder_decoder.freeze()
        self.encoder_decoder.eval()

        metrics = MetricCollection([ BinaryAccuracy(), BinaryAUROC(), BinaryRecall(), BinaryPrecision(), BinaryF1Score()
        ])

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.convs = nn.Sequential(ResidualConv2d(512, 256, 3, 1, 1), ResidualConv2d(256, 128, 3, 1, 1), ResidualConv2d(128, 64, 3, 1, 1), ResidualConv2d(64, 32, 3, 1, 1), ResidualConv2d(32, 16, 3, 1, 1), ResidualConv2d(16, 8, 3, 1, 1), ResidualConv2d(8, 1, 3, 1, 1))
        #self.convs = nn.Sequential(ResidualConv1d(256, 128, 3, 1), ResidualConv1d(128, 64, 3, 1), ResidualConv1d(64, 32, 3, 1), ResidualConv1d(32, 16, 3, 1), ResidualConv1d(16, 8, 3, 1), ResidualConv1d(8, 1, 3, 1))
        self.fcs = nn.Sequential(nn.BatchNorm1d(256*256), nn.Dropout(0.4), nn.Linear(256*256, 200), nn.ReLU(), nn.Linear(200, 20))



    def forward(self, x):
        y_cond = self.encoder_decoder.encoder(x)
        y_cond = self.encoder_decoder.decoder(y_cond)
        y = self.convs(y_cond)

        y = y.reshape(-1, 256*256)
        y = self.fcs(y)
        
        return y

    def training_step(self, batch, batch_idx):
        x, y, pos = batch

        y_pred = self(x)
        loss = torch.nn.BCEWithLogitsLoss()

        self.log("train_loss", loss(y_pred, y), on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)
        
        self.log_dict(self.train_metrics(y_pred, y), on_epoch=True, sync_dist=True, batch_size=x.shape[0])

        return loss(y_pred, y)
    
    def on_train_epoch_end(self):
        print('\n')

    def validation_step(self, batch, batch_idx):
        x, y, pos = batch
        y_pred = self(x)

        loss = torch.nn.BCEWithLogitsLoss()

        self.log("val_loss", loss(y_pred, y), on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)
        
        self.log_dict(self.valid_metrics(y_pred, y), on_epoch=True, sync_dist=True, batch_size=x.shape[0])


    def test_step(self, batch, batch_idx):
        x, y, pos = batch
        y_pred = self(x)

        loss = torch.nn.BCEWithLogitsLoss()

        self.log("val_loss", loss(y_pred, y), on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)
        
        self.log_dict(self.valid_metrics(y_pred, y), on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, pos = batch
        y_pred = self(x)

        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.00001)