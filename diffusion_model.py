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
from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, R2Score, PearsonCorrCoef, SpearmanCorrCoef, CosineSimilarity, ConcordanceCorrCoef, RelativeSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, UniversalImageQualityIndex, ErrorRelativeGlobalDimensionlessSynthesis, MultiScaleStructuralSimilarityIndexMeasure, PeakSignalNoiseRatioWithBlockedEffect, RelativeAverageSpectralError, RootMeanSquaredErrorUsingSlidingWindow, SpectralDistortionIndex, StructuralSimilarityIndexMeasure, VisualInformationFidelity
from torchmetrics import MetricCollection
from denoise_model import UnetConditional, GaussianDiffusionConditional
from tqdm import tqdm
from model import Interaction3DPredictor

def ptp(input):
    return input.max() - input.min()

size_img = 256

#starts_to_log = {18_100_000, 27_600_000, 36_600_000, 74_520_000, 83_520_000, 97_520_000, 110_020_000, 126_020_000} # HiC

starts_to_log = {18_100_000, 27_600_000, 36_600_000, 74_520_000, 83_520_000, 89_020_000, 97_520_000, 126_020_000} # HiChIP - added one interesting region

val_interations = 100
eps = 1e-7

def create_image(folder, y_pred, y_cond, y_real, epoch, chromosome, position):
        color_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red"])
        color_map_diff = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "white","red"])
        file_name = "%s/%s_%s_%s.png" % (folder, epoch, chromosome, str(position))
        fig = plt.figure(figsize=(8, 14), constrained_layout=True)
        axs = fig.subplot_mosaic([['TopLeft', 'TopRight'],['MiddleLeft', 'MiddleRight'], ['Bottom', 'Bottom'], ['Bottom', 'Bottom']])

        fig.suptitle('Output %s - %s %s' % (epoch, chromosome, str(position)))
        axs["TopLeft"].set_title('Predicted - final')
        axs["TopLeft"].imshow(y_pred, cmap=color_map, vmin=0, vmax=5)
        axs["TopRight"].set_title('Predicted - E/D')
        axs["TopRight"].imshow(y_cond, cmap=color_map, vmin=0, vmax=5)
        pearson = PearsonCorrCoef()
        axs["MiddleLeft"].set_title('Difference - final (PCC: %s)' % str(round(pearson(y_pred.view(-1), y_real.view(-1)).item(), 4)))
        axs["MiddleLeft"].imshow(y_real-y_pred, cmap=color_map_diff, vmin=-5, vmax=5)
        axs["MiddleRight"].set_title('Difference - E/D (PCC: %s)' % str(round(pearson(y_cond.view(-1), y_real.view(-1)).item(), 4)))
        for_scale = axs["MiddleRight"].imshow(y_real-y_cond, cmap=color_map_diff, vmin=-5, vmax=5)
        axs["Bottom"].set_title('Real')
        axs["Bottom"].imshow(y_real, cmap=color_map, vmin=0, vmax=5)
        fig.colorbar(for_scale, ax=list(axs.values()))
        plt.savefig(file_name, dpi=400)
        plt.cla()
    
class Interaction3DPredictorDiffusion(pl.LightningModule):
    def __init__(self, validation_folder, encoder_decoder_model):
        super().__init__()
        self.save_hyperparameters()
        self.validation_folder = validation_folder
        
        self.encoder_decoder = Interaction3DPredictor.load_from_checkpoint(encoder_decoder_model)
        self.encoder_decoder.freeze()
        self.encoder_decoder.eval()
        self.model = UnetConditional(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            flash_attn = True,
            channels=1
        )

        self.diffusion = GaussianDiffusionConditional(
            self.model,
            image_size = 256,
            timesteps = 1000,
            sampling_timesteps = 1000 # number of steps - change to 999 later
        )


        metrics = MetricCollection([ MeanAbsoluteError(), MeanAbsolutePercentageError(), MeanSquaredError(), PearsonCorrCoef(), SpearmanCorrCoef(), CosineSimilarity(), ConcordanceCorrCoef(), RelativeSquaredError(), R2Score()
        ])
        metrics_image = MetricCollection([ PeakSignalNoiseRatio(),  UniversalImageQualityIndex(), ErrorRelativeGlobalDimensionlessSynthesis(), MultiScaleStructuralSimilarityIndexMeasure(), PeakSignalNoiseRatioWithBlockedEffect(), RelativeAverageSpectralError(), RootMeanSquaredErrorUsingSlidingWindow(), SpectralDistortionIndex(), StructuralSimilarityIndexMeasure(), VisualInformationFidelity()
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.valid_metrics_cond = metrics.clone(prefix='val_cond_')
        self.valid_metrics_image = metrics_image.clone(prefix='val_image_')
        self.valid_metrics_cond_image = metrics_image.clone(prefix='val_cond_image_')

    def process_batch(self, batch):
        x, y, pos = batch
        
        y = y.view(-1, 1, size_img, size_img)

        y_cond = self.encoder_decoder.encoder(x)
        y_cond = self.encoder_decoder.decoder(y_cond)
        y_cond_decoded = self.encoder_decoder.reduce_layer(y_cond)
        y_cond_decoded = y_cond_decoded.view(-1, size_img, size_img)
        
        y_cond = y_cond.view(-1, 512, size_img, size_img)
        loss = self.diffusion(y, x_self_cond=y_cond_decoded.view(-1, 1, size_img, size_img))

        return loss, x, y, y_cond, y_cond_decoded, pos

    def training_step(self, batch, batch_idx):
        loss, x, _, _, _, _ = self.process_batch(batch)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)

        return loss
    
    def on_train_epoch_end(self):
        print('\n')
    
    def on_validation_epoch_end(self): # upload from previous epoch
        if(self.current_epoch >= 1):
            if(self.global_rank == 0):
                for pos in starts_to_log:
                    example_name = "example_%s_%s" % ("chr9", str(pos))
                    self.logger.log_image(key = example_name, images=["%s/%s_%s_%s.png" % (self.validation_folder, self.current_epoch-1, "chr9", str(pos))])

    def validation_step(self, batch, batch_idx):
        loss, x, y, y_cond, y_cond_decoded, pos = self.process_batch(batch)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)
        
        # every 10 epoch - log statistics, which means generating all the images
        if(self.current_epoch % 10 == 9):
            y_pred = self.diffusion.sample(batch_size = y_cond.shape[0], x_self_cond=y_cond_decoded.view(-1, 1, 256, 256), return_all_timesteps=False) # (1, 1, 256, 256)
            y_pred = y_pred+y_cond_decoded.view(-1, 1, size_img, size_img) # the y_pred is in form of y - y_cond
            
            y_pred_flat = y_pred.view(-1)
            y_flat = y.view(-1)
            y_cond_decoded_flat = y_cond_decoded.view(-1) 
            if(ptp(y_pred_flat) == 0.0):
               y_pred_flat[0] += eps
                
            if(ptp(y_flat) == 0.0):
                y_flat[0] += eps
                    
            if(ptp(y_cond_decoded_flat) == 0.0):
                y_cond_decoded_flat[0] += eps
            
            self.log_dict(self.valid_metrics(y_pred_flat, y_flat), sync_dist=True, on_epoch=True, batch_size=x.shape[0])
            self.log_dict(self.valid_metrics_cond(y_cond_decoded_flat, y_flat), sync_dist=True, on_epoch=True, batch_size=x.shape[0])
            
        # log sample images
        for i in range(0, x.shape[0]):
            if(pos[1][i].item() in starts_to_log):
                predicted_y = self.diffusion.sample(batch_size = 1, x_self_cond=y_cond_decoded[i].view(1, 1, 256, 256), return_all_timesteps=False) # (1, 1, 256, 256)
                predicted_y = predicted_y+y_cond_decoded[i].view(-1, 1, size_img, size_img) # the y_pred is in form of y - y_cond
                
                create_image(self.validation_folder, predicted_y.view(256, 256).cpu(), y_cond_decoded[i].view(256, 256).cpu(), y[i].view(256, 256).cpu(), self.current_epoch, pos[0][i], pos[1][i].item())

    def test_step(self, batch, batch_idx):
        loss, x, y, y_cond, y_cond_decoded, pos = self.process_batch(batch)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)
        
        y_pred = self.diffusion.sample(batch_size = y_cond.shape[0], x_self_cond=y_cond_decoded.view(-1, 1, 256, 256), return_all_timesteps=False) # (1, 1, 256, 256)
        y_pred = y_pred+y_cond_decoded.view(-1, 1, size_img, size_img) # the y_pred is in form of y - y_cond
        
        y_pred_flat = y_pred.view(-1)
        y_flat = y.view(-1)
        y_cond_decoded_flat = y_cond_decoded.view(-1) 
        if(ptp(y_pred_flat) == 0.0):
            y_pred_flat[0] += eps
            
        if(ptp(y_flat) == 0.0):
            y_flat[0] += eps
                
        if(ptp(y_cond_decoded_flat) == 0.0):
            y_cond_decoded_flat[0] += eps
            
        self.log_dict(self.valid_metrics(y_pred_flat, y_flat), sync_dist=True, on_epoch=True, batch_size=x.shape[0])
        self.log_dict(self.valid_metrics_cond(y_cond_decoded_flat, y_flat), sync_dist=True, on_epoch=True, batch_size=x.shape[0])
        
        y = y.view(-1, 1, size_img, size_img)
        y_pred = y_pred.view(-1, 1, size_img, size_img)
        y_cond_decoded = y_cond_decoded.view(-1, 1, size_img, size_img)
        
        self.log_dict(self.valid_metrics_image(y_pred, y), sync_dist=True, on_epoch=True, batch_size=x.shape[0])
        self.log_dict(self.valid_metrics_cond_image(y_cond_decoded, y), sync_dist=True, on_epoch=True, batch_size=x.shape[0])
            
        # log sample images
        for i in range(0, x.shape[0]):
            create_image("test_model_folder/", y_pred[i].view(256, 256).cpu(), y_cond_decoded[i].view(256, 256).cpu(), y[i].view(256, 256).cpu(), "final", pos[0][i], pos[1][i].item())
            example_name = "example_%s_%s" % ("chr9", str(pos[1][i].item()))
            self.logger.log_image(key = example_name, images=["%s/%s_%s_%s.png" % ("test_model_folder/", "final", str(pos[0][i]), str(pos[1][i].item()))])
                

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        loss, x, y, y_cond, y_cond_decoded, pos = self.process_batch(batch)
        
        y_pred = self.diffusion.sample(batch_size = y_cond.shape[0], x_self_cond=y_cond_decoded.view(-1, 1, 256, 256), return_all_timesteps=False) # (1, 1, 256, 256)
        y_pred = y_pred.view(-1, 1, size_img, size_img)
        y_cond_decoded = y_cond_decoded.view(-1, 1, size_img, size_img)
        
        return y, y_pred, y_cond_decoded

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)