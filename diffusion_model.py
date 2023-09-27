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
from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, R2Score, PearsonCorrCoef
from torchmetrics import MetricCollection
from diffusers import DDPMScheduler, UNet2DConditionModel, UNet2DModel
from sequence_vae import Interaction3DPredictorSequenceVAE
from diffusers import DDPMPipeline
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

#starts_to_log = {18_100_000, 27_600_000, 36_600_000, 74_520_000, 83_520_000, 97_520_000, 110_020_000, 126_020_000} # HiC

starts_to_log = {18_100_000, 27_600_000, 36_600_000, 74_520_000, 83_520_000, 89_020_000, 97_520_000, 126_020_000} # HiChIP - added one interesting region

val_interations = 100

def create_image(folder, y_pred, y_real, epoch, chromosome, position):
        color_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red"])
        color_map_diff = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "white","red"])
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
        plt.gca().set_title('Difference')
        plt.imshow(y_real-y_pred, cmap=color_map_diff, vmin=-5, vmax=5)
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
    
class Interaction3DPredictorDiffusion(pl.LightningModule):
    def __init__(self, validation_folder, prediction_folder, vae_model):
        super().__init__()
        self.save_hyperparameters()
        self.validation_folder = validation_folder
        self.prediction_folder = prediction_folder
        self.generator = torch.manual_seed(1996)

        #self.example_input_array = [torch.Tensor(2, 256, 256), torch.Tensor(2).long(), torch.Tensor(2, 256, 256)]
        
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
        self.encoder = Interaction3DPredictorSequenceVAE.load_from_checkpoint(vae_model).model.encoder
        self.encoder.requires_grad = False
        # DIFFUSION TIME
        self.unet = UNet2DModel(256, 1, 1, block_out_channels=(64, 128, 256, 512), class_embed_type="identity")
        #self.unet = UNet2DModel(256, 1, 1, block_out_channels=(256, 512, 1024, 2048), class_embed_type="identity")

        metrics = MetricCollection([ MeanAbsoluteError(), MeanAbsolutePercentageError(), MeanSquaredError(), R2Score(), PearsonCorrCoef()
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
    
    def forward(self, x, ts, encoder_hidden_states):
        x = self.unet(x, ts, encoder_hidden_states)

        return x

    def process_batch(self, batch):
        x, y, pos = batch
        y = y.view(-1, 1, 256, 256)

        sequence_embeddings = self.encoder(x)
        sequence_embeddings = sequence_embeddings.view(-1, 256)

        noise = torch.randn(y.shape, device=self.device)

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (y.shape[0],), device=self.device).long()

        noisy_y = self.noise_scheduler.add_noise(y, noise, timesteps)
        
        noise_prediction = self(noisy_y, timesteps, sequence_embeddings)[0]

        noise_prediction = noise_prediction.view(-1, 256, 256)
        noisy_y = noisy_y.view(-1, 256, 256)
        noise = noise.view(-1, 256, 256)

        loss = torch.nn.L1Loss()

        return loss(noise_prediction, noise), noise_prediction, y.view(-1, 256, 256), noisy_y, pos

    def training_step(self, batch, batch_idx):
        loss, noise_prediction, y, noisy_y, _ = self.process_batch(batch)

        y_pred = noisy_y-noise_prediction

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=noisy_y.shape[0], sync_dist=True)

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
        x, y, pos = batch
        loss, noise_prediction, y, noisy_y, _ = self.process_batch(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=noisy_y.shape[0], sync_dist=True)

        for i in range(0, x.shape[0]):
            if(pos[1][i].item() in starts_to_log):
                sequence_embeddings = self.encoder(x)
                sequence_embeddings = sequence_embeddings.view(-1, 256)
                image_shape = (x.shape[0], 1, 256, 256)
                image = randn_tensor(image_shape, generator=self.generator, device=self.device)

                self.noise_scheduler.set_timesteps(val_interations)

                for t in self.noise_scheduler.timesteps:
                    # 1. predict noise model_output
                    model_output = self(image, t, sequence_embeddings).sample

                    # 2. compute previous image: x_t -> x_t-1
                    image = self.noise_scheduler.step(model_output, t, image, generator=self.generator).prev_sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1)
                image = image.view(-1, 256, 256).numpy()
                create_image(self.validation_folder, image[i], y[i].cpu(), self.current_epoch, pos[0][i], pos[1][i].item())


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, pos = batch
        y_pred = self(x)

        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)