import datasets
import lightning.pytorch as pl
from hicdiffusion_model import HiCDiffusion
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loggers import WandbLogger
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn as nn
import torch
import math
import torchvision

def normalize(A):
    A = A.view(-1, 256, 256)
    outmap_min, _ = torch.min(A, dim=1, keepdim=True)
    outmap_max, _ = torch.max(A, dim=1, keepdim=True)
    outmap = (A - outmap_min) / (outmap_max - outmap_min)
    return outmap.view(-1, 1, 256, 256)

def create_image_fid_chart(y_real_value, chromosome, position, pos_end):
    y_real_value_tensor = normalize(y_real_value.view(256*256)).view(1, 1, 256, 256).repeat(1, 3, 1, 1).detach()
    y_real_value = y_real_value.view(256, 256).detach().numpy()

    color_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red"])
    file_name = "fid_example.png"
    fig = plt.figure(figsize=(8, 3), constrained_layout=True)
    axs = fig.subplot_mosaic([['Gauss_0','Gauss_1','Gauss_2','Gauss_3']])

    fig.suptitle('FID score of augmented data %s %s-%s' % (chromosome, str(position), str(pos_end)))

    axs["Gauss_0"].set_title('Real data')
    axs["Gauss_0"].imshow(y_real_value, cmap=color_map, vmin=0, vmax=5)
    fid = FrechetInceptionDistance(feature=64, normalize=True).set_dtype(torch.float64)
    # limitation of torchmetrics library - we need to provide at least 2 or more images from each distribution; 
    # however, it doesn't matter if we replicate it 2 or more times, the result is same
    fid.update(y_real_value_tensor, real=True)
    fid.update(y_real_value_tensor, real=True)
    fid.update(y_real_value_tensor, real=False)
    fid.update(y_real_value_tensor, real=False)
    fid_score = round(fid.compute().item(), 1)
    axs["Gauss_0"].set_xlabel(f'FID = {fid_score}')


    axs["Gauss_1"].set_title('Gauss σ=1')
    y_filtered = gaussian_filter(y_real_value, sigma=1)
    axs["Gauss_1"].imshow(y_filtered, cmap=color_map, vmin=0, vmax=5)
    fid = FrechetInceptionDistance(feature=64, normalize=True).set_dtype(torch.float64)
    y_filtered_tensor = normalize(torch.from_numpy(y_filtered)).view(1, 1, 256, 256).repeat(1, 3, 1, 1)
    fid.update(y_real_value_tensor, real=True)
    fid.update(y_real_value_tensor, real=True)
    fid.update(y_filtered_tensor, real=False)
    fid.update(y_filtered_tensor, real=False)
    fid_score = round(fid.compute().item(), 1)
    axs["Gauss_1"].set_xlabel(f'FID = {fid_score}')

    axs["Gauss_2"].set_title('Gauss σ=3')
    y_filtered = gaussian_filter(y_real_value, sigma=3)
    axs["Gauss_2"].imshow(y_filtered, cmap=color_map, vmin=0, vmax=5)
    fid = FrechetInceptionDistance(feature=64, normalize=True).set_dtype(torch.float64)
    y_filtered_tensor = normalize(torch.from_numpy(y_filtered)).view(1, 1, 256, 256).repeat(1, 3, 1, 1)
    fid.update(y_real_value_tensor, real=True)
    fid.update(y_real_value_tensor, real=True)
    fid.update(y_filtered_tensor, real=False)
    fid.update(y_filtered_tensor, real=False)
    fid_score = round(fid.compute().item(), 1)
    axs["Gauss_2"].set_xlabel(f'FID = {fid_score}')

    axs["Gauss_3"].set_title('Gauss σ=7')
    y_filtered = gaussian_filter(y_real_value, sigma=7)
    for_scale = axs["Gauss_3"].imshow(y_filtered, cmap=color_map, vmin=0, vmax=5)
    fid = FrechetInceptionDistance(feature=64, normalize=True).set_dtype(torch.float64)
    y_filtered_tensor = normalize(torch.from_numpy(y_filtered).view(256*256)).view(1, 1, 256, 256).repeat(1, 3, 1, 1)
    fid.update(y_real_value_tensor, real=True)
    fid.update(y_real_value_tensor, real=True)
    fid.update(y_filtered_tensor, real=False)
    fid.update(y_filtered_tensor, real=False)
    fid_score = round(fid.compute().item(), 1)
    axs["Gauss_3"].set_xlabel(f'FID = {fid_score}')

    fig.colorbar(for_scale, ax=list(axs.values()))
    plt.savefig(file_name, dpi=400)
    plt.cla()

def create_image(y_pred):
        color_map_diff = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "white","red"])
        plt.figure(figsize=(4, 4))
        plt.imshow(y_pred, cmap=color_map_diff, vmin=-1, vmax=1)
        plt.savefig("noise.svg", dpi=400)
        plt.cla()

def main():

    pl.seed_everything(1996)
    batch_size = 1
    #noise = torch.randn((256, 256))
    #create_image(noise)
    genomic_data_module = datasets.GenomicDataModule("GRCh38_full_analysis_set_plus_decoy_hla.fa", "exclude_regions.bed", 500_000, 1, ["chr8"], ["chr9"])
    genomic_data_module.setup()

    for value in iter(genomic_data_module.test_dataloader()):
        if(int(value[2][1][0].item()) == 8_600_000):
            create_image_fid_chart(value[1][0].view(1, 1, 256, 256), value[2][0][0], value[2][1][0].item(), value[2][2][0].item())
        
if __name__ == "__main__":

    main()