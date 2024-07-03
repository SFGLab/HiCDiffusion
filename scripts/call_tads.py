from cooler.create import ArrayLoader
import h5py
import cooler
from Bio import SeqIO, motifs
import lightning.pytorch as pl
from hicdiffusion_model import HiCDiffusion
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
import lightning.pytorch as pl
from hicdiffusion_model import HiCDiffusion
from torch.utils.data import Dataset, DataLoader
import pyranges as pr
import pandas as pd
import torch.nn.functional as Fun
import re
from pytadbit import HiC_data
import datasets
import math
from mutation_transdup import PertubationDataset
from pytadbit.tadbit import insulation_score, insulation_to_borders
import numpy as np
from scipy.stats.stats import pearsonr 
import seaborn as sns
wsize = (1, 5)
matplotlib.rcParams.update({'font.size': 35})

def _tads_value(axs, figure, y_pred_value, y_real_value, pred_borders, real_borders, chromosome, position, pos_end, insc1, insc2, l=False):
    y_real_value = y_real_value.view(256, 256).detach().numpy()
    y_pred_value = y_pred_value.view(256, 256).detach().numpy()
    color_map2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "white","red"])


    axs[f"Pred{figure}"].set_title('Predicted value')
    axs[f"Pred{figure}"].imshow(y_pred_value, cmap=color_map2, vmin=-5, vmax=5)
    axs[f"Pred{figure}"].set_xlabel(f'{chromosome}:{position}-{pos_end}')
    axs[f"Pred{figure}"].get_xaxis().set_ticks([])
    axs[f"Pred{figure}"].get_yaxis().set_visible(False)

    for border in pred_borders:
        axs[f"Pred{figure}"].plot(border[0], border[0], marker='D', color='orange', markersize=18)
    axs[f"Real{figure}"].set_title('Real value')
    axs[f"Real{figure}"].imshow(y_real_value, cmap=color_map2, vmin=-5, vmax=5)
    axs[f"Real{figure}"].set_xlabel(f'{chromosome}:{position}-{pos_end}') 
    axs[f"Real{figure}"].get_xaxis().set_ticks([])
    axs[f"Real{figure}"].get_yaxis().set_visible(False)
    for border in real_borders:
        axs[f"Real{figure}"].plot(border[0], border[0], marker='D', color='orange', markersize=18)

    if(not l):
        axs[f'Signal{figure}'].set_title(f'Insulation')
    l1 = axs[f'Signal{figure}'].plot([insc1[(wsize)].get(i, float('nan')) for i in range(max(insc1[(wsize)]))], label='Insulation score predicted', linewidth=5.0)
    l2 = axs[f'Signal{figure}'].plot([insc2[(wsize)].get(i, float('nan')) for i in range(max(insc2[(wsize)]))], label='Insulation score real', linewidth=5.0)
    axs[f"Signal{figure}"].set_xlabel(f'{chromosome}:{position}-{pos_end}') 
    axs[f"Signal{figure}"].get_xaxis().set_ticks([])

    axs[f'Signal{figure}'].grid()
    axs[f'Signal{figure}'].axhline(0, color='k')
    axs[f'Signal{figure}'].set_ylim(-1, 1.5)
    axs[f'Signal{figure}'].set_xlim(0, 256)
    if(l):
        axs[f'Signal{figure}'].legend(l1 + l2, [l.get_label() for l in l1 + l2], frameon=False, bbox_to_anchor=(0.5, -0.2), loc='upper center')

def tads_value(params1, params2, pearsons):

    y_pred_value1, y_real_value1, pred_borders1, real_borders1, chromosome1, position1, pos_end1, insc1_1, insc2_1 = params1
    y_pred_value2, y_real_value2, pred_borders2, real_borders2, chromosome2, position2, pos_end2, insc1_2, insc2_2 = params2

    fig = plt.figure(figsize=(18, 28), constrained_layout=True)
    fig.suptitle('Calling of TADs')
    if(position1 == position2):
        file_name = f"tads/{chromosome1}_{position1}_{pos_end1}"
    else:
        file_name = "tads"
    axs = fig.subplot_mosaic([['Pred1', 'Pred1', 'Pred2', 'Pred2'],
                              ['Pred1', 'Pred1', 'Pred2', 'Pred2'],
                              ['Real1', 'Real1', 'Real2', 'Real2'],
                              ['Real1', 'Real1', 'Real2', 'Real2'],
                              ['Signal1', 'Signal1', 'Signal1', 'Pearson'], 
                              ['Signal2', 'Signal2', 'Signal2', 'Pearson']])

    _tads_value(axs, "1", y_pred_value1, y_real_value1, pred_borders1, real_borders1, chromosome1, position1, pos_end1, insc1_1, insc2_1)
    _tads_value(axs, "2", y_pred_value2, y_real_value2, pred_borders2, real_borders2, chromosome2, position2, pos_end2, insc1_2, insc2_2, True)
    if not(pearsons is None):
        sns.violinplot(pearsons, ax=axs["Pearson"])
        axs[f'Pearson'].set_title(f'Insulation')
    plt.savefig(f"{file_name}.png", dpi=400)
    plt.savefig(f"{file_name}.svg", dpi=400)
    plt.cla()


def main():
    val_chr = "chr9"
    test_chr = "chr8"
    model_ckpt= "models/hicdiffusion_test_chr8_val_chr9/best_val_loss_hicdiffusion.ckpt"
    pl.seed_everything(2000)

    model = HiCDiffusion.load_from_checkpoint(model_ckpt, strict=False)

    dl = datasets.GenomicDataModule("GRCh38_full_analysis_set_plus_decoy_hla.fa", "exclude_regions.bed", 500_000, 1, [val_chr], [test_chr])
    dl.setup()
    dl = dl.predict_dataloader()

    trainer = pl.Trainer(devices=1, num_sanity_val_steps=0)
    predictions = trainer.predict(model, dl)
    avg_pearson = 0
    avg_pearson_n = 0
    pearsons = []
    to_plot = []
    for value in predictions:
        #h = h5py.File("cworld-test_hg19_C-40000-raw.hdf5", 'r')
        #heatmap = h['interactions']
        if(1-np.count_nonzero(value[3][0][0])/(256*256) > 0.05 or float(sum(value[3][0][0][-1])) == 0 or float(sum(value[3][0][0][0])) == 0): # skip the ones with 5% gaps in HiC and the ones starting / ending with all 0s
            continue
        # create some bins , using cooler-binnify or some other way
        hic_object_to_be = dict()
        bias = {}
        for i in range(value[1][0][0].shape[0]):
            for j in range(value[1][0][0].shape[1]):
                hic_object_to_be[(i*256+j)] = math.exp(float(value[1][0][0][i][j]))-1
                bias[i*256+j] = 1
        my_data = HiC_data(hic_object_to_be, 256, chromosomes={"current": 256})
        #my_data.bias = bias
        my_data.normalize_hic()
        my_data.normalize_expected()
        insc1, delta1 = insulation_score(my_data, [wsize], resolution=1, normalize=True, delta=1)
        borders1 = insulation_to_borders(insc1[wsize], delta1[wsize], min_strength=0.2) # PRED

        hic_object_to_be2 = dict()
        bias2 = {}
        for i in range(value[3][0][0].shape[0]):
            for j in range(value[3][0][0].shape[1]):
                hic_object_to_be2[(i*256+j)] = math.exp(float(value[3][0][0][i][j]))-1
                bias2[i*256+j] = 1
        my_data2 = HiC_data(hic_object_to_be2, 256, chromosomes={"current": 256})
        #my_data2.bias = bias2
        my_data2.normalize_hic()
        my_data2.normalize_expected()
        insc2, delta2 = insulation_score(my_data2, [wsize], resolution=1, normalize=True, delta=1)
        borders2 = insulation_to_borders(insc2[wsize], delta2[wsize], min_strength=0.2) # REAL

        insc1_signal = [insc1[wsize][y] for y in range(0+5, 256-5)]
        insc2_signal = [insc2[wsize][y] for y in range(0+5, 256-5)]

        pearson = pearsonr(insc1_signal, insc2_signal)[0]
        avg_pearson += pearson
        avg_pearson_n += 1
        print(f"Pearson:{pearson}, average pearson so far {avg_pearson/avg_pearson_n}")
        pearsons.append([f"{value[0][0][0]}_{int(value[0][1])}_{int(value[0][2])}", pearson])
        if(int(value[0][1]) == 20100000 or int(value[0][1]) == 130380000):
            to_plot.append((value[1][0][0], value[3][0][0], borders1, borders2, value[0][0][0], int(value[0][1][0]), int(value[0][2][0]), insc1, insc2))
        tads_value((value[1][0][0], value[3][0][0], borders1, borders2, value[0][0][0], int(value[0][1][0]), int(value[0][2][0]), insc1, insc2), 
            (value[1][0][0], value[3][0][0], borders1, borders2, value[0][0][0], int(value[0][1][0]), int(value[0][2][0]), insc1, insc2),
            None)
        pass
    pearsons = pd.DataFrame(pearsons, columns=["pos", "PCC"]).set_index("pos")
    tads_value(to_plot[0], to_plot[1], pd.DataFrame(pearsons))
    if(len(pearsons) > 2):
        pass
        pd.DataFrame(pearsons).to_csv("pearsons.csv")
if __name__ == "__main__":

    main()