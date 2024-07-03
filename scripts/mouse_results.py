import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.transform import resize
import matplotlib
import numpy as np
import comparison_datasets
import torch
plt.rcParams.update({'font.size': 20})
color_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "white","red"])
normal_chromosomes = ["chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8", "chr9", "chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21","chr22"]
window_size = 2_000_000
output_res = 10_000 # IT HAS TO BE ALSO RES OF BEDPE!!!
size_img = 256
length_to_input = 97152 # correction for input to network - easier to ooperate whith maxpooling when ^2


fig = plt.figure(figsize=(8, 14), constrained_layout=True)
axs = fig.subplot_mosaic([['Row1', 'Row1', 'Row1', 'Row1'], ['Row2', 'Row2', 'Row2', 'Row2'], ['Row3', 'Row3', 'Row3', 'Row3'],
                          ['Row4', 'Row4', 'Row4', 'Row4'], ['Row5', 'Row5', 'Row5', 'Extra']])

df = pd.read_csv("results_mouse.csv")
df["Chromosome"] = df["Chromosome"].str.capitalize()
df_pearson = df[["Chromosome", "pos", "pearson"]]
df_pearson["Metric"] = "PCC"
df_pearson["Value"] = df_pearson["pearson"]
df_pearson = df_pearson.drop(["pearson"], axis=1)
df_scc = df[["Chromosome", "pos", "scc"]]
df_scc["Metric"] = "SCC"
df_scc["Value"] = df_scc["scc"]
df_scc = df_scc.drop(["scc"], axis=1)

df = pd.concat([df_pearson, df_scc]).reset_index(drop=True)

for i in range(0, 5):
    if(i == 4):
        df_now = df[df["Chromosome"].isin([f"Chr{str(i*4+1)}", f"Chr{str(i*4+2)}", f"Chr{str(i*4+3)}"])]
    else:
        df_now = df[df["Chromosome"].isin([f"Chr{str(i*4+1)}", f"Chr{str(i*4+2)}", f"Chr{str(i*4+3)}", f"Chr{str(i*4+4)}"])]
    sns.violinplot(data=df_now, x="Chromosome", y="Value", hue="Metric", density_norm="count", cut=0, ax=axs["Row"+str(i+1)], legend=True, palette="rocket")
    axs["Row"+str(i+1)].set_ylim([0, 1])
    if(i < 4):
        if(axs["Row"+str(i+1)].legend_ is not None):
            axs["Row"+str(i+1)].legend_.remove()
axs["Extra"].axis('off')
plt.savefig("fig4_sup.svg")
plt.savefig("fig4_sup.png")
plt.close()

fig = plt.figure(figsize=(12, 12), constrained_layout=True)
axs = fig.subplot_mosaic([['HiCDiffusion', 'Real'],
                          ['Diff', 'Corr']])
axs["HiCDiffusion"].set_title("HiCDiffusion")
axs["Real"].set_title("Real")
axs["Diff"].set_title("Difference")
axs["Corr"].set_title("Metrics")

axs["HiCDiffusion"].set_xticks([])
axs["Real"].set_xticks([])
axs["Diff"].set_xticks([])

axs["HiCDiffusion"].set_yticks([])
axs["Real"].set_yticks([])
axs["Diff"].set_yticks([])


axs["HiCDiffusion"].imshow(np.load("scripts/out/gm12878/prediction/npy/chr8_21100000.npy"), cmap=color_map, vmin=-5, vmax=5) #manually add it
axs["Real"].imshow(np.load("scripts/out/gm12878/prediction/npy/chr8_21100000.npy"), cmap=color_map, vmin=-5, vmax=5) #manually add it
axs["Diff"].imshow(np.load("scripts/out/gm12878/prediction/npy/chr8_21100000.npy"), cmap=color_map, vmin=-5, vmax=5) #manually add it

sns.violinplot(data=df, x="Metric", y="Value", density_norm="count", cut=0, ax=axs["Corr"], legend=False, palette="rocket")
axs["Corr"].set_xlabel("")
plt.savefig("fig7.svg")
plt.savefig("fig7.png")
