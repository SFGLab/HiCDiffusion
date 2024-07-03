import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.transform import resize
import matplotlib
import numpy as np
import comparison_datasets
import torch
plt.rcParams.update({'font.size': 20})
color_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red"])
normal_chromosomes = ["chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8", "chr9", "chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21","chr22"]
window_size = 2_000_000
output_res = 10_000 # IT HAS TO BE ALSO RES OF BEDPE!!!
size_img = 256
length_to_input = 97152 # correction for input to network - easier to ooperate whith maxpooling when ^2

class DatasetWrapper():
    def __init__(self):
        self.comparison_dataset = {}
        for chromosome in normal_chromosomes:
            self.comparison_dataset[chromosome] = comparison_datasets.HiComparison()
            self.comparison_dataset[chromosome].load("hic/%s.npz" % chromosome)
    def get_real_y(self, chromosome, pos):
        return resize(torch.Tensor(self.comparison_dataset[chromosome].get(chromosome, pos-int(length_to_input/2), window_size+int(length_to_input/2), output_res)).to(torch.float), (size_img, size_img), anti_aliasing=True)

dataset_wrapper = DatasetWrapper()

list_of_dfs = []
for i in range(1, 23):
    chr_df = pd.read_csv(f"scripts/results_csv/hicdiffusion/chr{i}.csv")
    chr_df["chr"] = f"Chr{i}"
    chr_df["Model"] = "HiCDiffusion"
    chr_df = chr_df.dropna()
    #q = chr_df["pearson"].quantile(0.01)
    #chr_df = chr_df[chr_df["pearson"] > q]
    list_of_dfs.append(chr_df)
# TO CHANGE TO CORIGAMI
for i in range(1, 23):
    chr_df = pd.read_csv(f"scripts/results_csv/corigami/chr{i}.csv")
    chr_df["chr"] = f"Chr{i}"
    chr_df["Model"] = "C.Origami - Seq"
    chr_df = chr_df.dropna()
    #q = chr_df["pearson"].quantile(0.01)
    #chr_df = chr_df[chr_df["pearson"] > q]
    list_of_dfs.append(chr_df)
    
for i in range(1, 23):
    chr_df = pd.read_csv(f"scripts/results_csv/corigami_epi/chr{i}.csv")
    chr_df["chr"] = f"Chr{i}"
    chr_df["Model"] = "C.Origami"
    chr_df = chr_df.dropna()
    #q = chr_df["pearson"].quantile(0.01)
    #chr_df = chr_df[chr_df["pearson"] > q]
    list_of_dfs.append(chr_df)
    
df = pd.concat(list_of_dfs)
df = df.rename(columns={"chr": "Chromosome", "pearson": "Correlation", "scc": "SCC"})


fig = plt.figure(figsize=(8, 14), constrained_layout=True)
axs = fig.subplot_mosaic([['Row1', 'Row1', 'Row1', 'Row1'], ['Row2', 'Row2', 'Row2', 'Row2'], ['Row3', 'Row3', 'Row3', 'Row3'],
                          ['Row4', 'Row4', 'Row4', 'Row4'], ['Row5', 'Row5', 'Row5', 'Row5'], ['Row6', 'Row6', 'Extra', 'Extra']])

for i in range(0, 6):
    if(i == 5):
        df_now = df[df["Chromosome"].isin([f"Chr{str(i*4+1)}", f"Chr{str(i*4+2)}"])]
    else:
        df_now = df[df["Chromosome"].isin([f"Chr{str(i*4+1)}", f"Chr{str(i*4+2)}", f"Chr{str(i*4+3)}", f"Chr{str(i*4+4)}"])]
    sns.violinplot(data=df_now, x="Chromosome", y="Correlation", hue="Model", density_norm="count", cut=0, ax=axs["Row"+str(i+1)], legend=True, palette="rocket")
    axs["Row"+str(i+1)].set_ylim([0, 1])
    if(i < 5):
        if(axs["Row"+str(i+1)].legend_ is not None):
            axs["Row"+str(i+1)].legend_.remove()
axs["Extra"].axis('off')
plt.savefig("fig3_pearson_sup.svg")
plt.close()

fig = plt.figure(figsize=(8, 14), constrained_layout=True)
axs = fig.subplot_mosaic([['Row1', 'Row1', 'Row1', 'Row1'], ['Row2', 'Row2', 'Row2', 'Row2'], ['Row3', 'Row3', 'Row3', 'Row3'],
                          ['Row4', 'Row4', 'Row4', 'Row4'], ['Row5', 'Row5', 'Row5', 'Row5'], ['Row6', 'Row6', 'Extra', 'Extra']])

for i in range(0, 6):
    if(i == 5):
        df_now = df[df["Chromosome"].isin([f"Chr{str(i*4+1)}", f"Chr{str(i*4+2)}"])]
    else:
        df_now = df[df["Chromosome"].isin([f"Chr{str(i*4+1)}", f"Chr{str(i*4+2)}", f"Chr{str(i*4+3)}", f"Chr{str(i*4+4)}"])]
    sns.violinplot(data=df_now, x="Chromosome", y="SCC", hue="Model", density_norm="count", cut=0, ax=axs["Row"+str(i+1)], legend=True, palette="rocket")
    axs["Row"+str(i+1)].set_ylim([0, 1])
    if(i < 5):
        if(axs["Row"+str(i+1)].legend_ is not None):
            axs["Row"+str(i+1)].legend_.remove()
axs["Extra"].axis('off')
plt.savefig("fig3_scc_sup.svg")
plt.close()

average_fid = {}

hicdiff_stats = pd.read_csv("scripts/results_csv/hicdiffusion_results.csv")
hicdiff_stats["chr"] = hicdiff_stats["Name"].apply(lambda x: x.split("Test: ")[1].split(",")[0])
hicdiff_stats = hicdiff_stats[["chr", "fid", "fid_cond"]]
hicdiff_stats = hicdiff_stats.set_index("chr")
average_fid["HiCDiffusion"] = hicdiff_stats["fid"].mean()
average_fid["HiCDiffusion - E/D"] = hicdiff_stats["fid_cond"].mean()

corigami_stats = pd.read_csv("scripts/results_csv/corigami_results.csv", index_col=0)
corigami_stats = corigami_stats.rename(columns={"fid": "fid_corigami_seq_only"})
average_fid["C.Origami - Seq"] = corigami_stats["fid_corigami_seq_only"].mean()

corigami_epi_stats = pd.read_csv("scripts/results_csv/corigami_epi_results.csv", index_col=0)
corigami_epi_stats = corigami_epi_stats.rename(columns={"fid": "fid_corigami"})
average_fid["C.Origami"] = corigami_epi_stats["fid_corigami"].mean()

all_data = pd.concat([hicdiff_stats, corigami_stats, corigami_epi_stats], axis=1)

data_barplot = all_data.stack().reset_index().rename(columns={"level_0": "Chromosome", "level_1": "Model", 0: "FID"})
data_barplot["Chromosome"] = data_barplot["Chromosome"].apply(lambda x: x.capitalize())
data_barplot["Model"] = data_barplot["Model"].apply(lambda x: "HiCDiffusion" if(x == "fid") else "HiCDiffusion - E/D" if (x == "fid_cond") else "C.Origami - Seq" if (x == "fid_corigami_seq_only") else "C.Origami")
fig = plt.figure(figsize=(8, 14), constrained_layout=True)
axs = fig.subplot_mosaic([['Row1', 'Row1', 'Row1', 'Row1'], ['Row2', 'Row2', 'Row2', 'Row2'], ['Row3', 'Row3', 'Row3', 'Row3'],
                          ['Row4', 'Row4', 'Row4', 'Row4'], ['Row5', 'Row5', 'Row5', 'Row5'], ['Row6', 'Row6', 'Extra', 'Extra']])

for i in range(0, 6):
    if(i == 5):
        df_now = data_barplot[data_barplot["Chromosome"].isin([f"Chr{str(i*4+1)}", f"Chr{str(i*4+2)}"])]
    else:
        df_now = data_barplot[data_barplot["Chromosome"].isin([f"Chr{str(i*4+1)}", f"Chr{str(i*4+2)}", f"Chr{str(i*4+3)}", f"Chr{str(i*4+4)}"])]
    sns.barplot(df_now, x="Chromosome", y="FID", hue="Model", ax=axs["Row"+str(i+1)], palette="rocket")
    if(i < 5):
        axs["Row"+str(i+1)].legend_.remove()
axs["Extra"].axis('off')
plt.savefig("fig3_fid_sup.svg")
plt.close()

fig = plt.figure(figsize=(12, 24), constrained_layout=True)
axs = fig.subplot_mosaic([['HiCDiffusion', 'HiCDiffusionED'],
                          ['COrigami', 'COrigami_epi'],
                          ['Real', "FID"],
                          ['Pearson', 'SCC']])
axs["HiCDiffusion"].set_title("HiCDiffusion")
axs["HiCDiffusionED"].set_title("HiCDiffusion - E/D")
axs["COrigami"].set_title("C.Origami - Seq")
axs["COrigami_epi"].set_title("C.Origami")
axs["Real"].set_title("Real Hi-C")

axs["HiCDiffusion"].set_xticks([])
axs["HiCDiffusionED"].set_xticks([])
axs["COrigami"].set_xticks([])
axs["COrigami_epi"].set_xticks([])
axs["Real"].set_xticks([])

axs["HiCDiffusion"].set_yticks([])
axs["HiCDiffusionED"].set_yticks([])
axs["COrigami"].set_yticks([])
axs["COrigami_epi"].set_yticks([])
axs["Real"].set_yticks([])


axs["HiCDiffusion"].imshow(np.load("scripts/out/gm12878/prediction/npy/chr8_21100000.npy"), cmap=color_map, vmin=0, vmax=5) #manually add it
axs["HiCDiffusionED"].imshow(np.load("scripts/out/gm12878/prediction/npy/chr8_21100000.npy"), cmap=color_map, vmin=0, vmax=5) #manually add it
axs["COrigami"].imshow(np.load("scripts/out/gm12878/prediction/npy/chr8_21100000.npy"), cmap=color_map, vmin=0, vmax=5)
axs["COrigami_epi"].imshow(np.load("scripts/out_epi/gm12878/prediction/npy/chr8_21100000.npy"), cmap=color_map, vmin=0, vmax=5)
axs["Real"].imshow(dataset_wrapper.get_real_y("chr8", 21100000), cmap=color_map, vmin=0, vmax=5)

df_avg_fid = pd.DataFrame.from_dict(average_fid, orient="index", columns=["FID"])
df_avg_fid["Model"] = df_avg_fid.index
sns.barplot(df_avg_fid, x="Model", y="FID", ax=axs["FID"], palette="rocket")
axs["FID"].set_title("Fréchet inception distance")


sns.violinplot(data=df, x="Model", y="Correlation", density_norm="count", cut=0, ax=axs["Pearson"], legend=False, palette="rocket")
sns.violinplot(data=df, x="Model", y="SCC", density_norm="count", cut=0, ax=axs["SCC"], legend=False, palette="rocket")


axs["Pearson"].set_title("Correlation")
axs["SCC"].set_title("SCC")
axs["FID"].tick_params(axis='x', rotation=30)
axs["Pearson"].tick_params(axis='x', rotation=30)
axs["SCC"].tick_params(axis='x', rotation=30)
plt.savefig("fig3.svg")
