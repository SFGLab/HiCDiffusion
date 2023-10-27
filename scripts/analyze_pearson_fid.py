import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
plt.rcParams.update({'font.size': 20})
color_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red"])

list_of_dfs = []
for i in range(1, 23):
    chr_df = pd.read_csv(f"results_csv/hicdiffusion/chr{i}.csv")
    chr_df["chr"] = f"Chr{i}"
    chr_df["Model"] = "HiCDiffusion"
    chr_df = chr_df.dropna()
    #q = chr_df["pearson"].quantile(0.01)
    #chr_df = chr_df[chr_df["pearson"] > q]
    list_of_dfs.append(chr_df)
# TO CHANGE TO CORIGAMI
for i in range(1, 23):
    chr_df = pd.read_csv(f"results_csv/corigami/chr{i}.csv")
    chr_df["chr"] = f"Chr{i}"
    chr_df["Model"] = "C.Origami"
    chr_df = chr_df.dropna()
    #q = chr_df["pearson"].quantile(0.01)
    #chr_df = chr_df[chr_df["pearson"] > q]
    list_of_dfs.append(chr_df)
    
df = pd.concat(list_of_dfs)
df = df.rename(columns={"chr": "Chromosome", "pearson": "Correlation"})


fig = plt.figure(figsize=(8, 14), constrained_layout=True)
axs = fig.subplot_mosaic([['Row1', 'Row1', 'Row1', 'Row1'], ['Row2', 'Row2', 'Row2', 'Row2'], ['Row3', 'Row3', 'Row3', 'Row3'],
                          ['Row4', 'Row4', 'Row4', 'Row4'], ['Row5', 'Row5', 'Row5', 'Row5'], ['Row6', 'Row6', 'Extra', 'Extra']])

for i in range(0, 6):
    if(i == 5):
        df_now = df[df["Chromosome"].isin([f"Chr{str(i*4+1)}", f"Chr{str(i*4+2)}"])]
    else:
        df_now = df[df["Chromosome"].isin([f"Chr{str(i*4+1)}", f"Chr{str(i*4+2)}", f"Chr{str(i*4+3)}", f"Chr{str(i*4+4)}"])]
    sns.violinplot(data=df_now, x="Chromosome", y="Correlation", hue="Model", density_norm="count", cut=0, ax=axs["Row"+str(i+1)], legend=False, palette="rocket")
    axs["Row"+str(i+1)].set_ylim([0, 1])
    if(i < 5):
        axs["Row"+str(i+1)].legend_.remove()
axs["Extra"].axis('off')
plt.savefig("fig3_pearson_sup.svg")
plt.close()

average_fid = {}

hicdiff_stats = pd.read_csv("results_csv/hicdiffusion_results.csv")
hicdiff_stats["chr"] = hicdiff_stats["Name"].apply(lambda x: x.split("Test: ")[1].split(",")[0])
hicdiff_stats = hicdiff_stats[["chr", "fid", "fid_cond"]]
hicdiff_stats = hicdiff_stats.set_index("chr")
average_fid["HiCDiffusion"] = hicdiff_stats["fid"].mean()
average_fid["HiCDiffusion - E/D"] = hicdiff_stats["fid_cond"].mean()
corigami_stats = pd.read_csv("results_csv/corigami_results.csv", index_col=0)
corigami_stats = corigami_stats.rename(columns={"fid": "fid_corigami"})
average_fid["C.Origami"] = corigami_stats["fid_corigami"].mean()

all_data = pd.concat([hicdiff_stats, corigami_stats], axis=1)

data_barplot = all_data.stack().reset_index().rename(columns={"level_0": "Chromosome", "level_1": "Model", 0: "FID"})
data_barplot["Chromosome"] = data_barplot["Chromosome"].apply(lambda x: x.capitalize())
data_barplot["Model"] = data_barplot["Model"].apply(lambda x: "HiCDiffusion" if(x == "fid") else "HiCDiffusion - E/D" if (x == "fid_cond") else "C.Origami")
fig = plt.figure(figsize=(8, 14), constrained_layout=True)
axs = fig.subplot_mosaic([['Row1', 'Row1', 'Row1', 'Row1'], ['Row2', 'Row2', 'Row2', 'Row2'], ['Row3', 'Row3', 'Row3', 'Row3'],
                          ['Row4', 'Row4', 'Row4', 'Row4'], ['Row5', 'Row5', 'Row5', 'Row5'], ['Row6', 'Row6', 'Extra', 'Extra']])

for i in range(0, 6):
    if(i == 5):
        df_now = data_barplot[data_barplot["Chromosome"].isin([f"Chr{str(i*4+1)}", f"Chr{str(i*4+2)}"])]
    else:
        df_now = data_barplot[data_barplot["Chromosome"].isin([f"Chr{str(i*4+1)}", f"Chr{str(i*4+2)}", f"Chr{str(i*4+3)}", f"Chr{str(i*4+4)}"])]
    sns.barplot(df_now["Chromosome"], df_now["FID"], hue=df_now["Model"], ax=axs["Row"+str(i+1)], palette="rocket")
    if(i < 5):
        axs["Row"+str(i+1)].legend_.remove()
axs["Extra"].axis('off')
plt.savefig("fig3_fid_sup.svg")
plt.close()

fig = plt.figure(figsize=(15, 10), constrained_layout=True)
axs = fig.subplot_mosaic([['HiCDiffusion', 'HiCDiffusion', 'HiCDiffusionED', 'HiCDiffusionED', 'COrigami', 'COrigami'],
                          ['Pearson', 'Pearson', 'Pearson', "FID", "FID", "FID"]])
axs["HiCDiffusion"].set_title("HiCDiffusion")
axs["HiCDiffusionED"].set_title("HiCDiffusion - E/D")
axs["COrigami"].set_title("C.Origami")

axs["HiCDiffusion"].set_xticks([])
axs["HiCDiffusionED"].set_xticks([])
axs["COrigami"].set_xticks([])

axs["HiCDiffusion"].set_yticks([])
axs["HiCDiffusionED"].set_yticks([])
axs["COrigami"].set_yticks([])


axs["HiCDiffusion"].imshow(np.load("out/gm12878/prediction/npy/chr8_8600000.npy"), cmap=color_map, vmin=0, vmax=5)
axs["HiCDiffusionED"].imshow(np.load("out/gm12878/prediction/npy/chr8_8600000.npy"), cmap=color_map, vmin=0, vmax=5)
axs["COrigami"].imshow(np.load("out/gm12878/prediction/npy/chr8_8600000.npy"), cmap=color_map, vmin=0, vmax=5)

df_avg_fid = pd.DataFrame.from_dict(average_fid, orient="index", columns=["FID"])
df_avg_fid["Model"] = df_avg_fid.index
sns.barplot(df_avg_fid["Model"], df_avg_fid["FID"], ax=axs["FID"], palette="rocket")
axs["FID"].set_title("Fréchet inception distance")


sns.violinplot(data=df, x="Model", y="Correlation", density_norm="count", cut=0, ax=axs["Pearson"], legend=False, palette="rocket")
axs["Pearson"].set_title("Correlation")

plt.savefig("fig3.svg")
