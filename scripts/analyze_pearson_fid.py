import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
df = df.rename(columns={"chr": "Chromosome", "pearson": "Pearson Correlation"})


fig = plt.figure(figsize=(8, 14), constrained_layout=True)
axs = fig.subplot_mosaic([['Row1', 'Row1', 'Row1', 'Row1'], ['Row2', 'Row2', 'Row2', 'Row2'], ['Row3', 'Row3', 'Row3', 'Row3'],
                          ['Row4', 'Row4', 'Row4', 'Row4'], ['Row5', 'Row5', 'Row5', 'Row5'], ['Row6', 'Row6', 'Extra', 'Extra']])

for i in range(0, 6):
    if(i == 5):
        df_now = df[df["Chromosome"].isin([f"Chr{str(i*4+1)}", f"Chr{str(i*4+2)}"])]
    else:
        df_now = df[df["Chromosome"].isin([f"Chr{str(i*4+1)}", f"Chr{str(i*4+2)}", f"Chr{str(i*4+3)}", f"Chr{str(i*4+4)}"])]
    sns.violinplot(data=df_now, x="Chromosome", y="Pearson Correlation", hue="Model", density_norm="count", cut=0, ax=axs["Row"+str(i+1)], legend=False)
    axs["Row"+str(i+1)].set_ylim([0, 1])
    if(i < 5):
        axs["Row"+str(i+1)].legend_.remove()
axs["Extra"].axis('off')
plt.savefig("fig3_pearson_sup.png")
plt.close()

hicdiff_stats = pd.read_csv("results_csv/hicdiffusion_results.csv")
hicdiff_stats["chr"] = hicdiff_stats["Name"].apply(lambda x: x.split("Test: ")[1].split(",")[0])
hicdiff_stats = hicdiff_stats[["chr", "fid", "fid_cond"]]
hicdiff_stats = hicdiff_stats.set_index("chr")

corigami_stats = pd.read_csv("results_csv/corigami_results.csv", index_col=0)
corigami_stats = corigami_stats.rename(columns={"fid": "fid_corigami"})

all_data = pd.concat([hicdiff_stats, corigami_stats], axis=1)

data_barplot = all_data.stack().reset_index().rename(columns={"level_0": "Chromosome", "level_1": "Model", 0: "FID"})
data_barplot["Chromosome"] = data_barplot["Chromosome"].apply(lambda x: x.capitalize())
data_barplot["Model"] = data_barplot["Model"].apply(lambda x: "HiCDiffusion" if(x == "fid") else "HiCDiffusion - Encoder/Decoder" if (x == "fid_cond") else "C.Origami")
fig = plt.figure(figsize=(8, 14), constrained_layout=True)
axs = fig.subplot_mosaic([['Row1', 'Row1', 'Row1', 'Row1'], ['Row2', 'Row2', 'Row2', 'Row2'], ['Row3', 'Row3', 'Row3', 'Row3'],
                          ['Row4', 'Row4', 'Row4', 'Row4'], ['Row5', 'Row5', 'Row5', 'Row5'], ['Row6', 'Row6', 'Extra', 'Extra']])

for i in range(0, 6):
    if(i == 5):
        df_now = data_barplot[data_barplot["Chromosome"].isin([f"Chr{str(i*4+1)}", f"Chr{str(i*4+2)}"])]
    else:
        df_now = data_barplot[data_barplot["Chromosome"].isin([f"Chr{str(i*4+1)}", f"Chr{str(i*4+2)}", f"Chr{str(i*4+3)}", f"Chr{str(i*4+4)}"])]
    sns.barplot(df_now["Chromosome"], df_now["FID"], hue=df_now["Model"], ax=axs["Row"+str(i+1)])
    if(i < 5):
        axs["Row"+str(i+1)].legend_.remove()
axs["Extra"].axis('off')
plt.savefig("fig3_fid_sup.png")
plt.close()


