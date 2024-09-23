# Loops calling adapted from:
# https://github.com/XiaoTaoWang/HiCPeaks

import lightning.pytorch as pl
from hicdiffusion_model import HiCDiffusion
import matplotlib.pyplot as plt
import matplotlib
import lightning.pytorch as pl
from hicdiffusion_model import HiCDiffusion
import datasets
import numpy as np
import math
import numpy as np
import matplotlib.patches as patches
import itertools
matplotlib.rcParams.update({'font.size': 35})
def _loops_value(axs, figure, y_pred_value, y_real_value, pred_loops, real_loops, chromosome, position, pos_end, l=False):
    color_map2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red"])


    axs[f"Pred{figure}"].set_title('Predicted value')
    axs[f"Pred{figure}"].imshow(y_pred_value, cmap=color_map2, vmin=0, vmax=np.max(y_pred_value))
    axs[f"Pred{figure}"].set_xlabel(f'{chromosome}:{position}-{pos_end}')
    axs[f"Pred{figure}"].get_xaxis().set_ticks([])
    axs[f"Pred{figure}"].get_yaxis().set_visible(False)

    for loop in pred_loops:
        #axs[f"Pred{figure}"].plot(loop[0], loop[1], marker='D', color='orange', markersize=18)
        rect = patches.Rectangle((loop[0]-5, loop[1]-5), 10, 10, linewidth=5, edgecolor='b', facecolor='none')
        axs[f"Pred{figure}"].add_patch(rect)
    axs[f"Real{figure}"].set_title('Real value')
    axs[f"Real{figure}"].imshow(y_real_value, cmap=color_map2, vmin=0, vmax=np.max(y_real_value))
    axs[f"Real{figure}"].set_xlabel(f'{chromosome}:{position}-{pos_end}') 
    axs[f"Real{figure}"].get_xaxis().set_ticks([])
    axs[f"Real{figure}"].get_yaxis().set_visible(False)
    for loop in real_loops:
        #axs[f"Real{figure}"].plot(loop[0], loop[1], marker='D', color='orange', markersize=18)
        rect = patches.Rectangle((loop[0]-5, loop[1]-5), 10, 10, linewidth=5, edgecolor='b', facecolor='none')
        axs[f"Real{figure}"].add_patch(rect)

def loops_value(params1, params2):

    y_pred_value1, y_real_value1, pred_loops1, real_loops1, chromosome1, position1, pos_end1 = params1
    y_pred_value2, y_real_value2, pred_loops2, real_loops2, chromosome2, position2, pos_end2 = params2

    fig = plt.figure(figsize=(18, 18), constrained_layout=True)
    fig.suptitle('Loops calling')
    if(position1 == position2):
        file_name = f"loops/{chromosome1}_{position1}_{pos_end1}"
    else:
        file_name = "loops"
    axs = fig.subplot_mosaic([['Pred1', 'Pred1', 'Pred2', 'Pred2'],
                              ['Pred1', 'Pred1', 'Pred2', 'Pred2'],
                              ['Real1', 'Real1', 'Real2', 'Real2'],
                              ['Real1', 'Real1', 'Real2', 'Real2']])

    _loops_value(axs, "1", y_pred_value1, y_real_value1, pred_loops1, real_loops1, chromosome1, position1, pos_end1)
    _loops_value(axs, "2", y_pred_value2, y_real_value2, pred_loops2, real_loops2, chromosome2, position2, pos_end2)
    # if not(pearsons is None):
    #     sns.violinplot(pearsons, ax=axs["Pearson"])
    #     axs[f'Pearson'].set_title(f'Insulation')
    plt.savefig(f"{file_name}.png", dpi=400)
    plt.savefig(f"{file_name}.svg", dpi=400)
    plt.cla()

def test_pixel(value, width, i, j, mini):
    enrichment = 1.2
    if(value[i,j] < mini*0.75):
        return False
    # left-right
    left = value[i-width:i,j]
    right = value[i+1:i+width+1,j]
    leftright = np.concatenate([left, right])
    leftright = np.mean(leftright)
    if(value[i,j] < leftright*enrichment):
        return False
    #top-bottom
    top = value[i,j-width:j]
    bottom = value[i,j+1:j+width+1]
    topbottom = np.concatenate([top, bottom])
    topbottom = np.mean(topbottom)
    if(value[i,j] < topbottom*enrichment):
        return False
    leftbottom = value[i:i+width+1,j-width:j+1]
    count_leftbottom = leftbottom.size-1
    leftbottom = (np.sum(leftbottom)-value[i,j])
    leftbottom /= count_leftbottom
    if(value[i,j] < leftbottom*enrichment):
        return False
    donut = value[i-width:i+width+1,j-width:j+width+1]
    count_donut = donut.size-1
    donut = (np.sum(donut)-value[i,j])/count_donut
    if(value[i,j] < donut*enrichment):
        return False
    return True
def join_pair(loops):
    for p, q in itertools.combinations(loops, 2):
        if math.sqrt(math.pow(p[0]-q[0], 2)+math.pow(p[1]-q[1], 2)) < 25:
            loops.remove(p)
            loops.remove(q)
            loops.add(((p[0]+q[0])/2, (p[1]+q[1])/2))
            return True
    return False
def get_loops(value):
    width = 25
    loops = set()
    mini = np.mean(value)
    for i in range(width, value.shape[0]-width+1):
        for j in range(width, value.shape[0]-width+1):
            if(i <= j+width):
                continue
            if(test_pixel(value, width, i, j, mini)):
                loops.add((i, j))
    while join_pair(loops):
        pass
    loops_rounded = []
    for loop in loops:
        loops_rounded.append((round(loop[0]), round(loop[1])))
    return loops_rounded
def main():
    val_chr = "chr9"
    test_chr = "chr8"
    model_ckpt= "models/hicdiffusion_test_chr8_val_chr9/best_val_loss_hicdiffusion.ckpt"
    pl.seed_everything(1996)

    model = HiCDiffusion.load_from_checkpoint(model_ckpt, strict=False)

    dl = datasets.GenomicDataModule("GRCh38_full_analysis_set_plus_decoy_hla.fa", "exclude_regions.bed", 500_000, 1, [val_chr], [test_chr])
    dl.setup()
    dl = dl.predict_dataloader()

    trainer = pl.Trainer(devices=1, num_sanity_val_steps=0)
    predictions = trainer.predict(model, dl)
    to_plot = []
    for value in predictions:
        #h = h5py.File("cworld-test_hg19_C-40000-raw.hdf5", 'r')
        #heatmap = h['interactions']
        if(1-np.count_nonzero(value[3][0][0])/(256*256) > 0.05 or float(sum(value[3][0][0][-1])) == 0 or float(sum(value[3][0][0][0])) == 0): # skip the ones with 5% gaps in HiC and the ones starting / ending with all 0s
            continue
        
        v_pred = (np.array(value[1][0][0]))
        v_real = (np.array(value[3][0][0]))

        loops_predicted = get_loops(np.exp(v_pred))
        loops_real = get_loops(np.exp(v_real))

        parameters = (v_pred, v_real, loops_predicted, loops_real, value[0][0][0], int(value[0][1][0]), int(value[0][2][0]))
        #loops_value(parameters, parameters)
        to_plot.append(parameters)
        pass
    loops_value(to_plot[0], to_plot[1])
if __name__ == "__main__":

    main()