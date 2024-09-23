import numpy as np
import comparison_datasets
from skimage.transform import resize
import torch
import glob, os
from torchmetrics.regression import PearsonCorrCoef
import pandas as pd
from torchmetrics.image.fid import FrechetInceptionDistance
import hicreppy.hicrep as hcr
from scipy import sparse
import hicreppy.utils.mat_process as cu

normal_chromosomes = ["chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8", "chr9", "chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21","chr22"]

window_size = 2_000_000
output_res = 10_000 # IT HAS TO BE ALSO RES OF BEDPE!!!
size_img = 256
length_to_input = 97152 # correction for input to network - easier to ooperate whith maxpooling when ^2


def normalize(A):
    A = A.view(-1, 256, 256)
    outmap_min, _ = torch.min(A, dim=1, keepdim=True)
    outmap_max, _ = torch.max(A, dim=1, keepdim=True)
    outmap = (A - outmap_min) / (outmap_max - outmap_min)
    return outmap.view(-1, 1, 256, 256)

def ptp(input):
    return input.max() - input.min()

eps = 1e-7

class DatasetWrapper():
    def __init__(self):
        self.comparison_dataset = {}
        for chromosome in normal_chromosomes:
            self.comparison_dataset[chromosome] = comparison_datasets.HiComparison()
            self.comparison_dataset[chromosome].load("hic/%s.npz" % chromosome)
    def get_real_y(self, chromosome, pos):
        return resize(torch.Tensor(self.comparison_dataset[chromosome].get(chromosome, pos-int(length_to_input/2), window_size+int(length_to_input/2), output_res)).to(torch.float), (size_img, size_img), anti_aliasing=True)

dataset_wrapper = DatasetWrapper()
fid_dict = {}
for chromosome in normal_chromosomes:
    pearson_results = []
    hcr_results = []
    fid = FrechetInceptionDistance(feature=64, normalize=True)
    for file in glob.glob(f"scripts/out/gm12878/prediction/npy/{chromosome}_*.npy"):
        pos = int(file.split(f"scripts/out/gm12878/prediction/npy/{chromosome}_")[1].split(".npy")[0])
        y_predicted = torch.tensor(np.load(file))
        y_real = torch.Tensor(dataset_wrapper.get_real_y(chromosome, pos))

        y_predicted_np = np.load(file)
        y_real_np = np.array(dataset_wrapper.get_real_y(chromosome, pos))

        fid.update(normalize(y_predicted).repeat(1, 3, 1, 1), real=False)
        fid.update(normalize(y_real).repeat(1, 3, 1, 1), real=True)

        y_real = y_real.view(-1)
        y_predicted = y_predicted.view(-1)

        if(ptp(y_real) == 0.0):
            y_real[0] += eps
            
        if(ptp(y_predicted) == 0.0):
            y_predicted[0] += eps

        pearson = PearsonCorrCoef()
        pearson_calculated = pearson(y_predicted, y_real)
        pearson_results.append((pos, pearson_calculated.item()))
        hcr_results.append((pos, hcr.get_scc(cu.smooth(sparse.csr_matrix(y_predicted_np), 2), cu.smooth(sparse.csr_matrix(y_real_np), 2), 16)))
    df = pd.DataFrame(pearson_results, columns =['pos', 'pearson'])
    df2 = pd.DataFrame(hcr_results, columns =['pos', 'scc'])
    df = df.merge(df2, on="pos")
    df.to_csv(f"scripts/results_csv/corigami/{chromosome}.csv", index=False)
    fid_dict[chromosome] = fid.compute().item()

df_fid = pd.DataFrame.from_dict(fid_dict, orient='index', columns=['fid'])
df_fid.to_csv("scripts/results_csv/corigami_results.csv")