from Bio import SeqIO
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pyranges as pr
import numpy as np
import torch
import lightning.pytorch as pl
import torch.nn.functional as Fun
import re
import math
from scipy.ndimage.filters import gaussian_filter

normal_chromosomes = ["chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8", "chr9", "chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21","chr22"]
window_size = 1_000_000
slide_size = 100_000
output_res = 5_000 # IT HAS TO BE ALSO RES OF BEDPE!!!
unwanted_chars = "U|R|Y|K|M|S|W|B|D|H|V|N"
scaling_factor = 1_000
num_workers_loader = 8

class GenomicDataSet(Dataset):
    def __init__(self, bedpe_file, reference_genome_file, bed_exclude, chromosomes):
        bed_exclude_df = pd.read_csv(bed_exclude, sep="\t", header=None, usecols=[*range(0, 3)], names=["Chromosome", "Start", "End"])
        bed_exclude_df["Start"] = ((bed_exclude_df["Start"]/output_res).apply(np.floor)*output_res).astype(int)
        bed_exclude_df["End"] = ((bed_exclude_df["End"]/output_res).apply(np.ceil)*output_res).astype(int)
        self.bed_exclude = pr.PyRanges(bed_exclude_df)
        self.interactions = pd.read_csv(bedpe_file, sep="\t", header=None, usecols=[*range(0, 7)], names=["chr1", "pos1", "end1", "chr2", "pos2", "end2", "score"])

        reference_genome = self.load_reference(reference_genome_file, chromosomes)
        reference_genome = pr.PyRanges(reference_genome).subtract(self.bed_exclude)

        self.prepare_windows(reference_genome)

        pass

    def load_reference(self, reference_genome_file, chromosomes):
        self.chr_seq = dict()
        with open(reference_genome_file) as handle:
            seq_records = SeqIO.parse(handle, "fasta")
            for record in seq_records:
                if not(record.id in normal_chromosomes) or not(record.id in chromosomes): continue
                self.chr_seq[record.id] = str(record.seq)
            reference_genome = pd.DataFrame({"Chromosome": self.chr_seq.keys(), "Start": [100_000]*len(self.chr_seq.keys()), "End": [len(self.chr_seq[x])-100_000 for x in self.chr_seq.keys()]})
            return reference_genome
        
    def prepare_windows(self, reference_genome):
        all_chr_dfs = []
        for _, row in reference_genome.df.iterrows():
            starts = [*range(row["Start"], row["End"], slide_size)]
            ends = [x + window_size for x in starts]
            chr_df = pd.DataFrame({"Chromosome": [row["Chromosome"]]*len(starts), "Start": starts, "End": ends})
            all_chr_dfs.append(chr_df)
        
        windows_df = pd.concat(all_chr_dfs)
        self.windows = pr.PyRanges(windows_df).intersect(reference_genome).df
        self.windows = self.windows[self.windows["End"]-self.windows["Start"] == window_size]

    def get_interactions_in_window(self, window):
        interactions = self.interactions.loc[self.interactions["chr1"] == self.interactions["chr2"]].loc[self.interactions["chr1"] == window["Chromosome"]].loc[self.interactions["pos1"] >= window["Start"]].loc[self.interactions["pos2"] >= window["Start"]].loc[self.interactions["pos1"] <= window["End"]].loc[self.interactions["pos2"] <= window["End"]].loc[self.interactions["end1"] >= window["Start"]].loc[self.interactions["end2"] >= window["Start"]].loc[self.interactions["end1"] <= window["End"]].loc[self.interactions["end2"] <= window["End"]]
        if not(float.is_integer(window_size/output_res)):
            raise Exception("The window size is %s, and the output resolution is %s. The result of division is %s, which is not natural number. Please fix the dimensions." % (window_size, output_res, window_size/output_res))
        output_vector = np.zeros((int(window_size/output_res), int(window_size/output_res)))
        interactions_changed_coords = pd.DataFrame({"x": ((interactions["pos1"]-window["Start"])/output_res).astype(int), "y": ((interactions["pos2"]-window["Start"])/output_res).astype(int), "score": interactions["score"]})
        for _, row in interactions_changed_coords.iterrows():
            output_vector[row["x"], row["y"]] = row["score"]*scaling_factor
        output_vector = gaussian_filter(output_vector, sigma=5)
        return torch.Tensor(output_vector).to(torch.float)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows.iloc[idx]
        length_to_2 = 48576 # correction for input to network - easier to ooperate whith maxpooling when ^2
        sequence = self.chr_seq[window["Chromosome"]][window["Start"]-int(length_to_2/2):window["End"]+int(length_to_2/2)]
        return self.sequence_to_onehot(sequence), self.get_interactions_in_window(window), [window["Chromosome"], window["Start"], window["End"]]

    def sequence_to_onehot(self, sequence):
        sequence = re.sub(unwanted_chars, "N", sequence).replace("A", "0").replace("C", "1").replace("T", "2").replace("G", "3").replace("N", "4")
        sequence_list = list(sequence)
        sequence_list_int = list(map(int, sequence_list))
        sequence_encoding = Fun.one_hot(torch.Tensor(sequence_list_int).to(torch.int64), 5).to(torch.float)
        return torch.transpose(sequence_encoding, 0, 1)
    

class GenomicDataModule(pl.LightningDataModule):
    def __init__(self, bedpe_file, reference_genome_file, bed_exclude, batch_size: int = 16):
        super().__init__()
        self.bedpe_file = bedpe_file
        self.reference_genome_file = reference_genome_file
        self.bed_exclude = bed_exclude
        self.batch_size = batch_size

    def setup(self, stage=None):
        #self.genomic_train = GenomicDataSet(self.bedpe_file, self.reference_genome_file, self.bed_exclude, [x for x in normal_chromosomes if x not in ["chr9"]])
        self.genomic_train = GenomicDataSet(self.bedpe_file, self.reference_genome_file, self.bed_exclude, ["chr9"]) # overfitting scenario - just for tests
        self.genomic_val = GenomicDataSet(self.bedpe_file, self.reference_genome_file, self.bed_exclude, ["chr9"])

    def train_dataloader(self):
        return DataLoader(self.genomic_train, batch_size=self.batch_size, num_workers=num_workers_loader)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_val, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.genomic_val, batch_size=self.batch_size, num_workers=num_workers_loader, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.genomic_val, batch_size=self.batch_size, num_workers=num_workers_loader)
