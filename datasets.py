from Bio import SeqIO
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pyranges as pr
import numpy as np
import torch
import lightning.pytorch as pl
import torch.nn.functional as Fun
import re
from scipy.ndimage.filters import gaussian_filter
import comparison_datasets
import hic_dataset
from skimage.transform import resize

normal_chromosomes = ["chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8", "chr9", "chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21","chr22"]
window_size = 2_000_000
output_res = 10_000 # IT HAS TO BE ALSO RES OF BEDPE!!!
unwanted_chars = "U|R|Y|K|M|S|W|B|D|H|V|N"
num_workers_loader = 8 # in case of .hic, each loader uses around 16GB ram
size_img = 256

class GenomicDataSet(Dataset):
    def __init__(self, reference_genome_file, bed_exclude, chromosomes, slide_size, hic_file_name=""):
        self.hic_dataset = {}
        if(hic_file_name != ""):
            hic_dataset_full = hic_dataset.HiCDataset(hic_file_name, output_res)
        for chromosome in chromosomes:
            if(hic_file_name == ""):
                self.hic_dataset[chromosome] = comparison_datasets.HiComparison()
                self.hic_dataset[chromosome].load("hic_4DNFIXVAKX9Q_npy_no_balance/%s.npz" % chromosome)
            else:
                self.hic_dataset[chromosome] = hic_dataset_full
                
                

        self.slide_size = slide_size

        bed_exclude_df = pd.read_csv(bed_exclude, sep="\t", header=None, usecols=[*range(0, 3)], names=["Chromosome", "Start", "End"])
        bed_exclude_df["Start"] = ((bed_exclude_df["Start"]/output_res).apply(np.floor)*output_res).astype(int)
        bed_exclude_df["End"] = ((bed_exclude_df["End"]/output_res).apply(np.ceil)*output_res).astype(int)
        self.bed_exclude = pr.PyRanges(bed_exclude_df)

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
            reference_genome = pd.DataFrame({"Chromosome": self.chr_seq.keys(), "Start": [1_100_000]*len(self.chr_seq.keys()), "End": [len(self.chr_seq[x])-1_100_000 for x in self.chr_seq.keys()]})
            return reference_genome
        
    def prepare_windows(self, reference_genome):
        all_chr_dfs = []
        for _, row in reference_genome.df.iterrows():
            starts = [*range(row["Start"], row["End"], self.slide_size)]
            ends = [x + window_size for x in starts]
            chr_df = pd.DataFrame({"Chromosome": [row["Chromosome"]]*len(starts), "Start": starts, "End": ends})
            all_chr_dfs.append(chr_df)
        
        windows_df = pd.concat(all_chr_dfs)
        self.windows = pr.PyRanges(windows_df).intersect(reference_genome).df
        self.windows = self.windows[self.windows["End"]-self.windows["Start"] == window_size]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows.iloc[idx]
        length_to_input = 97152 # correction for input to network - easier to ooperate whith maxpooling when ^2
        sequence = self.chr_seq[window["Chromosome"]][window["Start"]-int(length_to_input/2):window["End"]+int(length_to_input/2)]
        return self.sequence_to_onehot(sequence), resize(torch.Tensor(self.hic_dataset[window["Chromosome"]].get(window["Chromosome"], window["Start"]-int(length_to_input/2), window_size+int(length_to_input/2), output_res)).to(torch.float), (size_img, size_img), anti_aliasing=True), [window["Chromosome"], window["Start"], window["End"]]

    def sequence_to_onehot(self, sequence):
        sequence = re.sub(unwanted_chars, "N", sequence).replace("A", "0").replace("C", "1").replace("T", "2").replace("G", "3").replace("N", "4")
        sequence_list = list(sequence)
        sequence_list_int = list(map(int, sequence_list))
        sequence_encoding = Fun.one_hot(torch.Tensor(sequence_list_int).to(torch.int64), 5).to(torch.float)
        return torch.transpose(sequence_encoding, 0, 1)
    

class GenomicDataModule(pl.LightningDataModule):
    def __init__(self, reference_genome_file, bed_exclude, slide_size = 500_000, batch_size: int = 4, val_chr = ["chr9"], test_chr = ["chr8"], hic_file_name=""):
        super().__init__()
        self.reference_genome_file = reference_genome_file
        self.bed_exclude = bed_exclude
        self.batch_size = batch_size
        self.slide_size = slide_size
        self.val_chr = val_chr
        self.test_chr = test_chr
        self.hic_file_name=hic_file_name

    def setup(self, stage=None):
        self.genomic_train = GenomicDataSet(self.reference_genome_file, self.bed_exclude, [x for x in normal_chromosomes if x not in self.val_chr+self.test_chr], self.slide_size, self.hic_file_name)
        self.genomic_val = GenomicDataSet(self.reference_genome_file, self.bed_exclude, self.val_chr, self.slide_size, self.hic_file_name)
        self.genomic_test = GenomicDataSet(self.reference_genome_file, self.bed_exclude, self.test_chr, self.slide_size, self.hic_file_name)

    def train_dataloader(self):
        return DataLoader(self.genomic_train, batch_size=self.batch_size, num_workers=num_workers_loader, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.genomic_test, batch_size=self.batch_size, num_workers=num_workers_loader, shuffle=False)
    
    def val_dataloader(self):
        return DataLoader(self.genomic_val, batch_size=self.batch_size, num_workers=num_workers_loader, shuffle=False)