from Bio import SeqIO
import pandas as pd
from torch.utils.data import Dataset
import pyranges as pr
import numpy as np

normal_chromosomes = ["chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8", "chr9", "chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21","chr22"]
window_size = 2_000_000
slide_size = 100_000
output_res = 5_000 # IT HAS TO BE ALSO RES OF BEDPE!!!
class GenomicDataSet(Dataset):
    def __init__(self, bedpe_file, reference_genome_file, bed_exclude):

        self.bed_exclude = pr.PyRanges(pd.read_csv(bed_exclude, sep="\t", header=None, usecols=[*range(0, 3)], names=["Chromosome", "Start", "End"]))
        self.interactions = pd.read_csv(bedpe_file, sep="\t", header=None, usecols=[*range(0, 7)], names=["chr1", "pos1", "end1", "chr2", "pos2", "end2", "score"])

        reference_genome = self.load_reference(reference_genome_file)
        reference_genome = pr.PyRanges(reference_genome).subtract(self.bed_exclude)

        self.prepare_windows(reference_genome)

        pass

    def load_reference(self, reference_genome_file):
        self.chr_seq = dict()
        with open(reference_genome_file) as handle:
            seq_records = SeqIO.parse(handle, "fasta")
            for record in seq_records:
                if not(record.id in normal_chromosomes): continue
                self.chr_seq[record.id] = str(record.seq)
                if(record.id == "chr10" or record.id == "chr2"): break
            reference_genome = pd.DataFrame({"Chromosome": self.chr_seq.keys(), "Start": [0]*len(self.chr_seq.keys()), "End": [len(self.chr_seq[x]) for x in self.chr_seq.keys()]})
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
        interactions = self.interactions[self.interactions["chr1"] == self.interactions["chr2"]][self.interactions["chr1"] == window["Chromosome"]][self.interactions["pos1"] >= window["Start"]][self.interactions["pos2"] >= window["Start"]][self.interactions["pos1"] <= window["End"]][self.interactions["pos2"] <= window["End"]][self.interactions["end1"] >= window["Start"]][self.interactions["end2"] >= window["Start"]][self.interactions["end1"] <= window["End"]][self.interactions["end2"] <= window["End"]]
        if not(float.is_integer(window_size/output_res)):
            raise Exception("The window size is %s, and the output resolution is %s. The result of division is %s, which is not natural number. Please fix the dimensions." % (window_size, output_res, window_size/output_res))
        output_vector = np.zeros((int(window_size/output_res), int(window_size/output_res)))
        interactions_changed_coords = pd.DataFrame({"x": ((interactions["pos1"]-window["Start"])/output_res).astype(int), "y": ((interactions["pos2"]-window["Start"])/output_res).astype(int), "score": interactions["score"]})
        for _, row in interactions_changed_coords.iterrows():
            output_vector[row["x"], row["y"]] = row["score"]
        return output_vector.flatten()

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows.iloc[idx]
        return self.chr_seq[window["Chromosome"]][window["Start"]:window["End"]], self.get_interactions_in_window(window)