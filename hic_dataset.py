# class from: https://github.com/tanjimin/C.Origami/blob/main/src/corigami/data/data_feature.py
# made for comparison with the c.origami to compare with HiC data

import numpy as np
import cooler

class HiCDataset():
    def __init__(self, hic_file_name, res):
        self.hic_dataset = cooler.Cooler(f"{hic_file_name}::/resolutions/{res}")
        if("chr" in self.hic_dataset.chromnames[0]):
            self.remove_chr = False
        else:
            self.remove_chr = True

    def get(self, chromosome, start, window = 2097152, res = 10000):
        if(self.remove_chr):
            chromosome = chromosome.replace("chr", "")
        return np.log(self.hic_dataset.matrix(field="count", balance=None).fetch(f"{chromosome}:{start}-{start+window}")+1)
