import torch
import datasets

my_data_module = datasets.GenomicDataModule("hg00512_CTCF_pooled.5k.2.sig3Dinteractions.bedpe", "/mnt/raid/GRCh38_full_analysis_set_plus_decoy_hla.fa", "exclude_regions.bed")

