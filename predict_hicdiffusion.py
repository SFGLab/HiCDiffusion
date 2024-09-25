import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import argparse
import os
import torch
from datasets import GenomicDataSet
import numpy
import datasets
import lightning.pytorch as pl
from hicdiffusion_model import HiCDiffusion
from lightning.pytorch.callbacks import ModelSummary
import os
import shutil
from lightning.pytorch.loggers import WandbLogger
import time
import argparse
import matplotlib.pyplot as plt
import matplotlib
import re
import torch.nn.functional as Fun
from Bio import SeqIO

unwanted_chars = "U|R|Y|K|M|S|W|B|D|H|V|N"

def sequence_to_onehot(sequence):
    sequence = re.sub(unwanted_chars, "N", sequence).replace("A", "0").replace("C", "1").replace("T", "2").replace("G", "3").replace("N", "4")
    sequence_list = list(sequence)
    sequence_list_int = list(map(int, sequence_list))
    sequence_encoding = Fun.one_hot(torch.Tensor(sequence_list_int).to(torch.int64), 5).to(torch.float)
    return torch.transpose(sequence_encoding, 0, 1)
def create_image(file_name, y_pred):
        color_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red"])
        file_name = "output"
        fig = plt.figure(figsize=(4, 5), constrained_layout=True)
        axs = fig.subplot_mosaic([['TopLeft']])

        axs["TopLeft"].set_title('Predicted')
        axs["TopLeft"].imshow(y_pred, cmap=color_map, vmin=0, vmax=5)
        plt.savefig(f"{file_name}.png", dpi=400)
        plt.cla()

def main(seq, chr, start, model_ckpt, model_ed_ckptt):
    pl.seed_everything(1996)
    if(seq is not None):
        with open(seq, 'r') as file:
            seq = file.read().replace('\n', '')
    else:
        start = int(start)        
        with open("GRCh38_full_analysis_set_plus_decoy_hla.fa") as handle:
            seq_records = SeqIO.parse(handle, "fasta")
            for record in seq_records:
                if not(record.id == chr): continue
                seq = str(record.seq)[start:start+2_097_152]
    seq = sequence_to_onehot(seq).reshape(1, 5, 2_097_152)
    model = HiCDiffusion.load_from_checkpoint(model_ckpt, encoder_decoder_model=model_ed_ckptt)
    
    model.eval()
    with torch.no_grad():
        y_pred = model(seq.to(model.device)).view(256, 256).cpu()
    create_image("output", y_pred)
    print("Prediction saved to output.png")
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-s', '--seq', required=False)
    parser.add_argument('-c', '--chr', required=False)
    parser.add_argument('-p', '--start', required=False)
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-me', '--model_ed', required=True)
    
    args = parser.parse_args()
    
    print("Running testing of HiCDiffusion. The configuration:", flush=True)
    print(args, flush=True)
    print(flush=True)
    if(args.seq is None and (args.chr is None or args.start is None)) or (args.seq is not None and args.chr is not None and args.start is not None):
        print("Wrong invocation!")
        exit(0)
    main(args.seq, args.chr, args.start, args.model, args.model_ed)