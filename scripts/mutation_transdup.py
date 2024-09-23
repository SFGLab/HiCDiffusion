from Bio import SeqIO, motifs
import lightning.pytorch as pl
from hicdiffusion_model import HiCDiffusion
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
import lightning.pytorch as pl
from hicdiffusion_model import HiCDiffusion
from torch.utils.data import Dataset, DataLoader
import pyranges as pr
import pandas as pd
import torch.nn.functional as Fun
import re

matplotlib.rcParams.update({'font.size': 32})
length_to_input = 97152 # correction for input to network - easier to ooperate whith maxpooling when ^2

def create_image_fid_chart(y_real_value, y_real_value_pert, chromosome, position, pos_end):
    DASH_W = 4.0
    DASHES = (4, 4)
    y_real_value = y_real_value.view(256, 256).detach().numpy()
    y_real_value_pert = y_real_value_pert.view(256, 256).detach().numpy()
    color_map2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "white","red"])
    file_name = "transdup"
    fig = plt.figure(figsize=(30, 10), constrained_layout=True)
    axs = fig.subplot_mosaic([['Gauss_0','Gauss_1','Gauss_2']])

    fig.suptitle('Modeling transduplication of %s:%s-%s' % (chromosome, str(position), str(pos_end)))

    axs["Gauss_0"].set_title('Raw model output')
    axs["Gauss_0"].imshow(y_real_value, cmap=color_map2, vmin=-5, vmax=5)
    axs["Gauss_0"].set_xlabel(f'chr8:32100000-34100000')
    axs["Gauss_0"].get_xaxis().set_ticks([])
    axs["Gauss_0"].get_yaxis().set_visible(False)
    # copied region
    axs["Gauss_0"].axvline(x=(1_350_000+int(length_to_input/2))/8192, color='b', ls="--", dashes=DASHES, lw=DASH_W, ymin=1, ymax=0) 
    axs["Gauss_0"].axvline(x=(1_550_000+int(length_to_input/2))/8192, color='b', ls="--", dashes=DASHES, lw=DASH_W, ymin=1, ymax=0) 
    axs["Gauss_0"].axhline(y=(1_350_000+int(length_to_input/2))/8192, color='b', ls="--", dashes=DASHES, lw=DASH_W) 
    axs["Gauss_0"].axhline(y=(1_550_000+int(length_to_input/2))/8192, color='b', ls="--", dashes=DASHES, lw=DASH_W) 


    axs["Gauss_1"].set_title('Transduplication of 200kbp ')
    axs["Gauss_1"].imshow(y_real_value_pert, cmap=color_map2, vmin=-5, vmax=5)
    axs["Gauss_1"].set_xlabel(f'chr8:33450000-33650000 to \nchr8:32750000-32950000')
    axs["Gauss_1"].get_xaxis().set_ticks([])
    axs["Gauss_1"].get_yaxis().set_visible(False)
    
    axs["Gauss_1"].axvline(x=(1_350_000+int(length_to_input/2))/8192, color='b', ls="--", dashes=DASHES, lw=DASH_W, ymin=1, ymax=0) 
    axs["Gauss_1"].axvline(x=(1_550_000+int(length_to_input/2))/8192, color='b', ls="--", dashes=DASHES, lw=DASH_W, ymin=1, ymax=0) 
    axs["Gauss_1"].axvline(x=(1_350_000-700_000+int(length_to_input/2))/8192, color='g', ls="--", dashes=DASHES, lw=DASH_W, ymin=1, ymax=0) 
    axs["Gauss_1"].axvline(x=(1_550_000-700_000+int(length_to_input/2))/8192, color='g', ls="--", dashes=DASHES, lw=DASH_W, ymin=1, ymax=0) 
    axs["Gauss_1"].axhline(y=(1_350_000+int(length_to_input/2))/8192, color='b', ls="--", dashes=DASHES, lw=DASH_W) 
    axs["Gauss_1"].axhline(y=(1_550_000+int(length_to_input/2))/8192, color='b', ls="--", dashes=DASHES, lw=DASH_W) 
    axs["Gauss_1"].axhline(y=(1_350_000-700_000+int(length_to_input/2))/8192, color='g', ls="--", dashes=DASHES, lw=DASH_W) 
    axs["Gauss_1"].axhline(y=(1_550_000-700_000+int(length_to_input/2))/8192, color='g', ls="--", dashes=DASHES, lw=DASH_W) 

    axs["Gauss_2"].set_title('Transduplication of 200kbp')
    axs["Gauss_2"].set_xlabel(f'Differential heatmap')
    axs["Gauss_2"].get_xaxis().set_ticks([])
    axs["Gauss_2"].get_yaxis().set_visible(False)
    axs["Gauss_2"].imshow(y_real_value_pert-y_real_value, cmap=color_map2, vmin=-2, vmax=2)
    axs["Gauss_2"].axvline(x=(1_350_000+int(length_to_input/2))/8192, color='b', ls="--", dashes=DASHES, lw=DASH_W, ymin=1, ymax=0) 
    axs["Gauss_2"].axvline(x=(1_550_000+int(length_to_input/2))/8192, color='b', ls="--", dashes=DASHES, lw=DASH_W, ymin=1, ymax=0) 
    axs["Gauss_2"].axvline(x=(1_350_000-700_000+int(length_to_input/2))/8192, color='g', ls="--", dashes=DASHES, lw=DASH_W, ymin=1, ymax=0) 
    axs["Gauss_2"].axvline(x=(1_550_000-700_000+int(length_to_input/2))/8192, color='g', ls="--", dashes=DASHES, lw=DASH_W, ymin=1, ymax=0) 
    axs["Gauss_2"].axhline(y=(1_350_000+int(length_to_input/2))/8192, color='b', ls="--", dashes=DASHES, lw=DASH_W) 
    axs["Gauss_2"].axhline(y=(1_550_000+int(length_to_input/2))/8192, color='b', ls="--", dashes=DASHES, lw=DASH_W) 
    axs["Gauss_2"].axhline(y=(1_350_000-700_000+int(length_to_input/2))/8192, color='g', ls="--", dashes=DASHES, lw=DASH_W) 
    axs["Gauss_2"].axhline(y=(1_550_000-700_000+int(length_to_input/2))/8192, color='g', ls="--", dashes=DASHES, lw=DASH_W) 

    #fig.colorbar(for_scale, ax=list(axs.values()))
    plt.savefig(f"{file_name}.png", dpi=400)
    plt.savefig(f"{file_name}.svg", dpi=400)
    plt.cla()

window_size = 2_000_000
output_res = 10_000 # IT HAS TO BE ALSO RES OF BEDPE!!!
unwanted_chars = "U|R|Y|K|M|S|W|B|D|H|V|N"
num_workers_loader = 8 # in case of .hic, each loader uses around 16GB ram
size_img = 256
class PertubationDataset(Dataset):
    def __init__(self, reference_genome_file, chromosome, pos, changes):
        self.pos = pos
        self.chromosome = chromosome
        self.changes = changes
        self.load_reference(reference_genome_file, [chromosome])

        pass

    def load_reference(self, reference_genome_file, chromosomes):
        self.chr_seq = dict()
        with open(reference_genome_file) as handle:
            seq_records = SeqIO.parse(handle, "fasta")
            for record in seq_records:
                if not(record.id in chromosomes): continue
                self.chr_seq[record.id] = str(record.seq)
            reference_genome = pd.DataFrame({"Chromosome": self.chr_seq.keys(), "Start": [1_100_000]*len(self.chr_seq.keys()), "End": [len(self.chr_seq[x])-1_100_000 for x in self.chr_seq.keys()]})
            return reference_genome
        

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        length_to_input = 97152 # correction for input to network - easier to ooperate whith maxpooling when ^2
        sequence = self.chr_seq[self.chromosome][self.pos-int(length_to_input/2):self.pos+window_size+int(length_to_input/2)]
        if(idx == 0):
            pass # seq is ok
        else:
            # do changes
            sequence = list(sequence)
            for i in range(1_350_000+int(length_to_input/2), 1_550_000+int(length_to_input/2)):
                sequence[i-700_000] = sequence[i]
            sequence = ''.join(sequence)
            pass
        return self.sequence_to_onehot(sequence), torch.zeros(size_img, size_img), [self.chromosome, self.pos, self.pos+window_size]

    def sequence_to_onehot(self, sequence):
        sequence = re.sub(unwanted_chars, "N", sequence).replace("A", "0").replace("C", "1").replace("T", "2").replace("G", "3").replace("N", "4")
        sequence_list = list(sequence)
        sequence_list_int = list(map(int, sequence_list))
        sequence_encoding = Fun.one_hot(torch.Tensor(sequence_list_int).to(torch.int64), 5).to(torch.float)
        return torch.transpose(sequence_encoding, 0, 1)
    
def main():
    model_ckpt= "models/hicdiffusion_test_chr8_val_chr9/best_val_loss_hicdiffusion.ckpt"
    pl.seed_everything(2000)

    model = HiCDiffusion.load_from_checkpoint(model_ckpt, strict=False)

    dl = DataLoader(PertubationDataset("GRCh38_full_analysis_set_plus_decoy_hla.fa", "chr8", 32_100_000, []), batch_size=1, shuffle=False)

    trainer = pl.Trainer(devices=1, num_sanity_val_steps=0)
    predictions = trainer.predict(model, dl)

    value = predictions[0]
    value_pert = predictions[1]
    if(int(value[0][1][0].item()) == 32_100_000):
        create_image_fid_chart(value[1][0].view(1, 1, 256, 256), value_pert[1][0].view(1, 1, 256, 256), value[0][0][0], value[0][1][0].item(), value[0][2][0].item())
        pass
if __name__ == "__main__":

    main()