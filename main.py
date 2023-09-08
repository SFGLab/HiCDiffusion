import torch
import datasets
import lightning.pytorch as pl
from model import Interaction3DPredictor
torch.set_float32_matmul_precision('medium')

def main(args=None):
    genomic_data_module = datasets.GenomicDataModule("hg00512_CTCF_pooled.5k.2.sig3Dinteractions.bedpe", "/mnt/raid/GRCh38_full_analysis_set_plus_decoy_hla.fa", "exclude_regions.bed")

    model = Interaction3DPredictor()
    #trainer = pl.Trainer(accelerator="gpu", devices=2, num_nodes=2, strategy="ddp")
    trainer = pl.Trainer(gradient_clip_val=0.5)#, accumulate_grad_batches=8)
    trainer.fit(model, genomic_data_module)


if __name__ == "__main__":
    # args = ...  # you can use your CLI parser of choice, or the `LightningCLI`
    # TRAIN
    main()