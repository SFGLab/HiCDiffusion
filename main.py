import torch
import datasets
import lightning.pytorch as pl
from model import Interaction3DPredictor
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import os

torch.set_float32_matmul_precision('medium')

def main(args=None):
    genomic_data_module = datasets.GenomicDataModule("hg00512_CTCF_pooled.5k.2.sig3Dinteractions.bedpe", "/mnt/raid/GRCh38_full_analysis_set_plus_decoy_hla.fa", "exclude_regions.bed")

    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.00, patience=2, verbose=False, mode="min")


    model = Interaction3DPredictor()
    #trainer = pl.Trainer(accelerator="gpu", devices=2, num_nodes=2, strategy="ddp")
    trainer = pl.Trainer(gradient_clip_val=0.5, overfit_batches=1, callbacks=[ModelSummary(max_depth=2), early_stop_callback])#, accumulate_grad_batches=8)
    trainer.fit(model, datamodule=genomic_data_module)
    predictions = trainer.predict(ckpt_path='best', datamodule=genomic_data_module)
    for prediction, real_data in zip(predictions, iter(genomic_data_module.predict_dataloader())):
        # save prediction & real_data[1] as real_data[2]
        pass
    #os.mkdir("real_")
    pass

if __name__ == "__main__":
    # args = ...  # you can use your CLI parser of choice, or the `LightningCLI`
    # TRAIN
    main()