import datasets
import lightning.pytorch as pl
from classification_model import ClassificationModel
from lightning.pytorch.callbacks import ModelSummary
import os
import shutil
from lightning.pytorch.loggers import WandbLogger
import time
import argparse
from lightning.pytorch.callbacks import ModelCheckpoint
import warnings # To suppress some warnings
 
# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="pyranges")

def main(val_chr, test_chr, encoder_decoder_model):
    pl.seed_everything(1996)
    batch_size = 16
    

    checkpoint_callback_best = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        dirpath=f"models2/classifier_hicdiffusion_test_{test_chr}_val_{val_chr}/",
        filename="best_val_loss_hicdiffusion",
        mode="min"
    )
    genomic_data_module = datasets.FeatureDataModule("GRCh38_full_analysis_set_plus_decoy_hla.fa", "exclude_regions.bed", 500_000, batch_size, [val_chr], [test_chr])

    model = ClassificationModel(encoder_decoder_model, val_chr, test_chr)

    logger = WandbLogger(project=f"Classifier_Subcomp_HiCDiffusion", log_model=True, name=f"Test: {test_chr}, Val: {val_chr}")
    trainer = pl.Trainer(logger=logger, gradient_clip_val=1, callbacks=[ModelSummary(max_depth=2), checkpoint_callback_best], max_epochs=100, num_sanity_val_steps=1, accumulate_grad_batches=2)

    logger.watch(model, log="all", log_freq=10)
    
    trainer.fit(model, datamodule=genomic_data_module)

if __name__ == "__main__":

    # parser = argparse.ArgumentParser(
    #                 prog='ProgramName',
    #                 description='What the program does',
    #                 epilog='Text at the bottom of help')
    # parser.add_argument('-v', '--val_chr', required=True)
    # parser.add_argument('-t', '--test_chr', required=True)
    # parser.add_argument('-m', '--model', required=True)
    
    # args = parser.parse_args()
    # print("Running training of Classifier HiCDiffusion. The configuration:", flush=True)
    # print(args, flush=True)
    # print(flush=True)
    
    main("chr10", "chr9", "models/nhicdiffusion_test_chr9_val_chr10/best_val_loss_encoder_decoder.ckpt")