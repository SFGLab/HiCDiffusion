import torch
import datasets
import lightning.pytorch as pl
from hicdiff_model import HiCDiff
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import os
import numpy as np
import shutil
import pyranges as pr
import pandas as pd
from tqdm import tqdm
from lightning.pytorch.loggers import WandbLogger
import time
from datetime import datetime
import argparse
from lightning.pytorch.callbacks import ModelCheckpoint

def main(val_chr, test_chr, encoder_decoder_model):
    pl.seed_everything(1996)
    batch_size = 16
    
    predictions_validation = "models/hicdiff_test_%s_val_%s/predictions_validation" % (test_chr, val_chr)

    checkpoint_callback_best = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        dirpath=f"models/hicdiff_test_{test_chr}_val_{val_chr}/",
        filename="best_val_loss_hicdiff",
        mode="min"
    )
    genomic_data_module = datasets.GenomicDataModule("GRCh38_full_analysis_set_plus_decoy_hla.fa", "exclude_regions.bed", 500_000, batch_size, [val_chr], [test_chr])

    model = HiCDiff(predictions_validation, encoder_decoder_model)

    logger = WandbLogger(project="HiCDiff", log_model=True, name=f"Test: {test_chr}, Val: {val_chr}")
    trainer = pl.Trainer(logger=logger, gradient_clip_val=1, detect_anomaly=True, callbacks=[ModelSummary(max_depth=2), checkpoint_callback_best], max_epochs=100, num_sanity_val_steps=1, accumulate_grad_batches=2)
    
    if(trainer.global_rank == 0):
        if os.path.exists(predictions_validation) and os.path.isdir(predictions_validation):
            shutil.rmtree(predictions_validation)
            time.sleep(2)
        try:
            os.mkdir(predictions_validation)
        except OSError:
            pass

    logger.watch(model, log="all", log_freq=10)
    
    trainer.fit(model, datamodule=genomic_data_module)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-v', '--val_chr', required=True)
    parser.add_argument('-t', '--test_chr', required=True)
    parser.add_argument('-m', '--model', required=True)
    
    args = parser.parse_args()
    
    print("Running training of HiCDiff. The configuration:", flush=True)
    print(args, flush=True)
    print(flush=True)
    
    main(args.val_chr, args.test_chr, args.model)