import torch
import datasets
import lightning.pytorch as pl
from sequence_vae import Interaction3DPredictorSequenceVAE
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

def main(jobid):
    pl.seed_everything(1996)
    
    batch_size = 2

    genomic_data_module = datasets.GenomicDataModule("GRCh38_full_analysis_set_plus_decoy_hla.fa", "exclude_regions.bed", 500_000, batch_size)

    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0, patience=30, verbose=False, mode="min")

    model = Interaction3DPredictorSequenceVAE()
    logger = WandbLogger(project="Interaction3DPredictorSequenceVAE", log_model=True)
    trainer = pl.Trainer(logger=logger, gradient_clip_val=1, detect_anomaly=True, callbacks=[ModelSummary(max_depth=4), early_stop_callback], max_epochs=100, num_sanity_val_steps=-1, accumulate_grad_batches=4)

    logger.watch(model, log="all", log_freq=10)
    
    trainer.fit(model, datamodule=genomic_data_module)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-j', '--jobid', required=False)
    args = parser.parse_args()
    if(args.jobid):
        main(args.jobid)
    else:
        main()