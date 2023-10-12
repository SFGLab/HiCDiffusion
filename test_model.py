import torch
import datasets
import lightning.pytorch as pl
from diffusion_model import Interaction3DPredictorDiffusion
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

def main(jobid):
    pl.seed_everything(1996)
    
    batch_size = 16
    
    test_model_folder = "test_model_folder/"
    encoder_decoder_model = "final_model.ckpt" # ckpt 53 # NOT FINAL MODEL!!!

    genomic_data_module = datasets.GenomicDataModule("GRCh38_full_analysis_set_plus_decoy_hla.fa", "exclude_regions.bed", 500_000, batch_size)

    model = Interaction3DPredictorDiffusion.load_from_checkpoint(encoder_decoder_model)

    logger = WandbLogger(project="Interaction3DPredictorDiffusionTest", log_model=True)
    trainer = pl.Trainer(logger=logger, callbacks=[ModelSummary(max_depth=2)], devices=1, num_sanity_val_steps=0)
    
    if os.path.exists(test_model_folder) and os.path.isdir(test_model_folder):
        shutil.rmtree(test_model_folder)
        time.sleep(2)
    try:
        os.mkdir(test_model_folder)
    except OSError:
        pass

    logger.watch(model, log="all", log_freq=10)
    
    trainer.test(model, datamodule=genomic_data_module)

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