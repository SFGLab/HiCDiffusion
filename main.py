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
    
    predictions_validation = "predictions/predictions_validation_"+jobid
    predictions_final = "predictions/predictions_final_"+jobid
    encoder_decoder_model = "encoder_decoder.ckpt"

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="val_PearsonCorrCoef",
        mode="min"
    )

    genomic_data_module = datasets.GenomicDataModule("GRCh38_full_analysis_set_plus_decoy_hla.fa", "exclude_regions.bed", 50_000, batch_size)

    model = Interaction3DPredictorDiffusion(predictions_validation, predictions_final, encoder_decoder_model)

    logger = WandbLogger(project="Interaction3DPredictorDiffusion", log_model=True)
    trainer = pl.Trainer(logger=logger, gradient_clip_val=1, detect_anomaly=True, callbacks=[ModelSummary(max_depth=2), checkpoint_callback], max_epochs=300, num_sanity_val_steps=-1, accumulate_grad_batches=2)
    
    if(trainer.global_rank == 0):
        if os.path.exists(predictions_validation) and os.path.isdir(predictions_validation):
            shutil.rmtree(predictions_validation)
            time.sleep(2)
        try:
            os.mkdir(predictions_validation)
        except OSError:
            pass
        if os.path.exists(predictions_final) and os.path.isdir(predictions_final):
            shutil.rmtree(predictions_final)
            time.sleep(2)
        try:
            os.mkdir(predictions_final)
        except OSError:
            pass

    logger.watch(model, log="all", log_freq=10)
    
    trainer.fit(model, datamodule=genomic_data_module)

    predictions = trainer.predict(model, datamodule=genomic_data_module)

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