import torch
import datasets
import lightning.pytorch as pl
from model import Interaction3DPredictor
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

min_to_be_positive = 1
produce_heatmaps = True
predict = True

def main(jobid):
    # dt = datetime.now()
    # ts = str(int(datetime.timestamp(dt)))
    predictions_validation = "predictions/predictions_validation_"+jobid
    predictions_final = "predictions/predictions_final_"+jobid
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

    genomic_data_module = datasets.GenomicDataModule("hg00512_CTCF_pooled.5k.2.sig3Dinteractions.bedpe", "GRCh38_full_analysis_set_plus_decoy_hla.fa", "exclude_regions.bed")

    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0, patience=20, verbose=False, mode="min")

    model = Interaction3DPredictor(predictions_validation, predictions_final)
    logger = WandbLogger(project="Interaction3DPredictor", log_model=True)
    trainer = pl.Trainer(profiler="simple", logger=logger, gradient_clip_val=1, detect_anomaly=True, callbacks=[ModelSummary(max_depth=2), early_stop_callback], max_epochs=1)
    logger.watch(model, log="all", log_freq=10)
    trainer.fit(model, datamodule=genomic_data_module)

    predictions = trainer.predict(model, datamodule=genomic_data_module)

    # all_interactions = []
    # for prediction_batch, real_data_batch in tqdm(zip(predictions, iter(genomic_data_module.predict_dataloader()))):
    #     for prediction, real_data, real_chr, real_pos, real_end in zip(prediction_batch, real_data_batch[1], real_data_batch[2][0], real_data_batch[2][1], real_data_batch[2][2]):
    #         for interaction in (prediction >= min_to_be_positive).nonzero():
    #             interaction_real_starts = interaction*datasets.output_res+real_pos
    #             all_interactions.append((real_chr, interaction_real_starts[0].item(), interaction_real_starts[0].item()+datasets.output_res, interaction_real_starts[1].item(), interaction_real_starts[1].item()+datasets.output_res, prediction[interaction[0].item(), interaction[1].item()].item()))
    # all_interactions_df = pd.DataFrame(all_interactions, columns=["chr", "pos1", "end1", "pos2", "end2", "score"])
    # all_interactions_df = all_interactions_df.groupby(["chr", "pos1", "end1", "pos2", "end2"]).agg(count=('score', 'size'), mean=('score', 'mean')).reset_index()
    # # optional - filter out based on count
    # # add code here
    # # end of optional
    # all_interactions_df['mean'] /= datasets.scaling_factor
    # all_interactions_df['score'] = all_interactions_df['mean'].astype(int)
    # all_interactions_df['chr1'] = all_interactions_df['chr']
    # all_interactions_df['chr2'] = all_interactions_df['chr']
    # all_interactions_df = all_interactions_df[['chr1', 'pos1', 'end1', 'chr2', 'pos2', 'end2', 'score']]
    # all_interactions_df = all_interactions_df[all_interactions_df["score"] > 1]
    # all_interactions_df.to_csv('predicted.bedpe', sep='\t', index=False)

if __name__ == "__main__":
    # args = ...  # you can use your CLI parser of choice, or the `LightningCLI`
    # TRAIN
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