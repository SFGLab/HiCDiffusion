import torch
import datasets
import lightning.pytorch as pl
from model import Interaction3DPredictor
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
import pyranges as pr
import pandas as pd
from tqdm import tqdm
from lightning.pytorch.loggers import WandbLogger


min_to_be_positive = 1
produce_heatmaps = True
predict = True
torch.set_float32_matmul_precision('medium')

def main(args=None):
    genomic_data_module = datasets.GenomicDataModule("hg00512_CTCF_pooled.5k.2.sig3Dinteractions.bedpe", "GRCh38_full_analysis_set_plus_decoy_hla.fa", "exclude_regions.bed")

    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0, patience=3, verbose=False, mode="min")


    model = Interaction3DPredictor(genomic_data_module.batch_size)
    #trainer = pl.Trainer(accelerator="gpu", devices=2, num_nodes=2, strategy="ddp")
    logger = WandbLogger(project="Interaction3DPredictor", log_model=True)
    trainer = pl.Trainer(logger=logger, gradient_clip_val=1, detect_anomaly=True, callbacks=[ModelSummary(max_depth=2), early_stop_callback], max_epochs=100)#, accumulate_grad_batches=8)
    logger.watch(model, log="all", log_freq=10)
    trainer.fit(model, datamodule=genomic_data_module)
    if(predict):
        predictions = trainer.predict(ckpt_path='best', datamodule=genomic_data_module)
        if os.path.exists("predictions") and os.path.isdir("predictions"):
            shutil.rmtree("predictions")
        os.mkdir("predictions")
        all_interactions = []
        for prediction_batch, real_data_batch in tqdm(zip(predictions, iter(genomic_data_module.predict_dataloader()))):
            for prediction, real_data, real_chr, real_pos, real_end in zip(prediction_batch, real_data_batch[1], real_data_batch[2][0], real_data_batch[2][1], real_data_batch[2][2]):
                if(produce_heatmaps):
                    pd.DataFrame(prediction).to_csv("predictions/%s_%s_%s.npy" % (real_chr, real_pos.item(), real_end.item()))
                    plt.subplot(1, 3, 1)
                    plt.suptitle('Output - %s %s:%s' % (real_chr, real_pos.item(), real_end.item()))
                    plt.gca().set_title('Predicted')
                    plt.imshow(prediction, cmap='binary', interpolation='nearest')
                    plt.subplot(1, 3, 2)
                    plt.gca().set_title('Real')
                    plt.imshow(real_data, cmap='binary', interpolation='nearest')
                    plt.subplot(1, 3, 3)
                    plt.gca().set_title('Difference')
                    plt.imshow(real_data-prediction, cmap='binary', interpolation='nearest')
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig("predictions/%s_%s_%s.png" % (real_chr, real_pos.item(), real_end.item()), dpi=400)
                    plt.cla()
                for interaction in (prediction >= min_to_be_positive).nonzero(): # change real_data to prediction
                    interaction_real_starts = interaction*datasets.output_res+real_pos
                    all_interactions.append((real_chr, interaction_real_starts[0].item(), interaction_real_starts[0].item()+datasets.output_res, interaction_real_starts[1].item(), interaction_real_starts[1].item()+datasets.output_res, prediction[interaction[0].item(), interaction[1].item()].item()))
        all_interactions_df = pd.DataFrame(all_interactions, columns=["chr", "pos1", "end1", "pos2", "end2", "score"])
        all_interactions_df = all_interactions_df.groupby(["chr", "pos1", "end1", "pos2", "end2"]).agg(count=('score', 'size'), mean=('score', 'mean')).reset_index()
        # optional - filter out based on count
        # add code here
        # end of optional
        all_interactions_df['mean'] /= datasets.scaling_factor
        all_interactions_df['score'] = all_interactions_df['mean'].astype(int)
        all_interactions_df['chr1'] = all_interactions_df['chr']
        all_interactions_df['chr2'] = all_interactions_df['chr']
        all_interactions_df = all_interactions_df[['chr1', 'pos1', 'end1', 'chr2', 'pos2', 'end2', 'score']]
        all_interactions_df.to_csv('predicted.bedpe', sep='\t', index=False)

if __name__ == "__main__":
    # args = ...  # you can use your CLI parser of choice, or the `LightningCLI`
    # TRAIN
    main()