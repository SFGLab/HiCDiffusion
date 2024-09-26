import datasets
import lightning.pytorch as pl
from hicdiffusion_model import HiCDiffusion
from classification_model import ClassificationModel
from lightning.pytorch.callbacks import ModelSummary
import os
import shutil
from lightning.pytorch.loggers import WandbLogger
import time
import argparse

def main(val_chr, test_chr, model_ckpt, model_ed):
    pl.seed_everything(1996)
    batch_size = 16
    

    genomic_data_module = datasets.FeatureDataModule("GRCh38_full_analysis_set_plus_decoy_hla.fa", "exclude_regions.bed", 500_000, batch_size, [val_chr], [test_chr])
    model = ClassificationModel.load_from_checkpoint(model_ckpt, encoder_decoder_model=model_ed)

    logger = WandbLogger(project=f"Classifier_Subcomp_HiCDiffusion_Test", log_model=True, name=f"Test: {test_chr}, Val: {val_chr}")
    trainer = pl.Trainer(logger=logger, callbacks=[ModelSummary(max_depth=2)], devices=1, num_sanity_val_steps=0)

    logger.watch(model, log="all", log_freq=10)
    
    trainer.test(model, datamodule=genomic_data_module)

if __name__ == "__main__":

    # parser = argparse.ArgumentParser(
    #                 prog='ProgramName',
    #                 description='What the program does',
    #                 epilog='Text at the bottom of help')
    # parser.add_argument('-j', '--jobid', required=False)
    # parser.add_argument('-v', '--val_chr', required=True)
    # parser.add_argument('-t', '--test_chr', required=True)
    # parser.add_argument('-m', '--model', required=True)
    # parser.add_argument('-me', '--model_ed', required=False)
    # parser.add_argument('-f', '--hic_filename', required=False, default="")
    
    # args = parser.parse_args()
    
    # print("Running testing of HiCDiffusion. The configuration:", flush=True)
    # print(args, flush=True)
    # print(flush=True)
    
    #main(args.val_chr, args.test_chr, args.model, args.hic_filename, args.model_ed)
    main("chr10", "chr9", "models2/classifier_hicdiffusion_test_chr9_val_chr10/best_val_loss_hicdiffusion-v12.ckpt", "models/nhicdiffusion_test_chr9_val_chr10/best_val_loss_encoder_decoder.ckpt")