import datasets
import lightning.pytorch as pl
from hicdiffusion_encoder_decoder_model import HiCDiffusionEncoderDecoder
from lightning.pytorch.callbacks import ModelSummary
import os
import shutil
from lightning.pytorch.loggers import WandbLogger
import time
import argparse
from lightning.pytorch.callbacks import ModelCheckpoint

def main(val_chr, test_chr, hic_filename):
    if(hic_filename != ""):
        filename_prefix = "_"+hic_filename
    else:
        filename_prefix = ""
    pl.seed_everything(1996)
    
    batch_size = 2
    
    predictions_validation = "models/hicdiffusion%s_test_%s_val_%s/predictions_encoder_decoder" % (filename_prefix, test_chr, val_chr)

    checkpoint_callback_best = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        dirpath=f"models/nhicdiffusion{filename_prefix}_test_{test_chr}_val_{val_chr}/",
        filename="best_val_loss_encoder_decoder",
        mode="min"
    )
    genomic_data_module = datasets.GenomicDataModule("GRCh38_full_analysis_set_plus_decoy_hla.fa", "exclude_regions.bed", 500_000, batch_size, [val_chr], [test_chr], hic_filename)

    model = HiCDiffusionEncoderDecoder(predictions_validation, val_chr, test_chr)

    logger = WandbLogger(project=f"HiCDiffusionEncoderDecoder{filename_prefix}", log_model=True, name=f"Test: {test_chr}, Val: {val_chr}")
    trainer = pl.Trainer(logger=logger, gradient_clip_val=1, callbacks=[ModelSummary(max_depth=2), checkpoint_callback_best], max_epochs=50, num_sanity_val_steps=1, accumulate_grad_batches=2)
    
    if(trainer.global_rank == 0):
        if os.path.exists(predictions_validation) and os.path.isdir(predictions_validation):
            shutil.rmtree(predictions_validation)
            time.sleep(2)
        try:
            os.makedirs(predictions_validation, exist_ok=True)
        except OSError:
            pass

    logger.watch(model, log="all", log_freq=1)
    
    trainer.fit(model, datamodule=genomic_data_module)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-v', '--val_chr', required=True)
    parser.add_argument('-t', '--test_chr', required=True)
    parser.add_argument('-f', '--hic_filename', required=False, default="")
    
    args = parser.parse_args()
    
    print("Running training of HiCDiffusionEncoderDecoder. The configuration:", flush=True)
    print(args, flush=True)
    print(flush=True)
    
    main(args.val_chr, args.test_chr, args.hic_filename)
