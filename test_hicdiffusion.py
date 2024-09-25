import datasets
import lightning.pytorch as pl
from hicdiffusion_model import HiCDiffusion
from lightning.pytorch.callbacks import ModelSummary
import os
import shutil
from lightning.pytorch.loggers import WandbLogger
import time
import argparse

def main(val_chr, test_chr, model_ckpt, hic_filename="", model_ed=None):
    if(hic_filename != ""):
        filename_prefix = "_"+hic_filename
    else:
        filename_prefix = ""
    pl.seed_everything(1996)
    batch_size = 16
    
    test_model_folder = "models/nhicdiffusion%s_test_%s_val_%s/predictions_test" % (filename_prefix, test_chr, val_chr)

    genomic_data_module = datasets.GenomicDataModule("GRCh38_full_analysis_set_plus_decoy_hla.fa", "exclude_regions.bed", 500_000, batch_size, [val_chr], [test_chr], hic_filename)
    if(model_ed is not None):
        model = HiCDiffusion.load_from_checkpoint(model_ckpt, encoder_decoder_model=model_ed)
    else:
        model = HiCDiffusion.load_from_checkpoint(model_ckpt)

    logger = WandbLogger(project=f"XDNHiCDiffusionTest{filename_prefix}", log_model=True, name=f"Test: {test_chr}, Val: {val_chr}")
    trainer = pl.Trainer(logger=logger, callbacks=[ModelSummary(max_depth=2)], devices=1, num_sanity_val_steps=0)
    
    if os.path.exists(test_model_folder) and os.path.isdir(test_model_folder):
        shutil.rmtree(test_model_folder)
        time.sleep(2)
    try:
        os.makedirs(test_model_folder, exist_ok=True)
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
    parser.add_argument('-v', '--val_chr', required=True)
    parser.add_argument('-t', '--test_chr', required=True)
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-me', '--model_ed', required=False)
    parser.add_argument('-f', '--hic_filename', required=False, default="")
    
    args = parser.parse_args()
    
    print("Running testing of HiCDiffusion. The configuration:", flush=True)
    print(args, flush=True)
    print(flush=True)
    
    main(args.val_chr, args.test_chr, args.model, args.hic_filename, args.model_ed)