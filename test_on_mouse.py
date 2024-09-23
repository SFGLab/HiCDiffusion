import datasets
import lightning.pytorch as pl
from hicdiffusion_model import HiCDiffusion
from lightning.pytorch.callbacks import ModelSummary
import os
import shutil
from lightning.pytorch.loggers import WandbLogger
import time

def main():
    hic_filename = "4DNFIPNP9H9T.mcool"
    model_ckpt= "models/hicdiffusion_test_chr8_val_chr9/best_val_loss_hicdiffusion.ckpt"
    if(hic_filename != ""):
        filename_prefix = "_"+hic_filename
    else:
        filename_prefix = ""
    
    pl.seed_everything(1996)
    batch_size = 16
    
    test_model_folder = f"test_{hic_filename}/"

    genomic_data_module = datasets.GenomicDataModule("mm10.fa", "exclude_regions_mm10.bed", 500_000, batch_size, [], [f"chr{x}" for x in range(20)], hic_filename, ["chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8", "chr9", "chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19"])

    model = HiCDiffusion.load_from_checkpoint(model_ckpt, strict=False)

    logger = WandbLogger(project=f"HiCDiffusionTestMouse", log_model=True, name=f"Mouse test {hic_filename}")
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
    main()