# HiCDiffusion

Table of Contents
=================

* [What is HiCDiffusion?](#what-is-consensusv)
* [Citation](#citation)
* [Requirements](#requirements)
* [Parameters](#parameters)

## What is HiCDiffusion?

Diffusion-based, from-sequence Hi-C matrices predictor.

## Citation

If you use HiCDiffusion in your research, we kindly ask you to cite the following publication:

```
TBD
```
## Data

To test the software using data from C.Origami, download the data from there:
https://zenodo.org/record/7226561/files/corigami_data_gm12878_add_on.tar.gz?download=1
And get the hic folder to the main folder of the software. You should also get reference genome and put it into the main folder, e.g., GRCh38_full_analysis_set_plus_decoy_hla.fa (which can be obtained from 1000 Genomes project: https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/)


## Requirements

The requirements are listed in requirements.txt. You can install a new environment using command:
```
conda create --name hicdiffusion -c conda-forge python=3.11
```
And then activate the environment & install the requirements using:
```
conda activate hicdiffusion
pip install -r requirements.txt
```

The full way of training can be done using the following command:
```
python run_experiment.py 
```

This command trains the encoder-decoder architecture, then diffusion model built upon that, and then tests it.

## Parameters

Parameters for the default pipeline (as well as ALL the other training scripts) are:

Short option | Long option | Description
-------------- | --------------- | ---------------
-f | --hic_filename | .mcool file that will be used as the dataset. Without this parameter, you need data from the C.Origami paper (it will try to perform a comparison based on their data).
-t | --test_chr | Test chromosome that the pipeline will use only for the testing in last stage.
-v | --val_chr | Validation chromosome that the pipeline will use for determining best model (based on loss on val set). It is not used in training.

Additionally, in train_hicdiff.py and test_hicdiff.py we have:

Short option | Long option | Description
-------------- | --------------- | ---------------
-m | --model | Path to the model (in case of training HiCDiffusion it is encoder-decoder model, in case of testing, it's final diffusion model)
