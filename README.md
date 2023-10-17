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

If you use ConsensuSV in your research, we kindly ask you to cite the following publication:

```
TBD
```

## Requirements

Requirements:
* torch
* lightning
* wandb
* torchvision
* pandas
* numpy
* denoising_diffusion_pytorch (https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main)
* biopython
* pyranges
* scikit-image
* scipy


The full way of training can be done using the following command:
```
python run_experiment.py 
```

This command trains the encoder-decoder architecture, then diffusion model built upon that, and then tests it.

## Parameters

Parameters for the default pipeline (as well as ALL the other training scripts) are:

Short option | Long option | Description
-------------- | --------------- | ---------------
-t | --test_chr | Test chromosome that the pipeline will use only for the testing in last stage.
-v | --val_chr | Validation chromosome that the pipeline will use for determining best model (based on loss on val set). It is not used in training.

Additionally, in train_hicdiff.py and test_hicdiff.py we have:

Short option | Long option | Description
-------------- | --------------- | ---------------
-m | --model | Path to the model (in case of training HiCDiffusion it is encoder-decoder model, in case of testing, it's final diffusion model)