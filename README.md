# HiCDiffusion

Table of Contents
=================

* [What is HiCDiffusion?](#what-is-consensusv)
* [Citation](#citation)
* [Data](#data)
* [Requirements](#requirements)
* [Training and testing (advanced)](#training-and-testing-(advanced))
* [Predicting](#predicting)
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

The pretrained HiCDiffusion models (recommended - can be used for predictions with minimal effort, training and testing the model requires configuration of wandb) can be downloaded from zenodo: https://zenodo.org/records/13840733

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

Furthermore, you need to create wandb.ai account, and set it up:
https://docs.wandb.ai/quickstart/

Doing so will provide you with all the metrics, and predicted examples in a friendly environment. Since it's very important to check for those metrics during training & testing, you need to do that.

However, if you wish to only predict heatmaps using a pre-trained model, you don't need to do that.


## Training and testing (advanced)

To train the model, we first need to train the encoder-decoder using following command (you can add -f parameter with .mcool filename that will be used for training - otherwise you must have data mentioned in Data section):
```
python train_hicdiffusion_encoder_decoder.py -t chr8 -v chr9
```

Furthermore, you need to train the actual HiCDiffusion model, using the obtained encoder-decoder model, using the following command:

```
python train_hicdiffusion.py -t chr8 -v chr9 --model path_to_model/encoder_decoder_model.ckpt
```

We can then start testing using:

```
python test_hicdiffusion.py -t chr8 -v chr9 --model path_to_model/hicdiffusion_model.ckpt --me path_to_model/encoder_decoder_model.ckpt
```

## Predicting

To predict a sequence, you can use simple script that takes 2Mbp sequence (2_097_152bp to be precise), and predicts the output given the models. An example run would be:
```
python predict_hicdiffusion.py -s example_seq.txt -m path_to_model/hicdiffusion_model.ckpt -me path_to_model/encoder_decoder_model.ckpt
```

Additionally, one can predict using reference genome (remember to download GRCh38_full_analysis_set_plus_decoy_hla.fa - as explained in Data section!) positions:

```
python predict_hicdiffusion.py -c chr9 -p 15000000 -m path_to_model/hicdiffusion_model.ckpt -me path_to_model/encoder_decoder_model.ckpt
```

## Parameters

Parameters for all the scripts from the pipeline are:

Short option | Long option | Description
-------------- | --------------- | ---------------
-f | --hic_filename | .mcool file that will be used as the dataset. Without this parameter, you need data from the C.Origami paper (it will try to perform a comparison based on their data).
-t | --test_chr | Test chromosome that the pipeline will use only for the testing in last stage.
-v | --val_chr | Validation chromosome that the pipeline will use for determining best model (based on loss on val set). It is not used in training.

Additionally, in train_hicdiff.py and test_hicdiff.py we have:

Short option | Long option | Description
-------------- | --------------- | ---------------
-m | --model | Path to the model (in case of training HiCDiffusion it is encoder-decoder model, in case of testing, it's final diffusion model)
-me | --model_ed | Path to the encoder/decoder model (in case of testing HiCDiffusion or prediction only!)
