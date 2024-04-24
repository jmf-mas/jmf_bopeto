# Unsupervised data cleaning
This repository collects different unsupervised machine learning algorithms to detect anomalies.
## Implemented models
We have implemented the following models. Our implementations of ALAD, DeepSVDD, 
DROCC and MemAE closely follows the original implementations already available on GitHub.
- [x] [AutoEncoder]()

## Dependencies
A complete dependency list is available in requirements.txt.
We list here the most important ones:
- torch@1.10.2 with CUDA 11.3
- numpy
- pandas
- scikit-learn

## Installation
Assumes latest version of Anaconda was installed.
```
$ conda create --name [ENV_NAME] python=3.8
$ conda activate [ENV_NAME]
$ pip install -r requirements.txt
$ chmod +x main.sh
$ ./main.sh
```
Replace `[ENV_NAME]` with the name of your environment.


Our model contains the following parameters:
- `--batch-size`: size of a training batch (**required**)
- `--dataset`: name of the selected dataset. Choices are `Arrhythmia`, `KDD10`, `IDS2018`, `NSLKDD`, `USBIDS`, `Thyroid` (**required**).
- `-epochs`: number of training epochs (default=200)
Please note that datasets must be stored in `.npz` or `.mat` files. Use the preprocessing scripts within `data_process`
to generate these files.

## Example
To cleaning KDD dataset using Bopeto:
```
$ python cleaning.py --dataset kdd 
```
To train a DAGMM on the KDD dataset:
```
$ python detection.py --dataset kdd --model dagmm --epochs 10
```
Make sure that your dataset is saved with a correct name as a .npz file with one key as your dataset name
