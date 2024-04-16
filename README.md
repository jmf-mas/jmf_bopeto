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
```
Replace `[ENV_NAME]` with the name of your environment.

## Usage
From the root of the project.
```
$ conda activate bopeto
$ python main.py --path data/kdd.npz --synthetic JMF --metric rmac --contamination_rate 0 0.1 0.2 0.3
```

Our model contains the following parameters:
- `-m`: selected machine learning model (**required**)
- `-d`: path to the dataset (**required**)
- `--batch-size`: size of a training batch (**required**)
- `--dataset`: name of the selected dataset. Choices are `Arrhythmia`, `KDD10`, `IDS2018`, `NSLKDD`, `USBIDS`, `Thyroid` (**required**).
- `-e`: number of training epochs (default=200)
- `--n-runs`: number of time the experiment is repeated (default=1)
- `--lr`: learning rate used during optimization (default=1e-4)
- `--pct`: percentage of the original data to keep (useful for large datasets, default=1.)
- `rho`: anomaly ratio within the training set (default=0.)
- `--results-path`: path where the results are stored (default="../results")
- `--model-path`: path where models will be stored (default="../models")
- `--test-mode`: loads models from `--model_path` and tests them (default=False)
Please note that datasets must be stored in `.npz` or `.mat` files. Use the preprocessing scripts within `data_process`
to generate these files.

## Example
To train a DAGMM on the KDD 10 percent dataset with the default parameters described in the original paper:
```
$ python main.py -m=AE --dataset=KDD10 --dataset-path=../toy_data/kdd.npz --n-runs=15 --batch-size=4096 --batch-size-test=12288 --pct=.8 --rho=0.12 --hold_out=0.3 --test_pct=.50 --eval-test -e=20 --lr=1e-3 --reg_n=1e-0 --reg_a=1e-2 --num_clusters=5 --warmup=0 -lat=6
```
Replace `[/path/to/dataset.npz]` with the path to the dataset in a numpy-friendly format.

Optionally, a Jupyter notebook is made available in `experiments.ipynb`
