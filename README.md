# Bopeto: Data cleaning for unsupervised anomaly detection
This repository implements Bopeto method and collects different unsupervised machine learning algorithms for anomaly detection.
## Implemented models
For unsupervised anomaly detection we have used the following models. The original implementations already available on GitHub.
- [x] [AE](https://github.com/intrudetection/robevalanodetect)
- [x] [ALAD](https://github.com/houssamzenati/Adversarially-Learned-Anomaly-Detection)
- [x] [DAGMM](https://github.com/intrudetection/robevalanodetect)
- [x] [DeepSVDD](https://github.com/lukasruff/Deep-SVDD)
- [x] [DSEBM](https://github.com/intrudetection/robevalanodetect)
- [x] [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [x] [LocalOutlierFactor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
- [x] [One-class SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)

## Dependencies
A complete dependency list is available in requirements.txt.
We list here the most important ones:
- torch@1.10.2 with CUDA 11.3
- numpy
- pandas
- scikit-learn
- seaborn

## Installation
```
$ conda create --name [ENV_NAME] python=3.8
$ conda activate [ENV_NAME]
$ pip install -r requirements.txt
```
Replace `[ENV_NAME]` with the name of your environment.

Our model contains the following parameters:
- `--batch_size`: size of a training batch (default=64)
- `--dataset`: name of the selected dataset. Choices are `ciciot`, `credit`, `ecg`, `ids`, `kdd`, `kitsune` (default=kdd). Please note that datasets must be stored in `.npz`. 
- `--epochs`: number of training epochs (default=100)
- `--learning_rate`: learning rate (default=0.001)
- `--weight_decay`: weight decay (default=0.001)
- `--mode`: running modes: cleaning, bopeto or iad (default=bopeto)
- `--cleaning`: type of cleaning (hard or soft) (default=hard)
- `--model`: anomaly detection model (svdd, ae, if, lof, alad, dsebm, dagmm) (default=svdd)
- `--num_contamination_subsets`: number of contamination ratios to generate (default=10)


## Example
Cleaning KDD dataset using Bopeto:
```
$ python ad.py --dataset kdd  --mode cleaning --num_contamination_subsets 3 
```
Detecting anomalous instances using SVDD on the KDD dataset:
```
$ python ad.py --dataset kdd --model svdd --mode bopeto
```
<!--
You can automate the whole process (data cleaning and anomaly detection) using the following
```
$ chmod +x ad.sh
$ ./ad.sh
```
-->
Make sure that your dataset is saved with a correct name as a .npz file with one key as your dataset name
