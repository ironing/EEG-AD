# Implementation

This is an official work in PyTorch reimplementation of **Task-oriented Self-supervised Learning for
Anomaly Detection in Electroencephalography**.

## Setup

Download the CHB-MIT Dataset from [here](https://physionet.org/content/chbmit/1.0.0/) and extract it into a new folder named `data`.

Install the following requirements:

1. Pytorch and torchvision
2. sklearn
3. pandas
4. seaborn
5. tensorboard

## Installation

```
git clone https://github.com/ironing/Task-oriented-SSL-EEG-AD.git
cd Task-oriented-SSL-EEG-AD
```

## Train

```
python pretreatment.py
```

The Script will process and split raw edf files.

```
python train.py --epochs 300 --learning_rate 0.0001 --inplane 18 --length 769
```

The Script will train a model save it in the AD_models Folder. The --inplane flag means the number of EEG channels and the --length flag means the length of EEG  segment.

## Anomaly Detection

```
python train.py --eval
```

This will run five random seeds and report mean AUC, F1-score and EER.
