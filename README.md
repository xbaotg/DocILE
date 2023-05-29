# DocILE - Document Information Localization and Extraction

---
This is the implementation of our team (UIT@AICLUB_TAB) on the KILE subtask. In this competition, we got the 3rd Prize.  

### Introduction
DocILE is a large-scale research benchmark for cross-evaluation of machine learning methods for Key Information Localization and Extraction (KILE) and Line Item Recognition (LIR) from semi-structured business documents such as invoices, orders, etc. Such large-scale benchmark was previously missing [(Skalický et al., 2022)](https://link.springer.com/chapter/10.1007/978-3-031-13643-6_8), hindering comparative evaluation.

### Folder Structure

```
├── experiments             # contains checkpoint when training
├── predictions             # contains predictions when inference
├── run_inference.sh
├── run_training.sh
├── train.py
├── inference.py
├── config.py               # make config here
└── utils                   # source code to create pseudo data or visualize data
```

### Dependencies

Make sure that you are using Python 3.8+ to run scripts in this. Run this command to install requirements:

```
apt install poppler-utils
pip install -r requirements.txt
```

### Config

You can view and edit the config [here](config.py). In this file, you can config whether the model will use Post-Processing, Fast Gradient Method, and how to Ensemble outputs, you can also customize any Optimizer and Scheduler you want to use to train the model.   

### Train

Before training, you should config hyperparameters in file `run_training.sh`. Don't forget to change the output directory, data path, checkpoint path, and GPU devices, too. 

```
./run_training.sh
```

To resume from the specific checkpoint, you need to change this  `TIMESTAMP=$(date +"%Y%m%d_%H%M_%S")` into the folder name containing checkpoint and add parameter `--resume` into `train_params`.

### Inference

Before inference, you also need to change the checkpoint path, data path, ... in file `run_inference.sh`. After that, run this command to inference model on validation data (or any data):

```
./run_inference.sh val
```

### Create Pseudo data for pretraining

You can view [this tutorial](utils/unlabeled/README.md).

### Links 

[https://github.com/rossumai/docile](https://github.com/rossumai/docile)

[https://docile.rossum.ai/](https://docile.rossum.ai/)