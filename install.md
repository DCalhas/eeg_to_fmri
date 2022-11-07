---
layout: default
title: Install EEG to fMRI
nav_order: 2
permalink: /
---

# Setup

Ideally, your machine has a GPU and is running Linux.

First of all, please install [anaconda](https://www.anaconda.com/) at ```$HOME/anaconda3/```. To setup the environment for this repository, please run the following commands:

```shell
git clone git@github.com:DCalhas/eeg_to_fmri.git
cd eeg_to_fmri
```

Download [cudnn](https://developer.nvidia.com/cudnn):

```shell
wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.0.1/cudnn-11.0-linux-x64-v8.0.1.13.tgz
```

Run the configuration file:

```shell
./config.sh
```

Please make sure to set the path to the datasets directory correclty. This path is stored in an environment variable, so every time you activate the environment, the variable is set and used in the code as ```os.environ['EEG_FMRI_DATASETS']```.
