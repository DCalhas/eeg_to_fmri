---
layout: default
title: EEG recording to fMRI volume
parent: Blog
nav_order: 1
mathjax: true
tags: 
  - latex
  - math
---


# EEG recoding to fMRI volume

In this post I will go over the procedure to project an EEG instance to an fMRI.

In the [EEG to fMRI paper](https://arxiv.org/abs/2203.03481), the methodology to do this projection is presented. The architecture of this model is shown in the Figure below.

<p align="center">
	<img src="./figures/architecture_eeg_benefits.png" width="200"/>
</p>

This model processes two inputs:
- EEG representation $$\vec{x} \in \mathbb{R}^{C \times F \times T}$$;
- fMRI volume representation $$\vec{y} \in \mathbb{R}^{M_1 \times M_2 \times M_3}$$.

The EEG has a drift in relation to the associated fMRI volume, since it takes into consideration $$10\times \mbox{TR}$$ seconds in total (for the [NODDI](https://osf.io/94c5t/) dataset this corresponds to $$10\times 2.160=21.6$$ seconds). In addition, the EEG has $$C$$ channels and $$F$$ frequency coefficients. The figure below shows a representation of an EEG.

<p align="center">
	<img src="./figures/eeg_stft.png" width="200"/>
</p>

On the other hand, we have a **single** fMRI volume associated with the respective EEG. This fMRI is described by three dimensions, corresponding to the 3-dimensional axis, where $$M_1=64$$, $$M_2=64$$ and $$M_3=30$$. The figure below shows an fMRI volume.

<p align="center">
	<img src="./figures/fmri_volume.png" width="600"/>
</p>

In terms of code this corresponds to loading the data. First we start by import the necessary libraries:

```
import tensorflow as tf

import numpy as np

from utils import tf_config

dataset="01"
memory_limit=1500

tf_config.set_seed(seed=2)
tf_config.setup_tensorflow(device="GPU", memory_limit=memory_limit)

import GPyOpt

import argparse

from utils import preprocess_data, train, losses_utils, metrics, eeg_utils, data_utils

from models import eeg_to_fmri

from pathlib import Path
```

Then we specify the number of individuals and $$T$$ (which corresponds to ```interval_eeg```), and load the data:

```
n_individuals=getattr(data_utils, "n_individuals_"+dataset)
interval_eeg=10

with tf.device('/CPU:0'):
	train_data, test_data = preprocess_data.dataset(dataset, n_individuals=n_individuals,
											interval_eeg=interval_eeg, 
											ind_volume_fit=False,
											standardize_fmri=True,
											iqr=False,
											verbose=True)
	eeg_train, fmri_train =train_data
	eeg_test, fmri_test = test_data

```