---
layout: default
title: Classification on EEG only datasets
parent: Blog
nav_order: 2
mathjax: true
tags: 
  - latex
  - math
---


# Classification on EEG only datasets

In this post I will talk about the application of this package to classify EEG data. The purpose is to be able to classify an EEG alone, without an fMRI pairing, while being able to project a synthesized fMRI view.

## EEG only dataset

The dataset consists of $$N$$ individuals. The individuals have a set of features $$X \in \mathbb{R}^{N \times C\times F\times T}$$ and labels $$y \in \mathbb{R}^{N \times 1}$$. The notation is similar to the previous blog post, where $$C$$ stands for the number of channels, $$F$$ is the frequency resolution and $$T$$ is the amount of time considered of the EEG recording. Note that, we only consider binary classification problems for this setup. An EEG instance looks like illustrated in the figure below.

<p align="center">
	<img src="./figures/eeg_stft.png" width="200"/>
</p>

This image corresponds to the STFT projection of one channel. The labels are $$y = \{0, 0, \dots, 1, 1\}$$.

## Classifying synthesized images

In this section, I will go over the methodology used to classify synthesized images (a projection that is done using the sinusoids). This is the novel contribution of this [paper](https://en.wikipedia.org/wiki/HTTP_404). It consists on picking the sinusoid projections $$cos(\omega \cdot \vec{z}_x + \beta)$$, with $$\omega,\beta$$ being trainable parameters, and separating both $$\omega \cdot \vec{z}_x + \beta$$ and $$cos(\omega \cdot \vec{z}_x + \beta)$$ using a contrastive loss. In the figure below it is shown how the points are initialized and how after training they are well separated in opposite sides of the unit circle, where the $$cos$$ takes values $$\approx 1$$ and $$\approx -1$$.

<p align="center">
	<img src="./figures/contrastive_optimization.png" width="400"/>
</p>

The loss that achieves this goal is an adaptation of the [contrastive loss](),

$$\mathcal{L}_D(x_1, x_2, y_p) = y_p \times D(x_1, x_2) + (1-y_p) \times \|D(x_1,x_2)-m\|_1,$$

where $$x_1$$ and $$x_2$$ are supposed to be the projection before the cosine, $$\omega \cdot \vec{z}_x + \beta$$, for two instances, and $$y_p$$ specifies if these two instances belong together or not. Two instances belong together if $$y_i == y_j$$ and are false pairs if $$y_i \neq y_j$$. Therefore, $$y_p$$ can be represented as $$1[y_i == y_j]$$. The term $$y_p \times D(x_1, x_2)$$ brings points with the same label closer together, while them term $$(1-y_p) \times \|D(x_1,x_2)-m\|_1$$ places points with different labels as far as $$m$$, which we set to $$m=\pi$$.

## Let us dive into the code!

First you need to do all the imports:

```python
import tensorflow as tf
from utils import data_utils, preprocess_data, tf_config, train, losses_utils, lrp, viz_utils, fmri_utils
from layers import topographical_attention
from models import classifiers, eeg_to_fmri
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
```

Then you set up the information about which dataset to use for the classification, for the synthesis, as well as setting up the seed for tensorflow and the memory limit it uses on the GPU.

```python
dataset="11"
dataset_synthesis="01"
raw_eeg=False
n_individuals=getattr(data_utils, "n_individuals_"+dataset)
memory_limit=1500
interval_eeg=10
tf_config.set_seed(seed=2)
tf_config.setup_tensorflow(device="GPU", memory_limit=memory_limit, run_eagerly=True)
```







## References

\[1\]: [Tancik, Matthew, et al. Fourier features let networks learn high frequency functions in low dimensional domains. Advances in Neural Information Processing Systems, 2020, 33: 7537-7547.](https://arxiv.org/abs/2006.10739)