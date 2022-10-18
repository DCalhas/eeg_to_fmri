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

In the [EEG to fMRI paper](https://arxiv.org/abs/2203.03481), the methodology to do this projection is presented. The architecture of this model is shown in the figure below.

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

```python
import tensorflow as tf
import numpy as np
from utils import tf_config

dataset="01"
memory_limit=1500
tf_config.set_seed(seed=2)
tf_config.setup_tensorflow(device="GPU", memory_limit=memory_limit)

import GPyOpt
import argparse
from utils import preprocess_data, train, losses_utils, metrics, eeg_utils, data_utils, viz_utils
from models import eeg_to_fmri
from pathlib import Path
```

Then we specify the number of individuals and $$T$$ (which corresponds to ```interval_eeg```), and load the data:

```python
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
print(eeg_train.shape, fmri_train.shape)
print(eeg_test.shape, fmri_test.shape)
```
The print output corresponds to:
```
<<< (64,134,10,1) (64,64,30,1)
<<< (64,134,10,1) (64,64,30,1)
```

After this we can start building the model presented in the first figure. To do this, we first unroll the hyperparameters for the neural network: learning rate, weight decay, etc:

```python
learning_rate,weight_decay ,kernel_size ,stride_size ,batch_size,latent_dimension,n_channels,max_pool,batch_norm,skip_connections,dropout,n_stacks,outfilter,local=eeg_to_fmri.parameters
```

Additionally, we also load the architecture specification:

```python
with open(str(Path.home())+"/eeg_to_fmri/na_models_eeg/na_specification_2", "rb") as f:
	na_specification_eeg = pickle.load(f)
with open(str(Path.home())+"/eeg_to_fmri/na_models_fmri/na_specification_2", "rb") as f:
	na_specification_fmri = pickle.load(f)
```

With these parameters, we can finally build the model and setup the optimizer, loss, training set and test set:

```python
with tf.device('/CPU:0'):
	model = eeg_to_fmri.EEG_to_fMRI(latent_dimension, eeg_train.shape[1:], na_specification_eeg, n_channels,
						weight_decay=weight_decay, skip_connections=True, batch_norm=True, fourier_features=True,
						random_fourier=True, topographical_attention=True, conditional_attention_style=True,
						conditional_attention_style_prior=False, local=True, seed=None, 
						fmri_args = (latent_dimension, fmri_train.shape[1:], kernel_size, stride_size, n_channels, 
						max_pool, batch_norm, weight_decay, skip_connections,
						n_stacks, True, False, outfilter, dropout, None, False, na_specification_fmri))
	model.build(eeg_train.shape, fmri_train.shape)
	optimizer = tf.keras.optimizers.Adam(learning_rate)
	loss_fn = losses_utils.mae_cosine
	train_set = tf.data.Dataset.from_tensor_slices((eeg_train, fmri_train)).batch(batch_size)
	test_set= tf.data.Dataset.from_tensor_slices((eeg_test, fmri_test)).batch(1)
```

Note that the loss used minimizes the objective of approximating the EEG with the fMRI, along with the latent representations of each other, meaning:

$$\mathcal{L}(\vec{x}, \vec{y}) = ||\vec{y}-\hat{y}||_1^1  + 1-\frac{\vec{z}_x^* \cdot \vec{z}_y}{||\vec{z}_x^*||_2^2\cdot ||\vec{z}_y||_2^2}$$

This loss approximates the output (predicted fMRI) with the ground truth (fMRI volume) with the L1 distance and approximates the latent representations with a pattern based (cosine) distance.

Finally, comes the fun part, where the model is trained with the objective of producing an fMRI volume similar to the one paired with the given EEG:

```python
train.train(train_set, model, optimizer, loss_fn, epochs=10, u_architecture=True, val_set=None, verbose=True, verbose_batch=True)
```

The output in my machine corresponds to:

```
<<< Epoch 1: 
```

Now we have a trained model that given an EEG representation, gives us an fMRI volume. You can check the visualization by using the visualization utilities file available in this repository:

```
for eeg, fmri in dev_set.repeat(1):
	viz_utils.plot_3D_representation_projected_slices(model(eeg, fmri)[0].numpy()[0], threshold=0.37).show()
	break
```

The output of this code corresponds to the figure below. Note that we give the model the EEG and the fMRI representation, however our goal is to produce an fMRI volume without an EEG reference. If we check the [call function](https://github.com/DCalhas/eeg_to_fmri/blob/ff7c1b988a7dca77f0db400bcb511c6127e82c33/src/models/eeg_to_fmri.py#L329) of the model, we see that it returns a list of tensors, where the first tensor is the predicted fMRI without influence of the original fMRI and only dependent on the EEG.