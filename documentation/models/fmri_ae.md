---
layout: default
title: fMRI Autoencoder
parent: Models
grand_parent: Documentation
nav_order: 1
mathjax: true
tags: 
  - latex
  - math
---


## fMRI AE

This class implements an fMRI autoencoder.

> **Attributes:**
>
>	- *latent_shape*: tuple, specifying the latent dimension which is the output shape of the fMRI;
>	- *input_shape*: tuple, specifying the dimension of the fMRI input, typically it follows the (X-Referential, Y-Referential, Z-Referential) representation;
>	- *kernel_size*: tuple, specifying the size of the kernel size. This argument is deprecated and no longer influences the architecture, see NA specification;
>	- *stride_size*: tuple, specifying the size of the stride size. This argument is deprecated and no longer influences the architecture, see NA specification;
>	- *n_channels*: int, specifying the number of filters/channles to use in the convolutional layers for the fMRI encoder;
>	- *max_pool*: float, this is the value used for the weight decay for the L2 regularizers of each layer;
> 	- *batch_norm*: bool, specifies if convolutions are followed by batch normalization layers;
>	- *weight_decay*: float, this is the value used for the weight decay for the L2 regularizers of each layer;
> 	- *skip_connections*: bool, specifies if skip connections are used in the [Resnet-18 block layers](https://github.com/DCalhas/eeg_to_fmri/blob/feeef2cb2f0c1c38587df225962b72b4c67f8932/src/layers/resnet_block.py#L21);
>	- *dropout*: bool, specifies if a [Dropout](https://en.wikipedia.org/wiki/Dilution_(neural_networks)) of $$p=5$$ is used after each Resnet-18 block;
>	- *n_stacks*: int, specifies the number of Resnet blocks to use in the encoder. This argument is deprecated and no longer influences the architecture, see NA specification;
>	- *local*: bool, specifies if one uses either [Convolutional](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3D) (if True) layers or [Locally Connected](https://github.com/DCalhas/eeg_to_fmri/blob/feeef2cb2f0c1c38587df225962b72b4c67f8932/src/layers/locally_connected.py#L11) layers (if False). Please see this [paper](https://proceedings.neurips.cc/paper/2020/file/5c528e25e1fdeaf9d8160dc24dbf4d60-Paper.pdf) for more information on Locally Connected layers;
>	- *local_attention*: bool, if True uses a self attention mechanism in the latent representation. This argument is deprecated and was not validated in research;
>	- *outfilter*: int, $$\in \{0,1,2\}$$, specifies what type of output filter is used to correct the representation. Default is a 1x1 Conv3D layer;
>	- *seed*: int, specifies the seed from which the random generator starts;
>	- *_build_decoder*: bool, specifies if the decoder is built or not. This is because this class is used to build the fMRI encoder for the EEG_to_fMRI model;
>	- *na_spec*: tuple, Please refer to the next subsection for a clear description of a neural architecture specification;
