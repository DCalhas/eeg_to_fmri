---
layout: default
title: Resnet Block
parent: Layers
grand_parent: Documentation
nav_order: 5
mathjax: true
tags: 
  - latex
  - math
---


# *ResBlock* Resnet Block ([He et al. 2015](https://arxiv.org/abs/1512.03385))

This layer specifically incorporates kernel and stride size information for each layer/block with the well known Resnet-18 Block.

> Arguments:
>	- *operation*: tf.keras.layer.Layer, can be either a convolution or a locally connected convolution;
>	- *kernel_size*: tuple, tuple of integers specifying the size of the sliding window. It has the same length of the length accepted by the *operation* layer;
>	- *stride_size*: tuple, tuple of integers specifying the stride jump of the kernel window;
>	- *n_channels*: int, spcifies the size of the channels/filters dimension;
>	- *max_pool*: bool, specifies if the *operation* layer is followed by a [Maxpooling](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool3D) operation;
>	- *batch_norm*: bool, specifies if the *operation* or *max_pool* layer is followed by a [BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) layer;
>	- *weight_decay*: float, specifies the decay used in a L2 regularization;
>	- *skip_connections*: bool, specifies if the block has a skip connections that resembles the residual connection;
>	- *max_pool_k*: tuple, tuple of integers specifying the size of the sliding window for the maxpool operation;
>	- *max_pool_s*: tuple, tuple of integers specifying the jump of the sliding window for the maxpool operation;
>	- *seed*: int, specifies the intiialization seed for a random generator;
>	- *kwargs*: dict, additional arguments;
>
> Methods:
>	- *\_\_init\_\_*: initializes the class;
>	- *set_layers*: resembles the build method of a tf.keras.layers.Layer;
>	- *call*: returns the output of a forward pass of a complete resnet-18 block;
>	- *lrp*: propagates relevances from the output to the input of this layer block;
>	- *get_config*: returns a dictionary with the configuration needed to serialize the layer;
>	- *from_config*: returns a *ResBlock* instanced class with the configuration received;