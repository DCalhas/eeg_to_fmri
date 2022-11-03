---
layout: default
title: Layers
parent: Documentation
nav_order: 1
mathjax: true
tags: 
  - latex
  - math
---

# Layers

In this section we will go over the layers that were implemented in this repository.

## Bayesian Module

This model implements variational layers:

- Dense Variational layer

### Dense Varitational

This layer is the bayesian version of the [tf.keras.layers.Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) layer. It is parametrized by $$W,b$$ and transforms the inputs, $$X\in \mathbb{R}^{N}$$, as:

$$a(W^\top \cdot X + b)$$

The parameters are variational and are setup as:

$$W = W_{\mbox{posterior}} + W_{\mbox{prior}} \times \epsilon: epsilon \sim \mathcal{N}(0,1)^{N \times \mbox{units}} \wedge W_{\mbox{posterior}}, W_{\mbox{prior}} \in \mathbb{R}^{N \times \mbox{units}},$$

and for the bias:

$$b = b_{\mbox{posterior}} + b_{\mbox{prior}} \times \epsilon: epsilon \sim \mathcal{N}(0,1)^{\mbox{units}} \wedge b_{\mbox{posterior}}, b_{\mbox{prior}} \in \mathbb{R}^{\mbox{units}},$$

> Arguments:
>	- *units*: int, specifies the output units;
>	- *activation*: func, specifies the activation $$a$$ used;
>	- *activity_regularizer*: func, specifies the activity regularization applied;
>	- *kernel_prior_initializer*: tf.keras.initializers.Initializer, specifies the initializations for the kernel prior, $$W_{\mbox{prior}}$$;
>	- *kernel_posterior_initializer*: tf.keras.initializers.Initializer, specifies the initializations for the kernel posterior, $$W_{\mbox{posterior}}$$;
>	- *bias_prior_initializer*: tf.keras.initializers.Initializer, specifies the initializations for the bias prior, $$b_{\mbox{prior}}$$;
>	- *bias_posterior_initializer*: tf.keras.initializers.Initializer, specifies the initializations for the bias posterior, $$b_{\mbox{posterior}}$$;
>	- *use_bias*: bool, specifies whether to use bias or not;
>	- *trainable*: bool, specifies whether the parameters are trainable;
>	- *seed*: bool, specifies the seed to generate random numbers;
>
> Methods:
>	- *call*: returns the output given an input;
>	- *get_config*: returns a dictionary with the configuration needed to serialize the layer (see [layer serialization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/serialize));
>	- *from_config*: returns a *DenseVariational* instanced class with the configuration received;

## FFT Module

This module implements various [Discrete Cosine Transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform) layers in tensorflow, namely:

- Discrete Cosine Transform 3-Dimensional
- Inverse Discrete Cosine Transform 3-Dimensional
- Padded Inverse Discrete Cosine Transform 3-Dimensional
- Variational Inverse Discrete Cosine Transform 3-Dimensional

### *DCT3D* Discrete Cosine Transform 3-Dimensional

Implements the discrete cosine transform according to $$X \in \mathbb{R}^N: X_k = \mathcal{F}(x)_k = \sum_{n=0}^{N-1} x_n cos\left[ \frac{\pi (2n+1)k}{2N} \right], \forall k \in \{0, \dots, N-1 \}$$.

> Arguments:
>	- *N1*: int, specifying the first dimension;
>	- *N2*: int, specifying the second dimension;
>	- *N3*: int, specifying the third dimension;
>
> Methods:
>	- *call*: returns the spectral representation of the input;
>	- *get_config*: returns a dictionary with the configuration needed to serialize the layer (see [layer serialization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/serialize));
>	- *from_config*: returns a *DCT3D* instanced class with the configuration received;


### *iDCT3D* Inverse Discrete Cosine Transform 3-Dimensional

Implements the discrete cosine transform according to $$x_k = \mathcal{F}^{-1}(X)_k = X_0 + 2\sum_{n=0}^{N-1} X_n cos\left[ \frac{\pi n(2k+1)}{2N} \right], \forall k \in \{0, \dots, N-1 \}$$.

> Arguments:
>	- *N1*: int, specifying the first dimension;
>	- *N2*: int, specifying the second dimension;
>	- *N3*: int, specifying the third dimension;
>
> Methods:
>	- *call*: returns the spatial representation of the input;
>	- *get_config*: returns a dictionary with the configuration needed to serialize the layer;
>	- *from_config*: returns a *iDCT3D* instanced class with the configuration received;


### *padded_iDCT3D* Padded Inverse Discrete Cosine Transform 3-Dimensional

Performs the inverse DCT, but adds zeroed frequencies, i.e. padds the spectral representation on the right with zeros and calls the *iDCT3D* layer.

> Arguments:
>	- *in1*: int, specifying the inputs' first dimension;
>	- *in2*: int, specifying the inputs' second dimension;
>	- *in3*: int, specifying the inputs' third dimension;
>	- *out1*: int, specifying the outputs' first dimension;
>	- *out2*: int, specifying the outputs' second dimension;
>	- *out3*: int, specifying the outputs' third dimension;
>
> Methods:
>	- *call*: returns the spatial representation of the input, with a higher resolution;
>	- *get_config*: returns a dictionary with the configuration needed to serialize the layer;
>	- *from_config*: returns a *padded_iDCT3D* instanced class with the configuration received;


### *variational_iDCT3D* Variational Inverse Discrete Cosine Transform 3-Dimensional

Performs the inverse DCT, but adds stochastic frequencies, i.e. padds the spectral representation on the right with random variables and calls the *padded_iDCT3D* layer if there still needs more coefficients to be added.

> Arguments:
>	- *in1*: int, specifying the inputs' first dimension;
>	- *in2*: int, specifying the inputs' second dimension;
>	- *in3*: int, specifying the inputs' third dimension;
>	- *out1*: int, specifying the outputs' first dimension;
>	- *out2*: int, specifying the outputs' second dimension;
>	- *out3*: int, specifying the outputs' third dimension;
>	- *rand1*: int, specifying the number of random variables to add to the first dimension;
>	- *rand2*: int, specifying the number of random variables to add to the second dimension;
>	- *rand3*: int, specifying the number of random variables to add to the third dimension;
>	- *coefs_perturb*: bool, if True perturbs the coefficients $$\in \mathbb{R}^{in1 \times in2 \times in3}$$ with guassian random variables parametrized by $$\mu, \sigma$$. These parameters are set as trainable;
>	- *dependent*: bool, if True builds the higher stochastic coefficients from the input resolution, with an attention mechanism, i.e. a sum of sinusoids;
>	- *posterior_dimension*: int, specifies the dimension of the sinusoids needed to estimate the high resolution coefficients;
>	- *distribution*: str, specifies the distribution used for the random variables. Currently, only the von Mises distribution is supported;
>
> Methods:
>	- *call*: returns the spatial representation of the input, with a higher resolution with stochastic spectral coefficients (see this [paper](https://en.wikipedia.org/wiki/HTTP_404));
>	- *get_config*: returns a dictionary with the configuration needed to serialize the layer;
>	- *from_config*: returns a *variational_iDCT3D* instanced class with the configuration received;

## Random Fourier Module

This module is a wrap of the implementation done in [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/RandomFourierFeatures), but some tweaks were needed to get it to work with our methodology. The file is similar to the tensorflow implementation.

## Topographical Attention Module

This module implements the topographical attention presented in this [paper](https://arxiv.org/abs/2203.03481).

### *Topographical_Attention* Topographical Attention

This layer has a weight matrix, $$A \in \mathbb{R}^{C \times F}$$, where $$C$$ and $$F$$ correspond to the EEG electrodes dimension and number of features (flattened representation), respectively. Performs attention on an input vector representation, $$X \in \mathbb{R}^{B \times C \times F}$$, where $$B$$ refers to the batch dimension. This is done by:

$$
W = X^\top \cdot A
$$

$$W$$ represents the attention weights, that are then normalized according to $$E = \frac{\mbox{exp}(W)}{\sum_j \mbox{exp}(W_j)}$$, the attention scores $$\in \mathbb{R}^{C \times C}$$, which are used to reorganize the channels as $$T_i = X \cdot E, \forall i \in \{1, \dots, C\}$$.

> Arguments:
>	- *channels*: int, specifying the number of electrodes of the EEG representation;
>	- *features*: int, specifying the inputs' second dimension;
>	- *regularizer*: [tf.keras.regularizers.Regularizer](https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/Regularizer), specifying the inputs' third dimension;
>	- *seed*: int, specifying the outputs' first dimension;
>	- *kwargs*: dict, additional arguments;
>
> Methods:
>	- *build*: initializes the weights of the layer according to a given *input_shape*;
>	- *call*: returns the reorganized representation of the input, using an attention mechanism;
>	- *lrp*: propagates relevances from the output of the layer to its input $$X$$;
>	- *lrp_attention*: propagates relevances from the output of the layer to the attention scores, $$E$$;
>	- *get_config*: returns a dictionary with the configuration needed to serialize the layer;
>	- *from_config*: returns a *Topographical_Attention* instanced class with the configuration received;


## Mask Module

In this module one can find implementations of layers that performs segmentation over a 3-dimensional MRI volume representation.

### *MRICircleMask* MRI Volume Circle Mask

This is a naive implementation of a 3D circle brain mask that is fixed for all instances.

> Arguments:
>	- *input_shape*: tuple, giving the dimensions of the fMRI volume, can be of length 5 (meaning the batch dimension is included along with the channels dimension, the channels dimension here refers to the the convolutional filters dimension) or of length 4 (meaning the batch dimension is not included);
>	- *radius*: float, specifying the size of the circle brain mask;
>
> Methods:
>	- *call*: returns the input representation filtered with the circle 3D mask designed in this layer;

## Resnet Block Module

This module allows an easy integration of neural architecture specifications that were generated automatically, according to the methodology described in this [paper](https://openreview.net/forum?id=TCvkaP15O7e).


### *ResBlock* Resnet Block ([He et al. 2015](https://arxiv.org/abs/1512.03385))

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
