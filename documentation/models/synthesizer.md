---
layout: default
title: EEG to fMRI Synthesizer
parent: Models
grand_parent: Documentation
nav_order: 2
mathjax: true
tags: 
  - latex
  - math
---


# EEG to fMRI

This class implements the EEG to fMRI Synthesis model proposed in this [paper](https://arxiv.org/abs/2203.03481). 

> **Attributes:**
>
> 	- *latent_shape*: tuple, specifying the latent dimension which is the output shape of the EEG encoder, as well as the fMRI encoder;
>	- *input_shape*: tuple, specifying the dimension of the EEG input, typically it follows the (Channels, Frequency, Time) representation;
>	- *na_spec*: tuple, Please refer to the next subsection for a clear description of a neural architecture specification;
>	- *n_channels*: int, specifying the number of filters/channles to use in the convolutional layers for the EEG encoder;
>	- *weight_decay*: float, this is the value used for the weight decay for the L2 regularizers of each layer;
> 	- *skip_connections*: bool, specifies if skip connections are used in the [Resnet-18 block layers](https://github.com/DCalhas/eeg_to_fmri/blob/feeef2cb2f0c1c38587df225962b72b4c67f8932/src/layers/resnet_block.py#L21);
> 	- *batch_norm*: bool, specifies if convolutions are followed by batch normalization layers;
>	- *dropout*: bool, specifies if a [Dropout](https://en.wikipedia.org/wiki/Dilution_(neural_networks)) of $$p=5$$ is used after each Resnet-18 block;
>	- *local*: bool, specifies if one uses either [Convolutional](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3D) (if True) layers or [Locally Connected](https://github.com/DCalhas/eeg_to_fmri/blob/feeef2cb2f0c1c38587df225962b72b4c67f8932/src/layers/locally_connected.py#L11) layers (if False). Please see this [paper](https://proceedings.neurips.cc/paper/2020/file/5c528e25e1fdeaf9d8160dc24dbf4d60-Paper.pdf) for more information on Locally Connected layers;
>	- *fourier_features*: bool, specifies the use of fourier features on the EEG latent representation, $$z_x$$;
>	- *random_fourier*: bool, specifies the use of [Random Fourier features](https://github.com/DCalhas/eeg_to_fmri/blob/feeef2cb2f0c1c38587df225962b72b4c67f8932/src/layers/fourier_features.py#L35). It requires *fourier_features* to be True;
> 	- *conditional_attention_style*: bool, specifies the use of the attention scores in the latent representation of the EEG, by usage of the [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)), simplifies in tensorflow to ```tf.Tensor * tf.Tensor```;
> 	- *conditional_attention_style_prior*: bool, specifies if the vector used to perform the product with the $$z_x$$, is either $$W$$ (if False), i.e. the attention scores, or an $$w$$ that is learnable (if True);
>	- *inverse_DFT*: bool, specifies if in the Decoder part, a DCT upsampling mechanism is used. In particular, this flag enables the use of the [iDCT3D](https://github.com/DCalhas/eeg_to_fmri/blob/feeef2cb2f0c1c38587df225962b72b4c67f8932/src/layers/fft.py#L263) layer;
>	- *DFT*: bool, similarly, this flag enables the use of the [DCT3D](https://github.com/DCalhas/eeg_to_fmri/blob/feeef2cb2f0c1c38587df225962b72b4c67f8932/src/layers/fft.py#L27) layer;
> 	- *variational_iDFT*: bool, similarly, this flag enables the use of the [variational_iDCT3D](https://github.com/DCalhas/eeg_to_fmri/blob/feeef2cb2f0c1c38587df225962b72b4c67f8932/src/layers/fft.py#L323) layer;
>	- *aleatoric_uncertainty*: bool, specifies the use of [tfp.layers.DenseFlipout](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseFlipout) as a Decoder (if DFT is False) and/or the an additional output to represent $$\hat{\sigma}(x_i)$$, by doing ```tf.keras.layers.Dense(1, activation=tf.keras.activations.exponential)```;
>	- *variational_coefs*: tuple, specifies the number of additional coefficients to add to the resolution, should be a tuple (R_1, R_2, R_3) with \forall i \in \{1, 2, 3\}: R_i < M_i-N_i, where M is the upsampled resolution and N the downsampled;
>	- *variational_dist*: str, specifies the distribution from which to sample the coefficients. Currently only the Von Mises is supported;
>	- *variational_iDFT_dependent*: bool, specifies if the high sampled resolutions have a dependency, implemented from an attention mechanism, from the lower resolutions;
>	- *variational_iDFT_dependent_dim*: int, specifies the dimension of the attention mechanism. This simplifies to a sum of sinusoids, i.e. a sum of the number of sinusoids specified by this argument;
>	- *resolution_decoder*: float, specifies the downsampled resolution;
>	- *low_resolution_decoder*: bool, flag to set the low resolution decoder mechanism to true;
>	- *topographical_attention*: bool, specifies if the topographical attention mechanism is used;
>	- *seed*: int, specifies the seed from which the random generator starts;
>	- *fmri_args*: tuple, contains the arguments given to the fMRI encoder part of the model. Please refer to the fmri_ae documentation;	

## Neural Architecture Specification

NA_specification - tuple - (list1, list2, bool, tuple1, tuple2)
                                    * list1 - kernel sizes
                                    * list2 - stride sizes
                                    * bool - maxpool
                                    * tuple1 - kernel size of maxpool
                                    * tuple2 - stride size of maxpool

Example:

```python
na = ([(2,2,2), (2,2,2)], [(1,1,1), (1,1,1)], True, (2,2,2), (1,1,1))
```

This ```na``` is a neural architecture with 2 layers, kernel of size 2 for all 3 dimensions stride of size 1 for all dimensions, between each layer a max pooling operation is applied with kernel size 2 for all dimensions and stride size 1 for all dimensions


## Example build

```
from models.eeg_to_fmri import EEG_to_fMRI

model = EEG_to_fMRI((7,7,7),
					EEG_SHAPE,**(None,))

```