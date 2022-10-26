---
layout: default
title: Documentation
nav_order: 2
mathjax: true
tags: 
  - latex
  - math
---

# Documentation

The code of this PhD thesis is extensive. Please read this section carefully if you have any issue with the code before you publish any issue or contact the authors. If any issue still persists after reading the documentation, please publish an issue on the Github repository.

## Main.py

The main.py file located at src/main.py is the script used for retrieve the results present in the [EEG to fMRI Synthesis](https://arxiv.org/abs/2203.03481) paper.

The arguments given to the file of the form:
```bash
python main.py <ARG1> <ARG2> ... <ARGN>
```
are as follows:

**Required:**
- *mode*
	- metrics: the model is trained on a selected dataset and metrics, such as RMSE, SSIM and [Sharpness](https://arxiv.org/abs/1609.04836) (for uncertainty) are computed;
	- residues: plots of the residues, on the test set data, in a white to black (bad to good, respectively) scale are retrieved. Example: TODO
	- quality: retrieves the plots of each synthesized fMRI view.
	- lrp_eeg_channels: Propagate the relevances from $$\hat{y}$$ to the channels [graph representation](https://github.com/DCalhas/eeg_to_fmri/blob/master/src/layers/topographical_attention.py). This is done using the [Layer-wise Relevance Propagation](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140). 
	- lrp_eeg_fmri: Propagate the relevances from the latent representation $$z_y$$ to $$y$$.
- *dataset*
	- 01: This corresponds to the [NODDI dataset](https://osf.io/94c5t/).
	- 02: This corresponds to the [Oddball dataset](https://legacy.openfmri.org/dataset/ds000116/).
	- 03: This corresponds to the [CN-EPFL dataset](https://openneuro.org/datasets/ds002158/versions/1.0.0).

>	**Optional:**
>	- *topographical_attention*: whether to use [topographical attention](https://github.com/DCalhas/eeg_to_fmri/blob/master/src/layers/topographical_attention.py) on the EEG channels/electrodes dimension.
>	- *conditional_attention_style*: use the attention scores to condition the latent representations. This is only used if *topographical_attention* is also used.
>	- *variational*: toggle variational (to compute uncertainty) of the model. The implementation done so far uses [sinusoidals](https://github.com/DCalhas/eeg_to_fmri/blob/2f4138c34549c4ebfe56869bd2877f922f57b8f6/src/layers/fft.py#L323) to integrate uncertainty.
>	- *variational_coefs*: the number of coefficients to insert as extra stochastic coefficients. Example: 7,7,7.
>	- *variational_dependent_h*: the dimension of the number of sinusoidals to use to estimate the high DCT spectral coefficients. This can also be interpreted as an attention mechanism. Both interpretation of sum of sinusoidals and attention are correct.
> 	- *variational_dist*: how the random variables of the high coefficients are distributed. Currently supported distributions are [von Mises](https://en.wikipedia.org/wiki/Von_Mises_distribution), a.k.a. spherical Gaussian distribution.
>	- *resolution_decoder*: A float $$\in ]1, \infty[$$ defining the resolution from which the higher resolution coefficients are built.
> 	- *aleatoric_uncertainty*: A flag that specifies whether the [EEG to fMRI model](https://github.com/DCalhas/eeg_to_fmri/blob/master/src/models/eeg_to_fmri.py) has an aleatoric uncertainty output. If this is true and the *variational* flag is False, the model uses an affine decoder using the [flipout trick](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseFlipout) as a variational decoder.
>	- *fourier_features*: Whether to use or not [fourier features](https://arxiv.org/abs/2006.10739). Do not mistake this as sinusoidal transforms.
>	- *random_fourier*: A specific case of fourier features, [random projections of sinusoidals](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/RandomFourierFeatures). Check this [paper](https://people.eecs.berkeley.edu/%7Ebrecht/papers/07.rah.rec.nips.pdf).
>	- *epochs*: An integer specifying the number of epochs to train the model.
>	- *batch_size*: An integer specifying the batch size used to train the model. DEPRECATION: this argument may be ill defined and not used!
>	- *na_path_eeg*: A string specifying the location of the architecture to use as the EEG encoder.
>	- *na_path_fmri*: A string specifying the location of the architecture to use as the fMRI encoder.
>	- *gpu_mem*: An integer specifying the GPU memory limit used to train the model. Please check your GPU specifications with the *nvidia-smi* commmand.
>	- *verbose*: A flag that specifies the verbosity of the script.
>	- *save_metrics*: A flag that specifies to whether or not to save the results.
>	- *metrics_path*: A string specifying the path to which the results are saved. Needs *save_metrics* to be True.
>	- *T*: An integer specifying the [Monte Carlo iterations](https://en.wikipedia.org/wiki/Monte_Carlo_method) to gather the results of variational models.
>	- *seed*: An integer specifying the random seed to use for random generator.

### Examples:

The command to retrieve metrics of the EEG to fMRI model on the NODDI dataset, with **topographical attention**, **random fourier features**:
```
python main.py metrics 01 -topographical_attention -fourier_features -random_fourier
```
Another example of retrieving the plots of each fMRI synthesized view with the EEG to fMRI model trained on the CN-EPFL dataset, with no topographical attention and no fourier feature projection:
```
python main.py quality 03
```
With the latter however you will not be saving the results, so do not forget to add the **save_metrics** flag and the **metrics_path** specification:
```
python main.py quality 03 -save_metrics -metrics_path /tmp/.
```

## Layers

In this section we will go over the layers that were implemented in this repository.

### Bayesian Module

This model implements variational layers:

- Dense Variational layer

#### Dense Varitational

This layer is the bayesian version of the [tf.keras.layers.Dense]() layer. It is parametrized by $$W,b$$ and transforms the inputs, $$X\in \mathbb{R}^{N}$$, as:

$$a(W^\top X + b)$$

The parameters are variational and are setup as:

$$W = W_{\mbox{posterior}} + W_{\mbox{prior}} \times \epsilon: epsilon \sim \mathcal{N}(0,1)^{N \times \mbox{units}} \wedge W_{\mbox{posterior}}, W_{\mbox{prior}} \in \mathbb{R}^{N \times \mbox{units}},$$

and for the bias:

$$b = b_{\mbox{posterior}} + b_{\mbox{prior}} \times \epsilon: epsilon \sim \mathcal{N}(0,1)^{\mbox{units}} \wedge b_{\mbox{posterior}}, b_{\mbox{prior}} \in \mathbb{R}^{\mbox{units}},$$

> Arguments:
>	- *

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
> Methods:
>	- *call*: returns the output given an input;
>	- *get_config*: returns a dictionary with the configuration needed to serialize the layer (see [layer serialization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/serialize));
>	- *from_config*: returns a *DenseVariational* instanced class with the configuration received;

### FFT Module

This module implements various [Discrete Cosine Transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform) layers in tensorflow, namely:

- Discrete Cosine Transform 3-Dimensional
- Inverse Discrete Cosine Transform 3-Dimensional
- Padded Inverse Discrete Cosine Transform 3-Dimensional
- Variational Inverse Discrete Cosine Transform 3-Dimensional

#### *DCT3D* Discrete Cosine Transform 3-Dimensional

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


#### *iDCT3D* Inverse Discrete Cosine Transform 3-Dimensional

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


#### *padded_iDCT3D* Padded Inverse Discrete Cosine Transform 3-Dimensional

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


#### *variational_iDCT3D* Variational Inverse Discrete Cosine Transform 3-Dimensional

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

### Random Fourier Module

This module is a wrap of the implementation done in [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/RandomFourierFeatures), but some tweaks were needed to git it into our methodology. The file is similar to the tensorflow implementation.

### Topographical Attention Module

This module implements the topographical attention presented in this [paper](https://arxiv.org/abs/2203.03481).

#### *Topographical_Attention* Topographical Attention

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


### Mask Module

In this module one can find implementations of layers that performs segmentation over a 3-dimensional MRI volume representation.

#### *MRICircleMask* MRI Volume Circle Mask

This is a naive implementation of a 3D circle brain mask that is fixed for all instances.

> Arguments:
>	- *input_shape*: tuple, giving the dimensions of the fMRI volume, can be of length 5 (meaning the batch dimension is included along with the channels dimension, the channels dimension here refers to the the convolutional filters dimension) or of length 4 (meaning the batch dimension is not included);
>	- *radius*: float, specifying the size of the circle brain mask;
>
> Methods:
>	- *call*: returns the input representation filtered with the circle 3D mask designed in this layer;

### Resnet Block Module

This module allows an easy integration of neural architecture specifications that were generated automatically, according to the methodology described in this [paper](https://openreview.net/forum?id=TCvkaP15O7e).


#### *ResBlock* Resnet Block ([He et al. 2015](https://arxiv.org/abs/1512.03385))

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


## Models

The models implemented in this repository are:

- **fMRI AE** at [models.fmri_ae.fMRI_AE](https://github.com/DCalhas/eeg_to_fmri/blob/5af97ee5d4891ecea11a93268f9121fa4963d4b4/src/models/fmri_ae.py#L43);
- **EEG to fMRI** at [models.eeg_to_fmri.EEG_to_fMRI](https://github.com/DCalhas/eeg_to_fmri/blob/ddaa57fe0225d298b45bd09ab77c71add0e168ed/src/models/eeg_to_fmri.py#L78);
- **Linear Classifier** at [models.classifiers.LinearClassifier](https://github.com/DCalhas/eeg_to_fmri/blob/ddaa57fe0225d298b45bd09ab77c71add0e168ed/src/models/classifiers.py#L5);
- **EEG View Classifier** at [models.classifiers.view_EEG_classifier](https://github.com/DCalhas/eeg_to_fmri/blob/ddaa57fe0225d298b45bd09ab77c71add0e168ed/src/models/classifiers.py#L27);

You may find more models implemented in the [models](https://github.com/DCalhas/eeg_to_fmri/tree/master/src/models) folder. Please beware that the latter may be either stale or deprecated.

	
### fMRI AE

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
	
### EEG to fMRI

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

#### Neural Architecture Specification

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


#### Example build

```
from models.eeg_to_fmri import EEG_to_fMRI

model = EEG_to_fMRI((7,7,7),
					EEG_SHAPE,**(None,))

```

### Linear Classifier


### EEG View Classifier

