---
layout: default
title: Documentation
nav_order: 4
has_children: true
mathjax: true
permalink: documentation
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
	- 04: This corresponds to the [Neurinfo-Rennes Xp1 dataset](https://openneuro.org/datasets/ds002336/versions/2.0.0).
	- 05: This corresponds to the [Neurinfo-Rennes Xp2 dataset](https://openneuro.org/datasets/ds002336/versions/2.0.0).

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
