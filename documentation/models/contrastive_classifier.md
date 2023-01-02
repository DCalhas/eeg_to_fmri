---
layout: default
title: EEG View Contrastive Classifier
parent: Models
grand_parent: Documentation
nav_order: 4
mathjax: true
tags: 
  - latex
  - math
---

# EEG View Classifier

This class uses Sinusoid separation explained in [this blog post](https://dcalhas.github.io/eeg_to_fmri/blog/Sinusoid_separation.html).

The name of this class is ```ViewLatentContrastiveClassifier```. You find in the file several variations to classify a view, however this class implements the methodology that is able to separate accurately the data.


> **Attributes:**
>
>	- *model*: models.eeg_to_fmri.EEG_to_fMRI, gives the pretrained model specification to be used for fMRI synthesis;
>	- *degree*: int, specifies the polynomial order of the linear transformation applied. Default is 1, which accounts for normal linear transformation;
>	- *activation*: tf.keras.activations.Activation, specifies the activation used in the encoder part of the EEG to fMRI synthesizer model;
>	- *regularizer*: tf.keras.regularizers.Regularizer, specifies the regularization applied to the parameters;
>	- *feature_selection*: bool, specifies whether to compute a masking mechanism to the synthesized fMRI volume. This is deprecated, since after implementing sinusoidal separation along with style prior, one does not need this type of processing;
>	- *segmentation_mask*: bool, specifies whether the synthesized fMRI goes through a mask, in order to avoid background noise. This is deprecated, since after implementing sinusoidal separation along with style prior, one does not need this type of processing;
>	- *siamese_projection*: bool, specifies whether in addition to the sinusoidal contrastive separation, one also performs contrastive separation in a new dimension;
>	- *siamese_projection_dimension*: int, the size of the auxiliary dimension where a contrastive loss is applied;
>	- *variational*: bool, specifies if the transformation (to classify) is done using the [DenseVariational](https://github.com/DCalhas/eeg_to_fmri/blob/550627618b402cb80b10b020bde996d8a38bc88e/src/layers/bayesian.py#L5) layer (if True) or the [tf.keras.layers.Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) layer (if False);
>	- *seed*: int, the seed used to generate random numbers;