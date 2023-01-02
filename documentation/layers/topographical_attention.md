---
layout: default
title: Topographical Attention
parent: Layers
grand_parent: Documentation
nav_order: 6
mathjax: true
tags: 
  - latex
  - math
---


# *Topographical_Attention* Topographical Attention

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
