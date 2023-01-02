---
layout: default
title: Dense Variational
parent: Layers
nav_order: 1
mathjax: true
tags: 
  - latex
  - math
---



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
