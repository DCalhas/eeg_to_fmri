---
layout: default
title: Variational iDCT3D
parent: Layers
grand_parent: Documentation
nav_order: 7
mathjax: true
tags: 
  - latex
  - math
---

# *variational_iDCT3D* Variational Inverse Discrete Cosine Transform 3-Dimensional

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
