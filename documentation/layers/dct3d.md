---
layout: default
title: DCT3D
parent: Layers
grand_parent: Documentation
nav_order: 1
mathjax: true
tags: 
  - latex
  - math
---


# *DCT3D* Discrete Cosine Transform 3-Dimensional

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


