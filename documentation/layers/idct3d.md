---
layout: default
title: iDCT3D
parent: Layers
grand_parent: Documentation
nav_order: 3
mathjax: true
tags: 
  - latex
  - math
---


# *iDCT3D* Inverse Discrete Cosine Transform 3-Dimensional

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

