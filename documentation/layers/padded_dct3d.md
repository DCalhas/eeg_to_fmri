---
layout: default
title: Padded iDCT3D
parent: Layers
grand_parent: Documentation
nav_order: 4
mathjax: true
tags: 
  - latex
  - math
---

# *padded_iDCT3D* Padded Inverse Discrete Cosine Transform 3-Dimensional

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
