---
layout: default
title: Layers
parent: Documentation
nav_order: 1
has_children: true
mathjax: true
permalink: documentation/layers
tags: 
  - latex
  - math
---

# Layers

In this section we will go over the layers that were implemented in this repository.

## Bayesian Module

This model implements variational layers:

- Dense Variational layer


## FFT Module

This module implements various [Discrete Cosine Transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform) layers in tensorflow, namely:

- Discrete Cosine Transform 3-Dimensional
- Inverse Discrete Cosine Transform 3-Dimensional
- Padded Inverse Discrete Cosine Transform 3-Dimensional
- Variational Inverse Discrete Cosine Transform 3-Dimensional

## Random Fourier Module

This module is a wrap of the implementation done in [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/RandomFourierFeatures), but some tweaks were needed to get it to work with our methodology. The file is similar to the tensorflow implementation.

## Topographical Attention Module

This module implements the topographical attention presented in this [paper](https://arxiv.org/abs/2203.03481).


## Resnet Block Module

This module allows an easy integration of neural architecture specifications that were generated automatically, according to the methodology described in this [paper](https://openreview.net/forum?id=TCvkaP15O7e).
