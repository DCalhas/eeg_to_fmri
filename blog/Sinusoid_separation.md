---
layout: default
title: Classification on EEG only datasets
parent: Blog
nav_order: 2
mathjax: true
tags: 
  - latex
  - math
---



# Classification on EEG only datasets

In this post I will talk about an approach that is able to use sinusoids to generate images [[1](#references)], while being able to separate the data according to labels.



<p align="center">
	<img src="./figures/cosine_image.png" width="400"/>
</p>


### Constrastive loss in the unit circle


<p align="center">
	<img src="./figures/contrastive_optimization.png" width="400"/>
</p>


### Bayesian versus Deterministic


<p align="center">
	<img src="./figures/bayesian_vs_deterministic.png" width="400"/>
</p>





## References

\[1\]: [Tancik, Matthew, et al. Fourier features let networks learn high frequency functions in low dimensional domains. Advances in Neural Information Processing Systems, 2020, 33: 7537-7547.](https://arxiv.org/abs/2006.10739)