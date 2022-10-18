---
layout: default
title: EEG recording to fMRI volume
parent: Blog
nav_order: 1
mathjax: true
tags: 
  - latex
  - math
---


# EEG recoding to fMRI volume

In this post I will go over the procedure to project an EEG instance to an fMRI.

In this (paper)[https://arxiv.org/abs/2203.03481], the methodology to do this projection is presented. The architecture of this model is shown in the Figure below.

<p align="center">
	<img src="./figures/architecture_eeg_benefits.png" width="400"/>
</p>

This model processes two inputs:
	- EEG representation $$\vec{x} \in \mathbb{R}^{C \times F \times T}$$;
	- fMRI volume representation $$\vec{y} \in \mathbb{R}^{M_1 \times M_2 \times M_3}$$

The EEG has a drift in relation to the associated fMRI volume, since it takes into consideration $$10\times \mbox{TR}$$ seconds in total (for the NODDI dataset this corresponds to $$10\times 2.160=21.6$$ seconds).