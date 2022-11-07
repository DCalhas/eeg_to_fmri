---
layout: default
title: Install EEG to fMRI
nav_order: 2
permalink: /install
---

# Installation guide

This install setup is tested for Linux (Ubuntu) with a GPU.

It is recommended to create an anaconda environment and install all sources there. For this, please go to [anaconda](https://www.anaconda.com/) and install. After you're done with installing anaconda, please setup the environment for the ```eeg-to-fmri``` package with:

```shell
source $PATH_TO_ANACONDA/bin/activate
conda create -n eeg_fmri python=3.8
source $PATH_TO_ANACONDA/etc/profile.d/conda.sh
conda activate eeg_fmri
```

If you installed anaconda at ```/home/you/anaconda3``` then that is your $PATH_TO_ANACONDA variable.

Following, let us setup cuda and cudnn. Download [cudnn](https://developer.nvidia.com/cudnn):

```shell
wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.0.1/cudnn-11.0-linux-x64-v8.0.1.13.tgz
```

Install ```cudatoolkit==11.2``` available at conda-forge:

```shell
conda install -c conda-forge cudatoolkit==11.2
```

And setup cudnn:

```shell
tar -xzvf cudnn-11.2-linux-x64-v8.1.1.33.tgz
mv cuda/include/cudnn*.h $CONDA_PREFIX/include/
mv cuda/lib64/libcudnn* $CONDA_PREFIX/lib/
rm -r cuda
chmod a+r $CONDA_PREFIX/include/cudnn*.h $CONDA_PREFIX/lib/libcudnn*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

Now, you can finally install the ```eeg-to-fmri``` package:

```shell
pip install --upgrade eeg-to-fmri
```

Some additional steps are needed to ensure the package runs accordingly. You need to specify the path for your datasets. For instance, I have my datasets at ```/home/david/datasets/```, then the environment variable for the datasets should be ```EEG_FMRI_DATASETS=/home/david/datasets/```. Since this variable only needs to be active, when running the ```eeg-to-fmri``` code, please put it in the activate folder, so everytime you activate the environemnt the variable exists.

```shell
echo 'export EEG_FMRI_DATASETS=/path/to/datasets/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```