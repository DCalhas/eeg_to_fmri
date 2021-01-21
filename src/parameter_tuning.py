import tensorflow as tf

import numpy as np

import GPyOpt

import argparse

from utils import tf_config, preprocess_data, search_algorithms

from models import fmri_ae

import matplotlib.pyplot as plt

import gc


parser = argparse.ArgumentParser()
parser.add_argument('model',
					choices=['fmri_ae'],
					help="Which model to run hyperparameter tuning algorithm")
parser.add_argument('algorithm',
					choices=['bo'],
					help="Which algorithm to run the search")
parser.add_argument('-iterations', default="100", type=str, help="Search algorithm iterations")
parser.add_argument('-dataset', default="01", type=str, help="Dataset to use")
parser.add_argument('-gpu_mem', default=800, type=int, help="GPU memory limit")
parser.add_argument('-n_individuals', default=8, type=int, help="Number individuals for hyperparameter algorithm (train+validation)")
parser.add_argument('-interval_eeg', default=6, type=int, help="Interval of seconds to include EEG recording: default is 6 that means 6x2 seconds")
parser.add_argument('-random_skip', action="store_true", help="Allow random skip connections through the network")
parser.add_argument('-verbose', default=0, type=int, help="0 - no prints, 1 - level prints...")
opt = parser.parse_args()

np.random.seed(42)

model_name = opt.model
algorithm = opt.algorithm
memory_limit=opt.gpu_mem
n_individuals=opt.n_individuals
interval_eeg=opt.interval_eeg
dataset=opt.dataset
random_skip=opt.random_skip

if(model_name == "fmri_ae"):
	model_class = fmri_ae
if(algorithm == "bo"):
	algorithm = search_algorithms.Bayesian_Optimization


tf_config.setup_tensorflow(memory_limit=memory_limit)

train_data, val_data = preprocess_data.dataset(dataset, n_individuals=n_individuals, interval_eeg=interval_eeg, verbose=True)
_, fmri_train =train_data
_, fmri_val =val_data

iterations = 100
param_tuner = algorithm(iterations, fmri_ae, fmri_train[1:])
param_tuner.set_hyperparameters(fmri_ae.search_space)
param_tuner.set_data(fmri_train, fmri_val)
param_tuner.run()


