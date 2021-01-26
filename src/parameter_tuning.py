import tensorflow as tf

import numpy as np

import GPyOpt

import argparse

from utils import tf_config, preprocess_data, search_algorithms

from models import fmri_ae, eeg_to_fmri, uniconv_fmri

import matplotlib.pyplot as plt

import gc

import os


parser = argparse.ArgumentParser()
parser.add_argument('model',
					choices=['fmri_ae', 'eeguniwit_to_fmriae'],
					help="Which model to run hyperparameter tuning algorithm")
parser.add_argument('algorithm',
					choices=['bo'],
					help="Which algorithm to run the search")
parser.add_argument('-iterations', default=100, type=int, help="Search algorithm iterations")
parser.add_argument('-dataset', default="01", type=str, help="Dataset to use")
parser.add_argument('-gpu_mem', default=800, type=int, help="GPU memory limit")
parser.add_argument('-out_file', default="out.txt", type=str, help="Output file")
parser.add_argument('-n_individuals', default=8, type=int, help="Number individuals for hyperparameter algorithm (train+validation)")
parser.add_argument('-interval_eeg', default=6, type=int, help="Interval of seconds to include EEG recording: default is 6 that means 6x2 seconds")
parser.add_argument('-random_skip', action="store_true", help="Allow random skip connections through the network")
parser.add_argument('-verbose', default=0, type=int, help="0 - no prints, 1 - level prints...")
opt = parser.parse_args()

np.random.seed(42)

model_name = opt.model
algorithm = opt.algorithm
iterations = opt.iterations
dataset=opt.dataset
memory_limit=opt.gpu_mem
out_file=opt.out_file
n_individuals=opt.n_individuals
interval_eeg=opt.interval_eeg
random_skip=opt.random_skip
verbose=opt.verbose

if(algorithm == "bo"):
	algorithm = search_algorithms.Bayesian_Optimization

file_output = open(out_file, 'w')

tf_config.setup_tensorflow(memory_limit=memory_limit)

with tf.device('/CPU:0'):
	train_data, val_data = preprocess_data.dataset(dataset, n_individuals=n_individuals, interval_eeg=interval_eeg, file_output=file_output, verbose=True)
	eeg_train, fmri_train =train_data
	eeg_val, fmri_val =val_data


if(model_name == "fmri_ae"):
	model_class = fmri_ae
	input_shape = fmri_train.shape[1:]
	data = (fmri_train, fmri_val)
elif(model_name == "eeguniwit_to_fmriae"):
	model_class = uniconv_fmri
	input_shape = eeg_train.shape[1:]
	data = (fmri_train, fmri_val, eeg_train, eeg_val)

param_tuner = algorithm(iterations, model_class, input_shape)
param_tuner.set_hyperparameters(model_class.search_space)
param_tuner.set_data(*data)
param_tuner.run(file_output=file_output)


