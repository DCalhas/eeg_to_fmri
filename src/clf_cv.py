from utils import process_utils, assertion_utils, memory_utils

import argparse

import multiprocessing

from multiprocessing import Manager

from pathlib import Path

import shutil

MAX_PROCESSES=200

import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'#limit the number of threads for numpy

import time

import numpy as np

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset_clf', choices=['10', '11'], help="Which dataset to load for classification")
	parser.add_argument('view', choices=['raw', 'stft', 'fmri'], help="Which view to consider for classification")
	parser.add_argument('-dataset_synth', default="01", type=str, help="Which dataset to load for synthesis")
	parser.add_argument('-feature_selection', action="store_true", help="Perform feature selection with low resolution")
	parser.add_argument('-segmentation_mask', action="store_true", help="Apply a brain segmentation mask")
	parser.add_argument('-style_prior', action="store_true", help="Whether to use a learned style prior or use the attention weights as a probe.")
	parser.add_argument('-fourier_norm', default="layer", type=str, help="Type of normalization to use at the sinusoids, it can be layer, tanh")
	parser.add_argument('-batch_norm_reg', action="store_true", help="Batch normalization regularizer for classification.")
	parser.add_argument('-padded', action="store_true", help="Fill higher resolutions with zero for the upsampling method.")
	parser.add_argument('-variational', action="store_true", help="Variational implementation of the model")
	parser.add_argument('-variational_clf', action="store_true", help="Variational linear classifier with reparametrization trick.")
	parser.add_argument('-variational_coefs', default=None, type=None, help="Number of extra stochastic resolution coefficients")
	parser.add_argument('-variational_dependent_h', default=None, type=int, help="Apply dependency mechanism on X to get high frequency coefficient\nDimension of the hidden boundary decision for stochastic heads")
	parser.add_argument('-variational_dist', default="Normal", type=str, help="Distribution used for the high resolution coefficients")
	parser.add_argument('-variational_random_padding', action="store_true", help="Whether to randomize positions in the DCT frequency space instead of predicting low resolution coefficients")
	parser.add_argument('-resolution_decoder', default=None, type=float, help="Resolution decoder intermediary before final transformation in decoder -- used in uncertainty")
	parser.add_argument('-aleatoric_uncertainty', action="store_true", help="Aleatoric uncertainty flag")
	parser.add_argument('-fold', default=0, type=int, help="Fold to start leave one out at - useful when bizantine fault error occurs and restart at state (fold)")
	parser.add_argument('-folds', default=5, type=int, help="Folds to consider in CV hyperparameter optimization")
	parser.add_argument('-epochs', default=10, type=int, help="Number of epochs")
	parser.add_argument('-optimizer', default="Adam", type=str, help="Optimizer to use for the learning session")
	parser.add_argument('-n_processes', default=1, type=int, help="Number of processes to use during CV to launch folds and run independently.")
	parser.add_argument('-gpu_mem', default=1500, type=int, help="GPU memory limit")
	parser.add_argument('-path_save_network', default="/tmp/network_synthesis", type=str, help="Path to save neural network synthesis architecture")
	parser.add_argument('-path_labels', default="/tmp/", type=str, help="Path to save labels of classification task, should be a directory")
	parser.add_argument('-save_explainability', action="store_true", help="save explainability features")
	parser.add_argument('-run_eagerly', action="store_true", help="Run eagerly, if not it runs in graph mode. This is important for activity regularization")
	parser.add_argument('-verbose', action="store_true", help="Whether to put verbosity in the training of the neural netowkrs")
	parser.add_argument('-seed', default=42, type=int, help="Seed for random state")
	opt = parser.parse_args()
	
	setting,dataset_synth,dataset_clf,feature_selection,segmentation_mask,style_prior,fourier_norm,batch_norm_reg,padded,variational,variational_clf,variational_coefs,variational_dependent_h,variational_dist,variational_random_padding,resolution_decoder,aleatoric_uncertainty,view,fold,folds,n_processes,epochs,optimizer,gpu_mem,path_save_network,seed,run_eagerly,path_labels,save_explainability,verbose = assertion_utils.clf_cv(opt)

#limit process memory
memory_utils.limit_CPU_memory(1024*1024*1024*24, MAX_PROCESSES)

#create directory to save
if(not os.path.exists(path_labels+"/"+ setting)):
	os.makedirs(path_labels+"/"+ setting)

#train neural network synthesis
if("fmri" in view and fold==0):
	pass
	#process_utils.launch_process(process_utils.train_synthesis, 
	#							(dataset_synth, epochs, style_prior, padded, variational, variational_coefs, variational_dependent_h, variational_dist, variational_random_padding, resolution_decoder, False, fourier_norm, batch_norm_reg, path_save_network, gpu_mem, seed, run_eagerly, verbose))

#create predictions and true labels
if(fold==0):
	process_utils.launch_process(process_utils.create_labels,
								(view, dataset_clf, path_labels, setting))

#create predictions and true labels
process_utils.setup_data_loocv(setting, view, dataset_clf, fold, folds, n_processes, epochs, optimizer, gpu_mem, seed, run_eagerly, save_explainability, path_save_network, path_labels, feature_selection, segmentation_mask, aleatoric_uncertainty, style_prior, variational_clf, verbose)

#report classification metrics
process_utils.launch_process(process_utils.compute_acc_metrics, 
							(view, path_labels, setting))