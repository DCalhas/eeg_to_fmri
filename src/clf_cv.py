from utils import process_utils

import argparse

import multiprocessing

from multiprocessing import Manager

from pathlib import Path

import shutil

import os

import time

import numpy as np

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset_synth', choices=['01', '03'], help="Which dataset to load for synthesis")
	parser.add_argument('dataset_clf', choices=['10', '11'], help="Which dataset to load for classification")
	parser.add_argument('-epochs', default=10, type=int, help="Number of epochs")
	parser.add_argument('-lr', default=0.001, type=float, help="Learning rate")
	parser.add_argument('-batch_size', default=4, type=int, help="Batch size in training session")
	parser.add_argument('-gpu_mem', default=1500, type=int, help="GPU memory limit")
	parser.add_argument('-path_save_network', default="/tmp/network_synthesis", type=str, help="Path to save neural network synthesis architecture")
	parser.add_argument('-path_labels', default="/tmp/", type=str, help="Path to save labels of classification task, should be a directory")
	parser.add_argument('-seed', default=42, type=int, help="Seed for random state")
	opt = parser.parse_args()

	dataset_synth=opt.dataset_synth
	dataset_clf=opt.dataset_clf
	epochs=opt.epochs
	learning_rate=opt.lr
	batch_size=opt.batch_size
	gpu_mem=opt.gpu_mem
	path_save_network=opt.path_save_network
	seed=opt.seed
	path_labels=opt.path_labels


#train neural network synthesis
process_utils.launch_process(process_utils.train_synthesis, 
							(dataset_synth, epochs, path_save_network, gpu_mem, seed))


#create predictions and true labels
process_utils.launch_process(process_utils.create_labels,
							(dataset_clf, path_labels))


#create predictions and true labels
process_utils.setup_data_loocv(dataset_clf, epochs, learning_rate, batch_size, gpu_mem, seed, path_save_network, path_labels)

