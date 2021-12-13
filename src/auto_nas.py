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
	parser.add_argument('setup', choices=['eeg', 'fmri'], help="Which setup to run")
	parser.add_argument('dataset', choices=['01', '02'], help="Which dataset to load")
	parser.add_argument('-epochs', default=10, type=int, help="Number of epochs")
	parser.add_argument('-lr', default=0.01, type=float, help="Learning rate")
	parser.add_argument('-batch_size', default=128, type=int, help="Batch size in training session")
	parser.add_argument('-batch_path', default=str(Path.home())+"/eeg_to_fmri/src/tmp_batches", type=str, help="Batches path to be saved")
	parser.add_argument('-gpu_mem', default=1500, type=int, help="GPU memory limit")
	parser.add_argument('-networks', default=2, type=int, help="Number of neural architectures to consider")
	parser.add_argument('-na_path', default=str(Path.home())+"/eeg_to_fmri/na_models", type=str, help="Neural architectures path.")
	parser.add_argument('-save_weights', action="store_true", help="Save weights of softmax continuous technique.")
	parser.add_argument('-save_weights_path', default=str(Path.home())+"/eeg_to_fmri/softmax_weights", type=str, help="Path of directory to save save weights.")
	parser.add_argument('-best_eeg_path', default=str(Path.home())+"/eeg_to_fmri/na_models_eeg", type=str, help="Path of directory to save save weights.")
	parser.add_argument('-seed', default=42, type=int, help="Seed for random state")
	opt = parser.parse_args()

	setup=opt.setup
	memory_limit=opt.gpu_mem
	dataset=opt.dataset
	epochs=opt.epochs
	learning_rate=opt.lr
	batch_size=opt.batch_size
	batch_path=opt.batch_path
	networks=opt.networks
	save_weights=opt.save_weights
	save_weights_path=opt.save_weights_path
	na_path=opt.na_path
	seed=opt.seed
	best_eeg_path=opt.best_eeg_path
	
	if(dataset=="01"):
		x_dim=64
		y_dim=64
		z_dim=30


raw_eeg=False#time or frequency features? raw-time nonraw-frequency
resampling=False
if(dataset=="01"):
	n_volumes=300-3
	n_individuals=10
	n_individuals_train=6
	n_individuals_val=2
if(dataset=="02"):
	n_volumes=170-3
#parametrize the interval eeg?
interval_eeg=10

process_utils.launch_process(process_utils.make_dir_batches, (dataset, n_individuals, n_individuals_train, 
															n_individuals_val, n_volumes, interval_eeg, 
															memory_limit, batch_size, batch_path))


n_batches = int(len(os.listdir(batch_path))/2)

for epoch in range(epochs):

	for batch in range(1,n_batches+1):
		#do not train with last batch, save weights if needed
		if(batch+1==n_batches+1):
			if(save_weights):
				process_utils.launch_process(process_utils.save_weights, 
										(epoch, na_path, save_weights_path))
			continue

		epoch_start = time.time()

		flattened_predictions = Manager().Array('d', range(batch_size*x_dim*y_dim*z_dim))
		o_predictions = np.zeros((networks,batch_size,x_dim,y_dim,z_dim,1), dtype=np.float32)

		for network in range(networks):
			process_utils.launch_process(process_utils.batch_prediction, 
										(flattened_predictions, setup, batch_path, batch, epoch, network, na_path, 
											batch_size, learning_rate, memory_limit, best_eeg_path, seed))
			continue
			o_predictions[network]=np.array(flattened_predictions).reshape((batch_size,x_dim,y_dim,z_dim,1))

		#train weights that allow continuous representation of the neural networks
		process_utils.launch_process(process_utils.continuous_training, 
									(o_predictions, batch_path, batch, learning_rate, epoch, na_path, memory_limit, seed))

		print("Took: ", time.time()-epoch_start, " seconds")

	print("Finished epoch ", epoch)
			

#removing batch directory
shutil.rmtree(batch_path)