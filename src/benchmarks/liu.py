import tensorflow as tf

import tensorflow_probability as tfp

if __name__ == "__main__":

	import sys

	sys.path.append("..")

from utils import data_utils, fmri_utils, eeg_utils, losses_utils

import numpy as np

"""
IEEE-Conference on Neural Engineering

This is a reproduction of the paper Liu et al. 2019, A Convolutional Neural Network for Transcoding Simultaneously Acquired EEG-fMRI Data

Disclaimer: This is not the official implementation and can contain operations that were not inteded by the authors of the paper

Example usage:
>>> from benchmarks import liu
>>> model = liu.Liu_et_al((64,10),(64,64,30,10))
>>> train_set, test_set = model.load_data("01",10)
"""
class Liu_et_al(tf.keras.Model):

	"""
	Inputs:
		* eeg_shape - tuple (n_channels, time)
		* fmri_shape - tuple (X,Y,Z,T)

	max number of channels to run on the personal computer is 4
	standard number is 16
	"""
	def __init__(self, eeg_shape, fmri_shape, n_channels=1, latent_dim=256, variational=False):
		super(Liu_et_al, self).__init__()

		self.eeg_shape = eeg_shape
		self.fmri_shape = fmri_shape

		self.time_dimension=fmri_shape[-1]
		self.spatial_dimension=fmri_shape[0]*fmri_shape[1]*fmri_shape[2]

		self.n_channels=n_channels
		self.latent_dim=latent_dim

		self.variational=variational
		self.fn=tf.keras.layers.Conv1D
		if(self.variational):
			self.fn=tfp.layers.Convolution1DFlipout

		self.build_model()

	def build_model(self):

		input_shape = tf.keras.layers.Input(shape=(self.eeg_shape[1], self.eeg_shape[0]))
		
		x = tf.keras.layers.Dense(self.latent_dim)(input_shape)
		#8*8*4=256
		x = tf.keras.layers.Reshape((8,8,4,self.fmri_shape[-1]))(x)

		#kernel size to map to fmri spatial dimension sizes
		#x = tf.keras.layers.Conv3DTranspose(self.time_dimension, kernel_size=(9,9,4))(x)

		#16x16x7x300 go through 12 layers that maintain dimension so kernel and stride should be both 1
		#channel dimension becomes 16 so 300*16
		for i in range(1):
			x = tf.keras.layers.Conv3D(self.time_dimension*self.n_channels, kernel_size=1, strides=1)(x)
		#channel dimension goes back to 1
		x = tf.keras.layers.Conv3D(self.time_dimension, kernel_size=1, strides=1)(x)
		
		#reshape to perform convolution on the time dimension "Temporal Convolutional Layer"
		x = tf.keras.layers.Reshape((self.time_dimension,self.latent_dim))(x)
		#now goes to the time slicing part 16*16*7=1792
		for i in range(1):
			x = self.fn(self.latent_dim*self.n_channels, kernel_size=1, strides=1)(x)
		x = tf.keras.layers.Conv1D(self.spatial_dimension, kernel_size=1, strides=1)(x)

		x = tf.keras.layers.Reshape(self.fmri_shape)(x)
		
		if(self.variational):
			x = [x, tf.keras.layers.Dense(1, activation=tf.keras.activations.exponential)(x)]

		self.nn=tf.keras.Model(input_shape, x)

	"""
	input_shape: (channels,time,1)
	output_shape: (X,Y,Z,time)
	"""
	def call(self, x, *args):
		return self.nn(x)


	"""
	Each benchmark has its own type of features
	Outputs:
		*tuple - (tf.DataLoader, tf.DataLoader)
	"""
	def load_data(self, dataset, time_length=10, batch_size=4):
		n_individuals=getattr(data_utils, "n_individuals_"+dataset)
		
		with tf.device('/CPU:0'):
			eeg_train, fmri_train,_ = data_utils.get_data(range(n_individuals),
														dataset=dataset,
														raw_eeg=True,
														#resample_raw_eeg=True,
														standardize_fmri=True,
														ind_volume_fit=False,
														iqr_outlier=False,
														raw_eeg_resample=True,
														eeg_resample=getattr(fmri_utils, "TR_"+dataset), 
														fmri_resolution_factor=1)

		n_individuals_train = getattr(data_utils, "n_individuals_train_"+dataset)
		n_individuals_test = getattr(data_utils, "n_individuals_test_"+dataset)
		n_volumes = getattr(fmri_utils, "n_volumes_"+dataset)

		x_eeg = np.empty((n_individuals*int(n_volumes/time_length),)+(eeg_train.shape[1], time_length))
		x_fmri = np.empty((n_individuals*int(n_volumes/time_length),)+fmri_train.shape[1:]+(time_length,))

		instance=0
		for individual in range(n_individuals):
			for index_volume in range(individual*(n_volumes), individual*(n_volumes)+n_volumes-time_length, time_length):
				x_eeg[instance] = np.transpose(eeg_train[index_volume:index_volume+time_length], (1,0))
				x_fmri[instance] = np.transpose(fmri_train[index_volume:index_volume+time_length], (1,2,3,0))
				instance+=1

		eeg_train, eeg_test = x_eeg[:int(n_volumes/time_length)*n_individuals_train], x_eeg[int(n_volumes/time_length)*n_individuals_train:int(n_volumes/time_length)*(n_individuals_train+n_individuals_test)]
		fmri_train, fmri_test = x_fmri[:int(n_volumes/time_length)*n_individuals_train], x_fmri[int(n_volumes/time_length)*n_individuals_train:int(n_volumes/time_length)*(n_individuals_train+n_individuals_test)]

		eeg_train = np.swapaxes(eeg_train, 1, 2)
		eeg_test = np.swapaxes(eeg_test, 1, 2)

		eeg_train = eeg_train.astype('float32')
		fmri_train = fmri_train.astype('float32')
		eeg_test = eeg_test.astype('float32')
		fmri_test = fmri_test.astype('float32')

		return tf.data.Dataset.from_tensor_slices((eeg_train, fmri_train)).batch(batch_size), tf.data.Dataset.from_tensor_slices((eeg_test, fmri_test)).batch(1)

	def __train__(self, train_set, loss_fn, opt):
		raise NotImplementedError

	def __evaluate__(self, test_set, metric_fn):
		raise NotImplementedError

	def get_loss(self):
		if(self.variational):
			return self.variational_loss
		return losses_utils.mae

	def variational_loss(self, y_true, y_pred):
		return tf.reduce_mean((1/(y_pred[1]+losses_utils.NON_DIVISION_ZERO))*tf.math.abs(y_pred[0] - y_true)+tf.math.log(2*(y_pred[1]+losses_utils.NON_DIVISION_ZERO)), axis=(1,2,3))




if __name__ == "__main__":

	from utils import tf_config	

	import GPyOpt

	import argparse

	from pathlib import Path

	import os

	from utils import train, metrics

	import gc

	parser = argparse.ArgumentParser()
	parser.add_argument('dataset', choices=['01', '02', '03', '04', '05', 'NEW'], help="Which dataset to load")
	parser.add_argument('-variational', action="store_true", help="Variational implementation of the model")
	parser.add_argument('-interval_eeg', default=1, type=int, help="interval eeg")
	parser.add_argument('-T', default=10, type=int, help="Monte Carlo Simulation number of samples taken to approximate.")
	parser.add_argument('-memory_limit', default=4000, type=int, help="GPU memory limit")
	parser.add_argument('-seed', default=42, type=int, help="Seed for random generator")
	parser.add_argument('-save', action="store_true", help="Save metrics")
	parser.add_argument('-save_path', default=str(Path.home())+"/eeg_to_fmri/metrics", type=str, help="interval eeg")
	opt = parser.parse_args()

	dataset=opt.dataset
	variational=opt.variational
	interval_eeg=opt.interval_eeg
	T=opt.T
	memory_limit=opt.memory_limit
	save=opt.save
	save_path=opt.save
	seed=opt.seed

	setting=dataset+"_liu"
	if(variational):
		setting+="_variational"
	setting+="_interval_"+str(interval_eeg)

	tf_config.set_seed(seed=seed)
	tf_config.setup_tensorflow(device="GPU", memory_limit=memory_limit)

	with tf.device('/CPU:0'):
		model = Liu_et_al((len(getattr(eeg_utils, "channels_"+dataset)),interval_eeg),
							getattr(fmri_utils, "fmri_shape_"+dataset)+(interval_eeg,),
							variational=variational)
		
		optimizer = tf.keras.optimizers.Adam(0.001)
		loss_fn = model.get_loss()

		train_set, test_set = model.load_data(dataset, time_length=interval_eeg, batch_size=2)


	train.train(train_set, model, optimizer, loss_fn, epochs=10, u_architecture=False, verbose=True)

	rmse_pop = metrics.rmse(test_set, model)
	gc.collect()
	ssim_pop = metrics.ssim(test_set, model)
	gc.collect()
	res_pop = metrics.residues(test_set, model, variational=variational, T=T)
	gc.collect()
	print("RMSE: ", np.mean(rmse_pop), "\pm", np.std(rmse_pop))
	print("SSIM: ", np.mean(ssim_pop), "\pm", np.std(ssim_pop))
	print("RES: ", np.mean(res_pop), "\pm", np.std(res_pop))

	#save
	if(save):
		if(not os.path.exists(save_path+"/"+setting)):
			os.makedirs(save_path+"/"+setting)
		with open(save_path+"/"+setting+"/res_"+"seed_"+str(seed)+".npy", 'wb') as f:
			np.save(f, res_pop)
		with open(save_path+"/"+setting+"/rmse_"+"seed_"+str(seed)+".npy", 'wb') as f:
			np.save(f, rmse_pop)
		with open(save_path+"/"+setting+"/ssim_"+"seed_"+str(seed)+".npy", 'wb') as f:
			np.save(f, ssim_pop)