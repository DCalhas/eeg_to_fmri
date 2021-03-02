import tensorflow as tf

import numpy as np

import random

import argparse

from utils import tf_config, preprocess_data, search_algorithms, train

from models import fmri_ae, eeg_to_fmri, uniconv_fmri

from layers import locally_connected

import matplotlib.pyplot as plt

import gc

import os

from sklearn.model_selection import train_test_split, KFold

from scipy.stats import normaltest, ttest_ind, wilcoxon


parser = argparse.ArgumentParser()
parser.add_argument('technique1',
					choices=['encoder_conv', 'encoder_nonlocal', 'encoder_attention'],
					help="Technique to be compared")
parser.add_argument('technique2',
					choices=['encoder_conv', 'encoder_nonlocal', 'encoder_attention'],
					help="Technique to be compared")
parser.add_argument('-splits', default=10, type=int, help="Number of splits for validation")
parser.add_argument('-outfilter', default=0, type=int, help="Number of splits for validation")
parser.add_argument('-epochs', default=10, type=int, help="Number of epochs for the training session")
parser.add_argument('-gpu_mem', default=1500, type=int, help="Memory limit for gpu")
parser.add_argument('-seed', default=42, type=int, help="Seed for random state")
opt = parser.parse_args()

technique1 = opt.technique1
technique2 = opt.technique2
outfilter = opt.outfilter
splits = opt.splits
epochs = opt.epochs
seed = opt.seed
gpu_mem = opt.gpu_mem

if(technique1 == "encoder_conv"):
	local_1=True
	local_attention_1=False
elif(technique1 == "encoder_nonlocal"):
	local_1=False
	local_attention_1=False
elif(technique1 == "encoder_attention"):
	local_1=True
	local_attention_1=True

if(technique2 == "encoder_conv"):
	local_2=True
	local_attention_2=False
elif(technique2 == "encoder_nonlocal"):
	local_2=False
	local_attention_2=False
elif(technique2 == "encoder_attention"):
	local_2=True
	local_attention_2=True

dataset="01"
n_individuals=8
interval_eeg=6

tf_config.set_seed(seed=seed)
tf_config.setup_tensorflow(device="GPU", memory_limit=gpu_mem)

with tf.device('/CPU:0'):
	tf_config.set_seed(seed=seed)
	train_data, _ = preprocess_data.dataset(dataset, n_individuals=n_individuals, interval_eeg=interval_eeg, verbose=True)
	_, fmri_train =train_data
	#eeg_val, fmri_val =val_data
	
	fmri_train = fmri_train[:296]
	
	kf = KFold(shuffle=False, random_state=seed, n_splits=splits)

tf_config.set_seed(seed=seed)

batch_size=16
learning_rate=0.001
skip_connections=True
maxpool=True
batch_norm=True
weight_decay=1e-3
n_channels=16
latent_dimension=(5,5,5)
kernel_size=(9,9,4)
stride_size=(1,1,1)
n_stacks=3

tf_config.set_seed(seed=seed)
optimizer = tf.keras.optimizers.Adam(learning_rate)
loss_fn = tf.keras.losses.MSE


dev_losses_1 = []
dev_losses_2 = []

fold = 1
for train_idx, dev_idx in kf.split(fmri_train):
	with tf.device('/CPU:0'):
		tf_config.set_seed(seed=seed)
		train_set = tf.data.Dataset.from_tensor_slices((fmri_train[train_idx], fmri_train[train_idx])).batch(batch_size)
		dev_set = tf.data.Dataset.from_tensor_slices((fmri_train[dev_idx], fmri_train[dev_idx])).batch(1)

		model = fmri_ae.fMRI_AE(latent_dimension, fmri_train.shape[1:], 
					kernel_size, stride_size, n_channels,
					maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay, 
					skip_connections=skip_connections, n_stacks=n_stacks, 
					local=local_1, local_attention=local_attention_1, outfilter=outfilter,
					seed=seed)

	#train
	train_loss, val_loss = train.train(train_set, model, optimizer, 
									loss_fn, epochs=epochs, 
									val_set=dev_set, verbose=True)

	dev_losses_1.append(val_loss[-1])

	gc.collect()
	tf.keras.backend.clear_session()
	tf_config.set_seed(seed=seed)

	with tf.device('/CPU:0'):
		tf_config.set_seed(seed=seed)
		model = fmri_ae.fMRI_AE(latent_dimension, fmri_train.shape[1:], 
					kernel_size, stride_size, n_channels,
					maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay, 
					skip_connections=skip_connections, n_stacks=n_stacks, 
					local=local_2, local_attention=local_attention_2, outfilter=outfilter,
					seed=seed)

	#train
	train_loss, val_loss = train.train(train_set, model, optimizer, 
									loss_fn, epochs=epochs, 
									val_set=dev_set, verbose=True)

	dev_losses_2.append(val_loss[-1])

	gc.collect()
	tf.keras.backend.clear_session()
	tf_config.set_seed(seed=seed)

	print("Finished fold ", fold)
	fold += 1

print("Model 1 with dev losses: ")
print(dev_losses_1)

print("Model 2 with dev losses: ")
print(dev_losses_2)



#Get significance
dev_losses_1 = np.array(dev_losses_1)
dev_losses_2 = np.array(dev_losses_2)

_, pv_normal1 = normaltest(dev_losses_1)
_, pv_normal2 = normaltest(dev_losses_2)

if(pv_normal2 > 0.05 and pv_normal1 > 0.05):
	_, pvalue = wilcoxon(dev_losses_1, dev_losses_2)
else:
	#two normal distributions
	_, pvalue = ttest_ind(dev_losses_1, dev_losses_2)

print("Difference with p-value of ", pvalue)

print("Model " + technique1 + " with mean K-Fold validation MSE: ", np.mean(dev_losses_1), "+/-", np.std(dev_losses_1))
print("Model " + technique2 + " with mean K-Fold validation MSE: ", np.mean(dev_losses_2), "+/-", np.std(dev_losses_2))