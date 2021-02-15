import tensorflow as tf

import numpy as np

import argparse

from utils import tf_config, preprocess_data, search_algorithms, train

from models import fmri_ae, eeg_to_fmri, uniconv_fmri

from layers import locally_connected

import matplotlib.pyplot as plt

import gc

import os

from sklearn.model_selection import train_test_split, KFold

from scipy.stats import normaltest, ttest_ind, wilcoxon

dataset="01"
memory_limit=1500
n_individuals=8
interval_eeg=6

tf_config.setup_tensorflow(device="GPU", memory_limit=memory_limit, seed=42)

with tf.device('/CPU:0'):
	train_data, _ = preprocess_data.dataset(dataset, n_individuals=n_individuals, interval_eeg=interval_eeg, verbose=True)
	_, fmri_train =train_data
	#eeg_val, fmri_val =val_data
	
	fmri_train = fmri_train[:296]
	
	kf = KFold(shuffle=True, random_state=42, n_splits=10)


batch_size=16
learning_rate=0.001
skip_connections=True
maxpool=True
batch_norm=True
weight_decay=1e-5
n_channels=16
latent_dimension=(5,5,5)
kernel_size=(9,9,5)
stride_size=(1,1,1)
n_stacks=3

optimizer = tf.keras.optimizers.Adam(learning_rate)
loss_fn = tf.keras.losses.MSE


dev_losses_1 = []
dev_losses_2 = []

fold = 1
for train_idx, dev_idx in kf.split(fmri_train):
	train_set = tf.data.Dataset.from_tensor_slices((fmri_train[train_idx], fmri_train[train_idx])).batch(batch_size)
	dev_set = tf.data.Dataset.from_tensor_slices((fmri_train[dev_idx], fmri_train[dev_idx])).batch(1)

	model = fmri_ae.fMRI_AE(latent_dimension, fmri_train.shape[1:], 
				kernel_size, stride_size, n_channels,
				maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay, 
				skip_connections=skip_connections, n_stacks=n_stacks, 
				local=True, local_attention=False)

	#train
	train_loss, val_loss = train.train(train_set, model, optimizer, 
									loss_fn, epochs=10, 
									val_set=dev_set, verbose=True)

	dev_losses_1.append(val_loss[-1])

	gc.collect()
	tf.keras.backend.clear_session()

	model = fmri_ae.fMRI_AE(latent_dimension, fmri_train.shape[1:], 
				kernel_size, stride_size, n_channels,
				maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay, 
				skip_connections=skip_connections, n_stacks=n_stacks, 
				local=False, local_attention=False)

	#train
	train_loss, val_loss = train.train(train_set, model, optimizer, 
									loss_fn, epochs=10, 
									val_set=dev_set, verbose=True)

	dev_losses_2.append(val_loss[-1])

	gc.collect()
	tf.keras.backend.clear_session()

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

	#two normal distributions
	_, pvalue = ttest_ind(dev_losses_1, dev_losses_2)


else:
	_, pvalue = wilcoxon(dev_losses_1, dev_losses_2)

print("Difference with p-value of ", pvalue)