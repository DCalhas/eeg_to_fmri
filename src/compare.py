#python compare.py 01 -name1 (i) -name2 (ii) -fourier_features1 -random_fourier1 -epochs 10 -na_path_eeg /home/david/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/david/eeg_to_fmri/na_models_fmri/na_specification_2 -seed 5 -gpu_mem 1000 -verbose

import argparse

import os

import numpy as np

import pickle

from utils import process_utils, train, losses_utils, viz_utils

from models.eeg_to_fmri import EEG_to_fMRI

import tensorflow as tf

from pathlib import Path

import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['01', '02'], help="Which dataset to load")
parser.add_argument('-topographical_attention1', action="store_true", help="Verbose")
parser.add_argument('-topographical_attention2', action="store_true", help="Verbose")
parser.add_argument('-conditional_attention_style1', action="store_true", help="Verbose")
parser.add_argument('-conditional_attention_style2', action="store_true", help="Verbose")
parser.add_argument('-fourier_features1', action="store_true", help="Verbose")
parser.add_argument('-fourier_features2', action="store_true", help="Verbose")
parser.add_argument('-random_fourier1', action="store_true", help="Verbose")
parser.add_argument('-random_fourier2', action="store_true", help="Verbose")
parser.add_argument('-epochs', default=10, type=int, help="Number of epochs")
parser.add_argument('-name1', default=None, type=str, help="Name to plot for the model1")
parser.add_argument('-name2', default=None, type=str, help="Name to plot for the model2")
parser.add_argument('-na_path_eeg', default=str(Path.home())+"/eeg_to_fmri/na_models_eeg", type=str, help="Neural architectures path for the EEG encoder.")
parser.add_argument('-na_path_fmri', default=str(Path.home())+"/eeg_to_fmri/na_models_fmri", type=str, help="Neural architectures path for the fMRI encoder.")
parser.add_argument('-metrics_path', default=str(Path.home())+"/eeg_to_fmri/metrics", type=str, help="Metrics save path.")
parser.add_argument('-gpu_mem', default=4000, type=int, help="GPU memory limit")
parser.add_argument('-verbose', action="store_true", help="Verbose")
parser.add_argument('-seed', default=42, type=int, help="Seed for random generator")
opt = parser.parse_args()

dataset=opt.dataset
topographical_attention1=opt.topographical_attention1
topographical_attention2=opt.topographical_attention2
fourier_features1=opt.fourier_features1
fourier_features2=opt.fourier_features2
random_fourier1=opt.random_fourier1
random_fourier2=opt.random_fourier2
conditional_attention_style1=opt.conditional_attention_style1
conditional_attention_style2=opt.conditional_attention_style2
epochs=opt.epochs
name1=opt.name1
name2=opt.name2
na_path_eeg=opt.na_path_eeg
na_path_fmri=opt.na_path_fmri
metrics_path=opt.metrics_path
gpu_mem=opt.gpu_mem
verbose=opt.verbose
seed=opt.seed

if(name1 is None):
	#assertion
	name1=""
	if(topographical_attention1):
		name1+="_topo"
	if(random_fourier1):
		assert fourier_features1, "To run random_fourier, fourier_features need to be active"
		name1+="_random"
	if(fourier_features1):
		name1+="_fourier"
	if(conditional_attention_style1):
		assert topographical_attention1, "To run conditional_attention_style, topographical_attention needs to be active"
		name1+="_style"
	name1=name1[1:]
if(name2 is None):
	name2=""
	if(topographical_attention2):
		name2+="_topo"
	if(random_fourier2):
		assert fourier_features2, "To run random_fourier, fourier_features need to be active"
		name2+="_random"
	if(fourier_features2):
		name2+="_fourier"
	if(conditional_attention_style2):
		assert topographical_attention2, "To run conditional_attention_style, topographical_attention needs to be active"
		name2+="_style"
	name2=name2[1:]

#set seed and configuration of memory
process_utils.process_setup_tensorflow(gpu_mem, seed=None)

#load data
raw_eeg=False#time or frequency features? raw-time nonraw-frequency
resampling=False
if(dataset=="01"):
	n_volumes=300-3
	n_individuals=10
	n_individuals_train=8
if(dataset=="02"):
	n_volumes=170-3
#parametrize the interval eeg?
interval_eeg=10

#return_test returns the test set, this is not active when running validation optimization
#setup_tf sets the tensorflow memory growth on GPU, this should not be done when already set, which is the case
train_data, test_data = process_utils.load_data_eeg_fmri(dataset, n_individuals, n_volumes, interval_eeg, gpu_mem, return_test=True, setup_tf=False)
#setup shapes and data loaders
eeg_shape, fmri_shape = (None,)+train_data[0].shape[1:], (None,)+train_data[1].shape[1:]
test_set = tf.data.Dataset.from_tensor_slices(test_data).batch(1)

#load model
#unroll hyperparameters
theta1=(0.002980911194116198,0.0004396489214334123,(9,9,4),(1,1,1),4,(7,7,7),4,True,True,True,True,3,1)
learning_rate1=0.002980911194116198
weight_decay1=float(theta1[1])
kernel_size1=theta1[2]
stride_size1=theta1[3]
batch_size1=int(theta1[4])
latent_dimension1=theta1[5]
n_channels1=int(theta1[6])
max_pool1=bool(theta1[7])
batch_norm1=bool(theta1[8])
skip_connections1=bool(theta1[9])
dropout1=bool(theta1[10])
n_stacks1=int(theta1[11])
outfilter1=int(theta1[12])
local1=True
#for second model
theta2=(0.002980911194116198,0.0004396489214334123,(9,9,4),(1,1,1),4,(7,7,7),4,True,True,True,True,3,1)
learning_rate2=0.002980911194116198
weight_decay2=float(theta2[1])
kernel_size2=theta2[2]
stride_size2=theta2[3]
batch_size2=int(theta2[4])
latent_dimension2=theta2[5]
n_channels2=int(theta2[6])
max_pool2=bool(theta2[7])
batch_norm2=bool(theta2[8])
skip_connections2=bool(theta2[9])
dropout2=bool(theta2[10])
n_stacks2=int(theta2[11])
outfilter2=int(theta2[12])
local2=True

with open(na_path_eeg, "rb") as f:
	na_specification_eeg = pickle.load(f)
with open(na_path_fmri, "rb") as f:
	na_specification_fmri = pickle.load(f)

#set seed and configuration of memory
process_utils.set_seed(seed=seed)

train_set = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size1)
optimizer1 = tf.keras.optimizers.Adam(learning_rate=learning_rate1)
model1 = EEG_to_fMRI(latent_dimension1, eeg_shape[1:], na_specification_eeg, n_channels1, weight_decay=weight_decay1, skip_connections=skip_connections1,
							batch_norm=batch_norm1, local=local1, fourier_features=fourier_features1,
							random_fourier=random_fourier1, conditional_attention_style=conditional_attention_style1,
							topographical_attention=topographical_attention1, seed=None, fmri_args = (latent_dimension1, fmri_shape[1:], 
							kernel_size1, stride_size1, n_channels1, max_pool1, batch_norm1, weight_decay1, skip_connections1,
							n_stacks1, True, False, outfilter1, dropout1, None, False, na_specification_fmri))
model1.build(eeg_shape, fmri_shape)
model1.compile(optimizer=optimizer1)
loss_fn = losses_utils.mae_cosine

#train model
train.train(train_set, model1, optimizer1, loss_fn, epochs=epochs, u_architecture=True, verbose=verbose)

#set seed and configuration of memory
process_utils.set_seed(seed=seed)

train_set = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size2)
optimizer2 = tf.keras.optimizers.Adam(learning_rate=learning_rate2)
model2 = EEG_to_fMRI(latent_dimension2, eeg_shape[1:], na_specification_eeg, n_channels2, weight_decay=weight_decay2, skip_connections=skip_connections2,
							batch_norm=batch_norm2, local=local2, fourier_features=fourier_features2,
							random_fourier=random_fourier2, conditional_attention_style=conditional_attention_style2,
							topographical_attention=topographical_attention2, seed=None, fmri_args = (latent_dimension2, fmri_shape[2:], 
							kernel_size2, stride_size2, n_channels2, max_pool2, batch_norm2, weight_decay2, skip_connections2,
							n_stacks2, True, False, outfilter2, dropout2, None, False, na_specification_fmri))
model2.build(eeg_shape, fmri_shape)
model2.compile(optimizer=optimizer2)
loss_fn = losses_utils.mae_cosine

#train model
train.train(train_set, model2, optimizer2, loss_fn, epochs=epochs, u_architecture=True, verbose=verbose)

res1=np.empty((0,)+fmri_shape[1:])
res2=np.empty((0,)+fmri_shape[1:])
for eeg, fmri in test_set.repeat(1):
	res1=np.append(res1, (fmri-model1(eeg,fmri)[0]).numpy(), axis=0)
	res2=np.append(res2, (fmri-model2(eeg,fmri)[0]).numpy(), axis=0)
pvalues=ttest_ind(res1, res2, axis=0).pvalue

#create dir setting if not exists
if(not os.path.exists(metrics_path+"/comparison")):
	os.makedirs(metrics_path+"/comparison")


fig=viz_utils.comparison_plot_3D_representation_projected_slices(np.mean(res1,axis=0), np.mean(res2, axis=0), pvalues, 
																	np.mean(test_data[1], axis=0),
																	model1=name1, model2=name2,
																	save=True, save_path=metrics_path+"/comparison/"+dataset+"_"+name1+"_vs_"+name2+"_seed_"+str(seed)+".pdf")