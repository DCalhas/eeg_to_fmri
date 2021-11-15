import argparse

import os

import numpy as np

import pickle

from utils import metrics, process_utils, train, losses_utils, viz_utils

from models.eeg_to_fmri import EEG_to_fMRI

import tensorflow as tf

from pathlib import Path

from scipy.stats import ttest_ind

parser = argparse.ArgumentParser()
parser.add_argument('mode',
					choices=['metrics', 'quality'],
					help="What to compute")
parser.add_argument('dataset', choices=['01', '02'], help="Which dataset to load")
parser.add_argument('-topographical_attention', action="store_true", help="Verbose")
parser.add_argument('-fourier_features', action="store_true", help="Verbose")
parser.add_argument('-epochs', default=10, type=int, help="Number of epochs")
parser.add_argument('-batch_size', default=4, type=int, help="Batch size")
parser.add_argument('-learning_rate', default=0.001, type=float, help="Learning rate")#to remove
parser.add_argument('-na_path', default=str(Path.home())+"/eeg_to_fmri/na_models", type=str, help="Neural architectures path.")
parser.add_argument('-gpu_mem', default=4000, type=int, help="GPU memory limit")
parser.add_argument('-verbose', action="store_true", help="Verbose")
parser.add_argument('-save_metrics', action="store_true", help="save metrics to compare afterwards")
parser.add_argument('-metrics_path', default=str(Path.home())+"/eeg_to_fmri/metrics", type=str, help="Metrics save path.")
parser.add_argument('-seed', default=42, type=int, help="Seed for random generator")
opt = parser.parse_args()

mode=opt.mode
dataset=opt.dataset
topographical_attention=opt.topographical_attention
fourier_features=opt.fourier_features
epochs=opt.epochs
batch_size=opt.batch_size
learning_rate=opt.learning_rate
na_path=opt.na_path
gpu_mem=opt.gpu_mem
verbose=opt.verbose
save_metrics=opt.save_metrics
metrics_path=opt.metrics_path
seed=opt.seed

setting=dataset
if(topographical_attention):
	setting+="_topographical_attention"
if(fourier_features):
	setting+="_fourier_features"

#set seed and configuration of memory
process_utils.process_setup_tensorflow(gpu_mem, seed=seed)

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
train_set = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)
test_set = tf.data.Dataset.from_tensor_slices(test_data).batch(1)

#load model
#unroll hyperparameters
theta = (0.002980911194116198, 0.0004396489214334123, (9, 9, 4), (1, 1, 1), 4, (7, 7, 7), 4, True, True, True, True, 3, 1)
learning_rate=float(theta[0])
weight_decay = float(theta[1])
kernel_size = theta[2]
stride_size = theta[3]
batch_size=int(theta[4])
latent_dimension=theta[5]
n_channels=int(theta[6])
max_pool=bool(theta[7])
batch_norm=bool(theta[8])
skip_connections=bool(theta[9])
dropout=bool(theta[10])
n_stacks=int(theta[11])
outfilter=int(theta[12])
local=True
with open(na_path, "rb") as f:
	na_specification = pickle.load(f)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model = EEG_to_fMRI(latent_dimension, eeg_shape[1:], na_specification, 4, weight_decay=0.000, skip_connections=True,
							batch_norm=True, local=True, fourier_features=fourier_features, 
							topographical_attention=topographical_attention, seed=None, fmri_args = (latent_dimension, fmri_shape[1:], 
							kernel_size, stride_size, n_channels, max_pool, batch_norm, weight_decay, skip_connections,
							n_stacks, True, False, outfilter, dropout))
model.build(eeg_shape, fmri_shape)
model.compile(optimizer=optimizer)
loss_fn = losses_utils.mse_cosine

#train model
history = train.train(train_set, model, optimizer, loss_fn, epochs=epochs, u_architecture=True, verbose=verbose)

if(mode=="metrics"):
	rmse_pop = metrics.rmse(test_set, model)
	ssim_pop = metrics.ssim(test_set, model)
	print("RMSE: ", np.mean(rmse_pop), "\pm", np.std(rmse_pop))
	print("SSIM: ", np.mean(ssim_pop), "\pm", np.std(ssim_pop))

	#compute p values against saved metrics
	for f in os.listdir(metrics_path):
		if("rmse" in f):
			other_pop_rmse = np.load(f, allow_pickle=True)
			print("p-value against", f.split("/")[-1][:-4], ttest_ind(rmse_pop, other_pop_rmse).pvalue)
		if("ssim" in f):
			other_pop_ssim = np.load(f, allow_pickle=True)
			print("p-value against", f.split("/")[-1][:-4], ttest_ind(ssim_pop, other_pop_ssim).pvalue)

	if(save_metrics):
		with open(metrics_path+"/rmse"+setting+".npy", 'wb') as f:
			np.save(f, rmse_pop)
		with open(metrics_path+"/ssim"+setting+".npy", 'wb') as f:
			np.save(f, ssim_pop)

elif(mode=="residues"):
	for eeg, fmri in test_set.repeat(1):
		viz_utils.plot_3D_representation_projected_slices(fmri_train[1]-np.random.normal(0,1,size=fmri_train[1].shape),
															cmap=plt.cm.gray,
															res_img=fmri,
															slice_label=False)