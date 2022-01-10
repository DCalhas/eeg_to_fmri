import argparse

import os

import numpy as np

import pickle

from utils import metrics, process_utils, train, losses_utils, viz_utils, lrp

from models.eeg_to_fmri import EEG_to_fMRI

import tensorflow as tf

from pathlib import Path

import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

parser = argparse.ArgumentParser()
parser.add_argument('mode',
					choices=['metrics', 'residues', 'mean_residues', 'quality', 'attention_graph', 'mean_attention_graph', 'lrp_eeg_channels'],
					help="What to compute")
parser.add_argument('dataset', choices=['01', '02'], help="Which dataset to load")
parser.add_argument('-topographical_attention', action="store_true", help="Verbose")
parser.add_argument('-conditional_attention_style', action="store_true", help="Verbose")
parser.add_argument('-fourier_features', action="store_true", help="Verbose")
parser.add_argument('-random_fourier', action="store_true", help="Verbose")
parser.add_argument('-epochs', default=10, type=int, help="Number of epochs")
parser.add_argument('-batch_size', default=4, type=int, help="Batch size")
parser.add_argument('-learning_rate', default=0.001, type=float, help="Learning rate")#to remove
parser.add_argument('-na_path_eeg', default=str(Path.home())+"/eeg_to_fmri/na_models_eeg", type=str, help="Neural architectures path for the EEG encoder.")
parser.add_argument('-na_path_fmri', default=str(Path.home())+"/eeg_to_fmri/na_models_fmri", type=str, help="Neural architectures path for the fMRI encoder.")
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
random_fourier=opt.random_fourier
conditional_attention_style=opt.conditional_attention_style
epochs=opt.epochs
batch_size=opt.batch_size
learning_rate=opt.learning_rate
na_path_eeg=opt.na_path_eeg
na_path_fmri=opt.na_path_fmri
gpu_mem=opt.gpu_mem
verbose=opt.verbose
save_metrics=opt.save_metrics
metrics_path=opt.metrics_path
seed=opt.seed

#assertion
setting=dataset
if(topographical_attention):
	setting+="_topographical_attention"
if(random_fourier):
	assert fourier_features, "To run random_fourier, fourier_features need to be active"
	setting+="_random"
if(fourier_features):
	setting+="_fourier_features"
if(conditional_attention_style):
	assert topographical_attention, "To run conditional_attention_style, topographical_attention needs to be active"
	setting+="_attention_style"

#set seed and configuration of memory
process_utils.process_setup_tensorflow(gpu_mem, seed=seed)

#create dir setting if not exists
if(not os.path.exists(metrics_path+"/"+ setting)):
	os.makedirs(metrics_path+"/"+ setting)


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
learning_rate=0.002980911194116198
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
with open(na_path_eeg, "rb") as f:
	na_specification_eeg = pickle.load(f)
with open(na_path_fmri, "rb") as f:
	na_specification_fmri = pickle.load(f)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model = EEG_to_fMRI(latent_dimension, eeg_shape[1:], na_specification_eeg, n_channels, weight_decay=weight_decay, skip_connections=skip_connections,
							batch_norm=batch_norm, local=local, fourier_features=fourier_features,
							random_fourier=random_fourier, conditional_attention_style=conditional_attention_style,
							topographical_attention=topographical_attention, seed=None, fmri_args = (latent_dimension, fmri_shape[1:], 
							kernel_size, stride_size, n_channels, max_pool, batch_norm, weight_decay, skip_connections,
							n_stacks, True, False, outfilter, dropout, None, False, na_specification_fmri))
model.build(eeg_shape, fmri_shape)
model.compile(optimizer=optimizer)
loss_fn = losses_utils.mae_cosine

#train model
history = train.train(train_set, model, optimizer, loss_fn, epochs=epochs, u_architecture=True, verbose=verbose)

if(mode=="metrics"):

	#create dir setting if not exists
	if(not os.path.exists(metrics_path+"/"+ setting+"/metrics")):
		os.makedirs(metrics_path+"/"+ setting+"/metrics")

	rmse_pop = metrics.rmse(test_set, model)
	ssim_pop = metrics.ssim(test_set, model)
	print("RMSE: ", np.mean(rmse_pop), "\pm", np.std(rmse_pop))
	print("SSIM: ", np.mean(ssim_pop), "\pm", np.std(ssim_pop))

	#compute p values against saved metrics
	for f in os.listdir(metrics_path):
		if("rmse" in f):
			other_pop_rmse = np.load(metrics_path+"/"+f, allow_pickle=True)
			print("p-value against", f.split("/")[-1][:-4], ttest_ind(rmse_pop, other_pop_rmse).pvalue)
		if("ssim" in f):
			other_pop_ssim = np.load(metrics_path+"/"+f, allow_pickle=True)
			print("p-value against", f.split("/")[-1][:-4], ttest_ind(ssim_pop, other_pop_ssim).pvalue)

	if(save_metrics):
		with open(metrics_path+"/"+setting+"/metrics"+"/rmse_"+"seed_"+str(seed)+".npy", 'wb') as f:
			np.save(f, rmse_pop)
		with open(metrics_path+"/"+setting+"/metrics"+"/ssim_"+"seed_"+str(seed)+".npy", 'wb') as f:
			np.save(f, ssim_pop)

elif(mode=="residues"):
	#create dir setting if not exists
	if(not os.path.exists(metrics_path+"/"+ setting+"/residues")):
		os.makedirs(metrics_path+"/"+ setting+"/residues")
	instance=0
	for eeg, fmri in test_set.repeat(1):
		viz_utils.plot_3D_representation_projected_slices(fmri.numpy()[0]-model(eeg, fmri)[0].numpy()[0],
															cmap=plt.cm.gray,
															res_img=fmri.numpy()[0],
															slice_label=False,
															save=True, save_path=metrics_path+"/"+setting+"/residues"+"/"+ str(instance)+"_instance_seed_"+str(seed)+".pdf")
		instance+=1
elif(mode=="quality"):
	#create dir setting if not exists
	if(not os.path.exists(metrics_path+"/"+ setting+"/quality")):
		os.makedirs(metrics_path+"/"+ setting+"/quality")

	instance=0
	for eeg, fmri in test_set.repeat(1):
		viz_utils.plot_3D_representation_projected_slices(model(eeg, fmri)[0].numpy()[0],
															res_img=fmri.numpy()[0],
															save=True, save_path=metrics_path+"/"+setting+"/quality"+"/"+ mode +"_" + str(instance)+"_instance.pdf")
		instance+=1
elif(mode=="mean_residues"):
	#create dir setting if not exists
	if(not os.path.exists(metrics_path+"/"+ setting+"/mean_residues")):
		os.makedirs(metrics_path+"/"+ setting+"/mean_residues")

	instance=0
	mean_fmri = tf.zeros((1,)+fmri_shape[1:])
	mean_synth_fmri = tf.zeros((1,)+fmri_shape[1:])
	for eeg, fmri in test_set.repeat(1):
		mean_fmri = mean_fmri + fmri
		mean_synth_fmri = mean_synth_fmri + model(eeg, fmri)[0]
		instance+=1
	viz_utils.plot_3D_representation_projected_slices(np.abs((mean_fmri.numpy()-mean_synth_fmri.numpy())[0]/instance),
															cmap=plt.cm.gray,
															res_img=mean_fmri.numpy()[0]/instance,
															slice_label=False,
															normalize_residues=True,
															save=True, save_path=metrics_path+"/"+setting+"/mean_residues"+"/"+"_mean_residues"+"_seed_"+str(seed)+".pdf")
	viz_utils.plot_3D_representation_projected_slices(np.abs((mean_fmri.numpy()-mean_synth_fmri.numpy())[0]/instance),
															cmap=plt.cm.gray,
															res_img=mean_fmri.numpy()[0]/instance,
															slice_label=False,
															normalize_residues=False,
															save=True, save_path=metrics_path+"/"+setting+"/mean_residues"+"/"+"_mean_normalized_residues"+"_seed_"+str(seed)+".pdf")
elif(mode=='lrp_eeg_channels'):
	#explain and then get the relevances
	if(topographical_attention):
		#create dir setting if not exists
		if(not os.path.exists(metrics_path+"/"+ setting+"/explainability")):
			os.makedirs(metrics_path+"/"+ setting+"/explainability")

		explainer = lrp.LRP_EEG(model)
		R=lrp.explain(explainer, dev_set, eeg=True, eeg_attention=True, fmri=False, verbose=True)

		#placeholder
		attention_scores = np.random.randn(len(getattr(eeg_utils, "channels_"+dataset)), 
												   len(getattr(eeg_utils, "channels_"+dataset)))

		viz_utils.plot_attention_eeg(attention_scores,
									dataset="01",
									plot_names=True,
									edge_threshold=np.percentile(attention_scores, 99.9),
									save=True, save_path=metrics_path+"/"+setting+"/explainability"+"/"+"_channels_attention_" + "seed_"+str(seed)+".pdf")
else:
	raise NotImplementedError