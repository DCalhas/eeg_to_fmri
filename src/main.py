import argparse

import os

import numpy as np

import pickle

from utils import metrics, process_utils, train, losses_utils, viz_utils, lrp, data_utils, eeg_utils, fmri_utils, bnn_utils, assertion_utils

from models.eeg_to_fmri import EEG_to_fMRI

from regularizers import path_sgd

import tensorflow as tf

from pathlib import Path

import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

parser = argparse.ArgumentParser()
parser.add_argument('mode',
					choices=['metrics', 'residues', 'uncertainty', 'mean_residues', 'quality', 'attention_graph', 'mean_attention_graph', 'lrp_eeg_channels', 'lrp_eeg_fmri'],
					help="What to compute")
parser.add_argument('dataset', choices=['01', '02', '03', '04', '05', 'NEW'], help="Which dataset to load")
parser.add_argument('-TRs', default=1, type=int, help="Number of volumes to predict")
parser.add_argument('-topographical_attention', action="store_true", help="Topographical attention on EEG channels")
parser.add_argument('-channel_organization', action="store_true", help="Organization of EEG channels, without surpressing information from any channel")
parser.add_argument('-conditional_attention_style', action="store_true", help="Conditional attention style on the latent space")
parser.add_argument('-conditional_attention_style_prior', action="store_true", help="Style prior on the latent space")
parser.add_argument('-consistency', action="store_true", help="Apply consistency learning at the sinusoids")
parser.add_argument('-padded', action="store_true", help="Fill higher resolutions with zero for the upsampling method.")
parser.add_argument('-variational', action="store_true", help="Variational implementation of the model")
parser.add_argument('-variational_coefs', default=None, type=None, help="Number of extra stochastic resolution coefficients")
parser.add_argument('-variational_dependent_h', default=None, type=int, help="Apply dependency mechanism on X to get high frequency coefficient\nDimension of the hidden boundary decision for stochastic heads")
parser.add_argument('-variational_dist', default="Normal", type=str, help="Distribution used for the high resolution coefficients")
parser.add_argument('-variational_random_padding', action="store_true", help="Whether to randomize positions in the DCT frequency space instead of predicting low resolution coefficients")
parser.add_argument('-resolution_decoder', default=None, type=float, help="Resolution decoder intermediary before final transformation in decoder -- used in uncertainty")
parser.add_argument('-aleatoric_uncertainty', action="store_true", help="Aleatoric uncertainty flag")
parser.add_argument('-fourier_features', action="store_true", help="Fourier features flag")
parser.add_argument('-random_fourier', action="store_true", help="Use random fourier features projection")
parser.add_argument('-epochs', default=10, type=int, help="Number of epochs")
parser.add_argument('-batch_size', default=4, type=int, help="Batch size")
parser.add_argument('-optimizer', default="Adam", type=str, help="Optimizer to use for the learning session")
parser.add_argument('-na_path_eeg', default=os.environ['EEG_FMRI']+"/na_models_eeg/na_specification_2", type=str, help="Neural architectures path for the EEG encoder.")
parser.add_argument('-na_path_fmri', default=os.environ['EEG_FMRI']+"/na_models_fmri/na_specification_2", type=str, help="Neural architectures path for the fMRI encoder.")
parser.add_argument('-gpu_mem', default=4000, type=int, help="GPU memory limit")
parser.add_argument('-verbose', action="store_true", help="Verbose")
parser.add_argument('-save_metrics', action="store_true", help="save metrics to compare afterwards")
parser.add_argument('-metrics_path', default=os.environ['EEG_FMRI']+"/metrics", type=str, help="Metrics save path.")
parser.add_argument('-T', default=100, type=int, help="Monte Carlo Simulation number of samples taken to approximate.")
parser.add_argument('-run_eagerly', action="store_true", help="Run eagerly, if not it runs in graph mode. This is important for activity regularization")
parser.add_argument('-seed', default=42, type=int, help="Seed for random generator")
opt = parser.parse_args()

mode, dataset, TRs, topographical_attention, channel_organization, consistency, padded, variational, variational_coefs, variational_dependent_h, variational_dist, variational_random_padding, resolution_decoder, aleatoric_uncertainty, fourier_features, random_fourier, conditional_attention_style, conditional_attention_style_prior, epochs, batch_size, optimizer, na_path_eeg, na_path_fmri, gpu_mem, verbose, save_metrics, metrics_path, T, seed, run_eagerly, setting = assertion_utils.main(opt)

#set seed and configuration of memory
process_utils.process_setup_tensorflow(gpu_mem, seed=seed, run_eagerly=run_eagerly)

#create dir setting if not exists
if(not os.path.exists(metrics_path+"/"+ setting)):
	os.makedirs(metrics_path+"/"+ setting)

#load data
interval_eeg=10
n_volumes=getattr(fmri_utils, "n_volumes_"+dataset)
n_individuals=getattr(data_utils, "n_individuals_"+dataset)
threshold_plot=getattr(data_utils, "threshold_plot_"+dataset)

#return_test returns the test set, this is not active when running validation optimization
#setup_tf sets the tensorflow memory growth on GPU, this should not be done when already set, which is the case
train_data, test_data = process_utils.load_data_eeg_fmri(dataset, n_individuals, n_volumes, interval_eeg, gpu_mem, return_test=True, setup_tf=False)

with tf.device('/CPU:0'):
	#setup shapes and data loaders
	eeg_shape, fmri_shape = (None,)+train_data[0].shape[1:], (None,)+train_data[1].shape[1:]
	train_set = tf.data.Dataset.from_tensor_slices(train_data).shuffle(1).batch(batch_size)
	test_set = tf.data.Dataset.from_tensor_slices(test_data).batch(1)

	#load model
	#unroll hyperparameters
	#learning_rate,weight_decay ,kernel_size ,stride_size ,batch_size,latent_dimension,n_channels,max_pool,batch_norm,skip_connections,dropout,n_stacks,outfilter,local = (0.002980911194116198, 0.0004396489214334123, (9, 9, 4), (1, 1, 1), 4, (7, 7, 7), 4, True, True, True, True, 3, 1, True)
	learning_rate,weight_decay ,kernel_size ,stride_size ,batch_size,latent_dimension,n_channels,max_pool,batch_norm,skip_connections,dropout,n_stacks,outfilter,local = (0.002980911194116198, 0.0, (9, 9, 4), (1, 1, 1), 4, (7, 7, 7), 4, True, True, True, True, 3, 1, True)

	with open(na_path_eeg, "rb") as f:
		na_specification_eeg = pickle.load(f)
	with open(na_path_fmri, "rb") as f:
		na_specification_fmri = pickle.load(f)

	#placeholder not pretty please correct me
	_resolution_decoder=None
	if(type(resolution_decoder) is float):
		_resolution_decoder=(int(fmri_shape[1]/resolution_decoder),int(fmri_shape[2]/resolution_decoder),int(fmri_shape[3]/resolution_decoder))

	model = EEG_to_fMRI(latent_dimension, eeg_shape[1:], na_specification_eeg, n_channels, weight_decay=weight_decay, consistency=consistency,
								batch_norm=batch_norm, local=local, fourier_features=fourier_features, random_fourier=random_fourier, 
								conditional_attention_style=conditional_attention_style, topographical_attention=topographical_attention, 
								conditional_attention_style_prior=conditional_attention_style_prior, skip_connections=skip_connections,
								organize_channels=channel_organization, inverse_DFT=variational or padded, DFT=variational or padded, 
								variational_dist=variational_dist, variational_iDFT=variational, variational_coefs=variational_coefs, 
								variational_iDFT_dependent=variational_dependent_h>1, variational_iDFT_dependent_dim=variational_dependent_h,
								aleatoric_uncertainty=aleatoric_uncertainty, low_resolution_decoder=type(resolution_decoder) is float, 
								variational_random_padding=variational_random_padding, resolution_decoder=_resolution_decoder, seed=None, 
								fmri_args = (latent_dimension, fmri_shape[1:], 
								kernel_size, stride_size, n_channels, max_pool, batch_norm, weight_decay, skip_connections,
								n_stacks, True, False, outfilter, dropout, None, False, na_specification_fmri))
	model.build(eeg_shape, fmri_shape)
	optimizer=path_sgd.optimizer(optimizer, [(1,)+eeg_shape[1:],(1,)+fmri_shape[1:]], model, learning_rate)
	model.compile(optimizer=optimizer)
	loss_fn = list(losses_utils.LOSS_FNS.values())[int(aleatoric_uncertainty)]#if variational get loss fn at index 1

#train model
train.train(train_set, model, optimizer, loss_fn, epochs=epochs, u_architecture=True, verbose=verbose, verbose_batch=verbose)

if(mode=="metrics"):
	#create dir setting if not exists
	if(not os.path.exists(metrics_path+"/"+ setting+"/metrics")):
		os.makedirs(metrics_path+"/"+ setting+"/metrics")

	res_pop = metrics.residues(test_set, model, variational=aleatoric_uncertainty, T=T)
	rmse_pop = metrics.rmse(test_set, model, variational=aleatoric_uncertainty, T=T)
	ssim_pop = metrics.ssim(test_set, model, variational=aleatoric_uncertainty, T=T)
	if(aleatoric_uncertainty):
		sharpness = metrics.sharpness(test_set, model)
	print("RMSE: ", np.mean(rmse_pop), "\pm", np.std(rmse_pop))
	print("SSIM: ", np.mean(ssim_pop), "\pm", np.std(ssim_pop))
	if(aleatoric_uncertainty):
		print("SHARPNESS: ", np.mean(sharpness), "\pm", np.std(sharpness))
	#compute p values against saved metrics
	for f in os.listdir(metrics_path):
		if("rmse" in f):
			other_pop_rmse = np.load(metrics_path+"/"+f, allow_pickle=True)
			print("p-value against", f.split("/")[-1][:-4], ttest_ind(rmse_pop, other_pop_rmse).pvalue)
		if("ssim" in f):
			other_pop_ssim = np.load(metrics_path+"/"+f, allow_pickle=True)
			print("p-value against", f.split("/")[-1][:-4], ttest_ind(ssim_pop, other_pop_ssim).pvalue)
	if(save_metrics):
		with open(metrics_path+"/"+setting+"/metrics"+"/res_"+"seed_"+str(seed)+".npy", 'wb') as f:
			np.save(f, res_pop)
		with open(metrics_path+"/"+setting+"/metrics"+"/rmse_"+"seed_"+str(seed)+".npy", 'wb') as f:
			np.save(f, rmse_pop)
		with open(metrics_path+"/"+setting+"/metrics"+"/ssim_"+"seed_"+str(seed)+".npy", 'wb') as f:
			np.save(f, ssim_pop)
elif(mode=="uncertainty"):
	#create dir setting if not exists
	if(not os.path.exists(metrics_path+"/"+ setting+"/uncertainty")):
		os.makedirs(metrics_path+"/"+ setting+"/uncertainty")
	if(not os.path.exists(metrics_path+"/"+ setting+"/uncertainty/epistemic")):
		os.makedirs(metrics_path+"/"+ setting+"/uncertainty/epistemic")
	if(not os.path.exists(metrics_path+"/"+ setting+"/uncertainty/aleatoric")):
		os.makedirs(metrics_path+"/"+ setting+"/uncertainty/aleatoric")
	if(not os.path.exists(metrics_path+"/"+ setting+"/uncertainty/quality")):
		os.makedirs(metrics_path+"/"+ setting+"/uncertainty/quality")
	if(not os.path.exists(metrics_path+"/"+ setting+"/uncertainty/quality/single/")):
		os.makedirs(metrics_path+"/"+ setting+"/uncertainty/quality/single")
	if(not os.path.exists(metrics_path+"/"+ setting+"/uncertainty/quality/whole/")):
		os.makedirs(metrics_path+"/"+ setting+"/uncertainty/quality/whole")
	
	instance=0
	for eeg, fmri in test_set.repeat(1):
		ims = (fmri.numpy(), bnn_utils.predict_MC(model, (eeg, fmri), T=T).numpy(), bnn_utils.epistemic_uncertainty(model, (eeg, fmri), T=T).numpy(), model(eeg, fmri)[0][1].numpy())
		viz_utils.single_display_gt_pred_espistemic_aleatoric(*ims, name=["DenseFlipout", "DCTVariational"][int(variational and aleatoric_uncertainty)], save=True, save_path=metrics_path+"/"+setting+"/uncertainty/quality/single"+"/" + str(instance)+"_instance.pdf", save_format="pdf")
		viz_utils.whole_display_gt_pred_espistemic_aleatoric(*ims, save=True, save_path=metrics_path+"/"+setting+"/uncertainty/quality/whole"+"/" + str(instance)+"_instance.pdf", save_format="pdf")
		instance+=1
elif(mode=="residues"):
	#create dir setting if not exists
	if(not os.path.exists(metrics_path+"/"+ setting+"/residues")):
		os.makedirs(metrics_path+"/"+ setting+"/residues")
	instance=0
	for eeg, fmri in test_set.repeat(1):
		viz_utils.plot_3D_representation_projected_slices(fmri.numpy()[0]-model(eeg, fmri)[0].numpy()[0], threshold=threshold_plot,cmap=plt.cm.gray,res_img=fmri.numpy()[0],slice_label=False,save=True, save_path=metrics_path+"/"+setting+"/residues"+"/"+ str(instance)+"_instance_seed_"+str(seed)+".pdf")
		instance+=1
elif(mode=="quality"):
	#create dir setting if not exists
	if(not os.path.exists(metrics_path+"/"+ setting+"/quality")):
		os.makedirs(metrics_path+"/"+ setting+"/quality")

	instance=0
	for eeg, fmri in test_set.repeat(1):
		viz_utils.plot_3D_representation_projected_slices(model(eeg, fmri)[0].numpy()[0], threshold=threshold_plot, res_img=fmri.numpy()[0], save=True, save_path=metrics_path+"/"+setting+"/quality"+"/" + str(instance)+"_instance.pdf")
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
	viz_utils.plot_3D_representation_projected_slices(np.abs((mean_fmri.numpy()-mean_synth_fmri.numpy())[0]/instance), threshold=threshold_plot,cmap=plt.cm.gray,res_img=mean_fmri.numpy()[0]/instance,slice_label=False,normalize_residues=True,save=True, save_path=metrics_path+"/"+setting+"/mean_residues"+"/"+"_mean_residues"+"_seed_"+str(seed)+".pdf")
	viz_utils.plot_3D_representation_projected_slices(np.abs((mean_fmri.numpy()-mean_synth_fmri.numpy())[0]/instance), threshold=threshold_plot,cmap=plt.cm.gray,res_img=mean_fmri.numpy()[0]/instance,slice_label=False,normalize_residues=False,save=True, save_path=metrics_path+"/"+setting+"/mean_residues"+"/"+"_mean_normalized_residues"+"_seed_"+str(seed)+".pdf")
elif(mode=='lrp_eeg_channels'):
	#explain and then get the relevances
	if(topographical_attention):
		#create dir setting if not exists
		if(not os.path.exists(metrics_path+"/"+ setting+"/explainability")):
			os.makedirs(metrics_path+"/"+ setting+"/explainability")

		explainer = lrp.LRP_EEG(model.decoder, conditional_attention_style=conditional_attention_style)
		attention_scores=lrp.explain(explainer, test_set, eeg=True, eeg_attention=True, fmri=False, verbose=True)

		for percentile in [98, 99, 99.5, 99.7, 99.9]:
			viz_utils.plot_attention_eeg(np.mean(attention_scores, axis=0),dataset=dataset,plot_names=True,edge_threshold=np.percentile(attention_scores, percentile),save=True, save_path=metrics_path+"/"+setting+"/explainability"+"/"+str(percentile)+"_channels_attention_" + "seed_"+str(seed)+".pdf")
elif(mode=='lrp_eeg_fmri'):
	#create dir setting if not exists
	if(not os.path.exists(metrics_path+"/"+ setting+"/explainability")):
		os.makedirs(metrics_path+"/"+ setting+"/explainability")

	#explain eeg
	explainer = lrp.LRP_EEG(model.decoder)
	R=lrp.explain(explainer, test_set, eeg=True, fmri=False, verbose=True)

	viz_utils.R_channels(R, test_data[0], ch_names=eeg_utils.channels_01, save=True, save_path=metrics_path+"/"+setting+"/explainability"+"/eeg_channels_" + "seed_"+str(seed)+".pdf")
	viz_utils.R_analysis_channels(R, test_data[0].shape[1], ch_names=getattr(eeg_utils, "channels_"+dataset), save=True, save_path=metrics_path+"/"+setting+"/explainability"+"/eeg_channels_relevance_" + "seed_"+str(seed)+".pdf")
	viz_utils.R_analysis_freqs(R, test_data[0].shape[2], save=True, save_path=metrics_path+"/"+setting+"/explainability"+"/eeg_freq_relevance_" + "seed_"+str(seed)+".pdf")
	viz_utils.R_analysis_times(R, test_data[0].shape[3], save=True, save_path=metrics_path+"/"+setting+"/explainability"+"/eeg_time_relevance_" + "seed_"+str(seed)+".pdf")
	viz_utils.R_analysis_dimensions(R, ch_names=getattr(eeg_utils, "channels_"+dataset), save=True, save_path=metrics_path+"/"+setting+"/explainability"+"/eeg_relevance_report_" + "seed_"+str(seed)+".pdf")
	viz_utils.R_analysis_times_freqs(R, R.shape[3], R.shape[2], func=metrics.ttest_1samp_r, save=True, save_path=metrics_path+"/"+setting+"/explainability"+"/eeg_relevance_time_freq_" + "seed_"+str(seed)+".pdf")
	viz_utils.R_analysis_channels_freqs(R, R.shape[1], R.shape[2], func=metrics.ttest_1samp_r, ch_names=getattr(eeg_utils, "channels_"+dataset), save=True, save_path=metrics_path+"/"+setting+"/explainability"+"/eeg_relevance_channel_freq_" + "seed_"+str(seed)+".pdf")
	viz_utils.R_analysis_times_channels(R, R.shape[3], R.shape[1], func=metrics.ttest_1samp_r, ch_names=getattr(eeg_utils, "channels_"+dataset), save=True, save_path=metrics_path+"/"+setting+"/explainability"+"/eeg_relevance_time_channel_" + "seed_"+str(seed)+".pdf")

	#explain fmri
	explainer = lrp.LRP(model.fmri_encoder)
	R=lrp.explain(explainer, test_set, eeg=False, fmri=True, verbose=True)

	fig = viz_utils.plot_3D_representation_projected_slices(np.mean(R, axis=0),res_img=np.mean(test_data[1],axis=0),slice_label=False,uncertainty=True,cmap=plt.cm.Blues,legend_colorbar=r"$\mu[R]$",max_min_legend=["",""], save=True, save_path=metrics_path+"/"+setting+"/explainability"+"/fmri_mean_R_" + "seed_"+str(seed)+".pdf", threshold=threshold_plot)
	fig = viz_utils.plot_3D_representation_projected_slices(np.std(R, axis=0),res_img=np.mean(test_data[1],axis=0),slice_label=False,uncertainty=True,cmap=plt.cm.Blues,legend_colorbar=r"$Var[R]$",max_min_legend=["",""], save=True, save_path=metrics_path+"/"+setting+"/explainability"+"/fmri_Var_R_" + "seed_"+str(seed)+".pdf", threshold=threshold_plot)
	fig = viz_utils.plot_3D_representation_projected_slices(np.amax(R, axis=0),res_img=np.mean(test_data[1],axis=0),slice_label=False,uncertainty=True,cmap=plt.cm.Blues,legend_colorbar=r"$max(R)$",max_min_legend=["Non Relevant","Relevant"],save=True, save_path=metrics_path+"/"+setting+"/explainability"+"/fmri_max_R_" + "seed_"+str(seed)+".pdf", threshold=threshold_plot)
	fig = viz_utils.plot_3D_representation_projected_slices(np.amin(R, axis=0),res_img=np.mean(test_data[1],axis=0),slice_label=False,uncertainty=True,cmap=plt.cm.Blues_r,legend_colorbar=r"$min(R)$",max_min_legend=["Neg Relevant","Non Relevant"],save=True, save_path=metrics_path+"/"+setting+"/explainability"+"/fmri_min_R_" + "seed_"+str(seed)+".pdf", threshold=threshold_plot)
	fig = viz_utils.plot_3D_representation_projected_slices(metrics.ttest_1samp_r(R, np.mean(R), axis=0),res_img=np.mean(test_data[1],axis=0),slice_label=False,uncertainty=True, cmap=plt.cm.Blues, legend_colorbar=r"$p-value$", max_min_legend=[r"$p=1.0$",r"$p=0.0$"], save=True, save_path=metrics_path+"/"+setting+"/explainability"+"/fmri_pvalues_R_" + "seed_"+str(seed)+".pdf", threshold=threshold_plot)
else:
	raise NotImplementedError