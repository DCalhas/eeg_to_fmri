


def main(opt):
	"""
	return the arguments in the opt variable,

	first it performs the according assertions needed for the script to execute

	"""
	mode=opt.mode
	dataset=opt.dataset
	topographical_attention=opt.topographical_attention
	padded=opt.padded
	variational=opt.variational
	variational_coefs=opt.variational_coefs
	variational_dependent_h=opt.variational_dependent_h
	variational_dist=opt.variational_dist
	variational_random_padding=opt.variational_random_padding
	resolution_decoder=opt.resolution_decoder
	aleatoric_uncertainty=opt.aleatoric_uncertainty
	fourier_features=opt.fourier_features
	random_fourier=opt.random_fourier
	conditional_attention_style=opt.conditional_attention_style
	epochs=opt.epochs
	batch_size=opt.batch_size
	na_path_eeg=opt.na_path_eeg
	na_path_fmri=opt.na_path_fmri
	gpu_mem=opt.gpu_mem
	verbose=opt.verbose
	save_metrics=opt.save_metrics
	metrics_path=opt.metrics_path
	T=opt.T
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
	if(padded):
		assert not variational, "No variational model along with padded version of filling with zeros"
		assert type(resolution_decoder) is float, "There needs to be a specification of the lower resolution"
		setting+="_padded"
	if(variational):
		assert variational_coefs, "Need to be specified number of coefs, always upsampling for now, set issue to allow better implementation"
		setting+="_variational"
	if(type(variational_dist) is str):
		assert variational_dist in ["Normal", "VonMises"]
		setting+="_"+variational_dist
	if(variational_dependent_h is None):
		variational_dependent_h=1
	if(variational_dependent_h > 1 and variational):
		setting+="_dependent_h_"+str(variational_dependent_h)
	if(type(variational_coefs) is str):
		assert variational, "Only done with variational flag set to True"
		variational_coefs=tuple(map(int ,variational_coefs.split(",")))
		assert len(variational_coefs) == 3, "Needs to specify all dimensions"
		assert type(variational_coefs[0]) is int and type(variational_coefs[1]) is int and type(variational_coefs[2]) is int, "Integers"
		assert variational_coefs[0] > 0 and variational_coefs[1] > 0 and variational_coefs[2] > 0, "Positive integers"
		setting+="_"+str(variational_coefs[0])+"x"+str(variational_coefs[1])+"x"+str(variational_coefs[2])
	if(type(resolution_decoder) is float):
		assert resolution_decoder > 1, "Resolution decoder needs to be \in [1,+\infty]"
		setting+="_res_"+"{:.1f}".format(resolution_decoder)
	if(variational_random_padding):
		assert variational, "Only done with variational flag set to True"
		setting+="_random_padding"



	return mode, dataset, topographical_attention, padded, variational, variational_coefs, variational_dependent_h, variational_dist, variational_random_padding, resolution_decoder, aleatoric_uncertainty, fourier_features, random_fourier, conditional_attention_style, epochs, batch_size, na_path_eeg, na_path_fmri, gpu_mem, verbose, save_metrics, metrics_path, T, seed, setting