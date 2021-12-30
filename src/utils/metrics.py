import numpy as np

import tensorflow as tf

from scipy.stats import ttest_1samp


"""
ssim:
	Inputs:
		* data - tf.data.DataLoader
		* model - tf.keras.Model
		* factor - int default factor=3
	Outputs:
		* SSIM values - list
		* SSIM mean - float
"""
def ssim(data, model, factor=3):
	_ssim = []

	for instance_x, instance_y in data.repeat(1):
		y_pred = model(instance_x, instance_y)[0]
		#error = tf.image.ssim(instance_y, y_pred, max_val=np.amax([np.amax(y_pred.numpy()), np.amax(instance_y.numpy())]))

		ssim_img = 0.0

		for axis in range((instance_y[:,:,:,:].shape[3])//factor):
			im1 = instance_y[:,:,:,axis*factor,:]
			im2 = y_pred[:,:,:,axis*factor,:]
			max_val = np.amax([np.amax(im1.numpy()), np.amax(im2.numpy())])

			ssim_img+=tf.image.ssim(im1, im2, max_val=max_val).numpy()[0]

		_ssim += [ssim_img/((instance_y[:,:,:,:].shape[3])//factor)]

	return _ssim

"""
rmse:
	Inputs:
		* data - tf.data.DataLoader
		* model - tf.keras.Model
	Outputs:
		* RMSE values - list
		* RMSE mean - float
"""
def rmse(data, model):
	_rmse = []

	for instance_x, instance_y in data.repeat(1):
		y_pred = model(instance_x, instance_y)[0]

		_rmse += [(tf.reduce_mean((y_pred-instance_y)**2))**(1/2)]

	return _rmse


"""
fid:
	Inputs:
		* data - tf.data.DataLoader
		* model - tf.keras.Model
	Outputs:
		* fid values - list
		* fid mean - float

Stands for Frechet inception distance
"""
def fid(data, model):
	mu_truth = None
	sigma_truth = None
	mu_pred = None
	sigma_pred = None

	instances = 0
	for instance_x, instance_y in data.repeat(1):
		y_pred,_,latent_ground_truth = model(instance_x, instance_y)
		latent_pred = model(instance_x, y_pred)[-1]

		if(mu_truth is None and mu_pred is None):
			mu_truth = latent_ground_truth
			mu_pred = latent_pred
		else:
			mu_truth = mu_truth + latent_ground_truth
			mu_pred = mu_pred + latent_pred

		instances+=1

	mu_truth = mu_truth/instances
	mu_pred = mu_pred/instances

	for instance_x, instance_y in data.repeat(1):
		y_pred,_,latent_ground_truth = model(instance_x, instance_y)
		latent_pred = model(instance_x, y_pred)[-1]

		if(sigma_truth is None and sigma_pred is None):
			sigma_truth = (mu_truth-latent_ground_truth)**2
			sigma_pred = (mu_pred-latent_pred)**2
		else:
			sigma_truth = sigma_truth + (mu_truth-latent_ground_truth)**2
			sigma_pred = sigma_pred + (mu_pred-latent_pred)**2

	sigma_truth = tf.reshape(sigma_truth/(instances-1), (sigma_truth.shape[1]*sigma_truth.shape[2], sigma_truth.shape[3]))
	sigma_pred = tf.reshape(sigma_pred/(instances-1), (sigma_pred.shape[1]*sigma_pred.shape[2], sigma_pred.shape[3]))

	return (tf.norm(mu_truth-mu_pred, ord=2)+tf.linalg.trace(sigma_truth+sigma_pred-2*(sigma_pred*sigma_truth)**(1/2))).numpy()



"""
"""
def ttest_1samp_r(a, m, axis=0, **kwargs):

	return 1-ttest_1samp(a, m, axis=axis, **kwargs).pvalue