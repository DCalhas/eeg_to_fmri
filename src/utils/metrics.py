import numpy as np

import tensorflow as tf



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
		n_instances+=1

	return _rmse