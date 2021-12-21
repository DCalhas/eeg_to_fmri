import tensorflow as tf

import numpy as np


def explain(explainer, dataset, eeg=True, fmri=False, verbose=False):
	R = None

	if(eeg and not fmri):
		index=0
	elif(not eeg and fmri):
		index=1

	instance = 1
	for X in dataset.repeat(1):
		if(R is None):
			R = explainer(X[index]).numpy()
		else:
			R = np.append(R, explainer(X[index]).numpy(), axis=0)
		
		if(verbose):
			print("Explaining instance", str(instance), end="\r")
		instance+=1

	return R

"""
lrp - Layer-wise  Relevance Propagation
	Inputs:
		* x - tf.Tensor input of layer
		* y - tf.Tensor output of layer
		* layer - tf.keras.layers.Layer layer
	Outputs:
		* tf.Tensor - containing relevances
"""
def lrp(x, y, layer):
	if(type(layer) is tf.keras.layers.Reshape or type(layer) is tf.keras.layers.BatchNormalization):
		return tf.reshape(y, x.shape)

	with tf.GradientTape(watch_accessed_variables=False) as tape:
		if(type(x) is list):
			with tf.GradientTape(watch_accessed_variables=False) as tape1:
				tape.watch(x[0])
				tape1.watch(x[1])

				z = layer(x)+1e-9
				s = y/tf.reshape(z, y.shape)
				s = tf.reshape(s, z.shape)


				R = x[0]*tape.gradient(tf.reduce_sum(z*s.numpy()), x[0]) + x[1]*tape1.gradient(tf.reduce_sum(z*s.numpy()), x[1])
		else:
			tape.watch(x)
			z = layer(x)+1e-9
			s = y/tf.reshape(z, y.shape)
			s = tf.reshape(s, z.shape)
			
			c = tape.gradient(tf.reduce_sum(z*s.numpy()), x)
			R = x*c

	return R


"""
LRP_EEG: propagates relevances through a model of type models.eeg_to_fmri.EEG_to_fMRI
"""
class LRP_EEG(tf.keras.layers.Layer):
	"""
		Inputs:
			* model: models.eeg_to_fmri.EEG_to_fMRI
	"""
	def __init__(self, model):
		super(LRP_EEG, self).__init__()
		
		self.model = model
		
	"""
		Inputs:
			* X: list(tf.Tensor)
		Outputs:
			* tf.Tensor - output of model
	"""
	def forward(self, X):

		self.encoder_activations = []
		self.decoder_activations = []

		z = X
		#forward pass
		for layer in self.model.eeg_encoder.layers:
			z = layer(z)
			self.encoder_activations += [z]
		for layer in self.model.decoder.layers:
			z = layer(z)
			self.decoder_activations += [z]
		
		return z
	
	"""
		Inputs:
			* X - tf.Tensor
			* R - tf.Tensor
			* model - eeg_to_fmri.EEG_to_fMRI
			* activations - list
		Outputs: 
			* tf.Tensor
	"""
	def propagate(self, X, R, model, activations):
		for layer in range(len(model.layers))[::-1]:
			if(hasattr(model.layers[layer], "lrp")):
				R = model.layers[layer].lrp(activations[layer-1], R)
			else:
				if(layer-1 >= 0):
					R = lrp(activations[layer-1], R, model.layers[layer])
				else:
					R = lrp(X, R, model.layers[layer])
		
		return R
			
	"""
		Inputs:
			* X - tf.Tensor
			* R - tf.Tensor
		Outputs: 
			* tf.Tensor
	"""
	def backward(self, X, R):
		
		R = self.propagate(self.encoder_activations[-1], R, self.model.decoder, self.decoder_activations)
		
		return self.propagate(X, R, self.model.eeg_encoder, self.encoder_activations)

	"""
		Inptus
			* X - tf.Tensor
		Outputs: 
			* tf.Tensor
	"""
	def call(self, X):
		
		y = self.forward(X)
		
		return self.backward(X, y)

	
class LRP(tf.keras.layers.Layer):
	"""
		Inputs:
			* model: tf.keras.Model
	"""
	def __init__(self, model):
		super(LRP, self).__init__()
		
		self.model = model
		
	"""
		Inputs:
			* X: tf.Tensor
		Outputs:
			* tf.Tensor - output of model
	"""
	def forward(self, X):

		self.activations = []

		z = X
		#forward pass
		for layer in self.model.layers:
			z = layer(z)
			self.activations += [z]
			
		return z
	
	"""
		Inputs:
			* X - tf.Tensor
			* R - tf.Tensor
			* model - tf.keras.Model
			* activations - list
		Outputs: 
			* tf.Tensor
	"""
	def propagate(self, X, R, model, activations):
		for layer in range(len(model.layers))[::-1]:
			if(hasattr(model.layers[layer], "lrp")):
				R = model.layers[layer].lrp(activations[layer-1], R)
			else:
				if(layer-1 >= 0):
					R = lrp(activations[layer-1], R, model.layers[layer])
				else:
					R = lrp(X, R, model.layers[layer])
		
		return R
			
	"""
		Inputs:
			* X - tf.Tensor
			* R - tf.Tensor
		Outputs: 
			* tf.Tensor
	"""
	def backward(self, X, R):
		
		return self.propagate(X, R, self.model, self.activations)

	"""
		Inptus
			* X - tf.Tensor
		Outputs: 
			* tf.Tensor
	"""
	def call(self, X):
		
		y = self.forward(X)
		
		return self.backward(X, y)