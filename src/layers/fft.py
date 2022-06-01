import tensorflow as tf

import tensorflow_probability as tfp

import numpy as np

from layers import fourier_features #to import _get_default_scale




"""
DCT3D - real Discrete Cosine Transform

Performs the discrete cosine transform

Example usage:
>>> import numpy as np
>>> import tensorflow as tf
>>> import tensorflow_probability as tfp
>>>
>>> x = tf.constant(np.expand_dims(np.random.rand(16,10),axis=-1), dtype=tf.float32)
>>> N = x.shape[1]
>>> irdft = irDFT(N, out=N*2)
>>> irdft(x)
"""
class DCT3D(tf.keras.layers.Layer):

	def __init__(self, N1, N2, N3):

		super(DCT3D, self).__init__()

		self.N1=N1
		self.N2=N2
		self.N3=N3

		n1 = np.arange(N1)
		k1 = n1.reshape((N1,1))
		n2 = np.arange(N2)
		k2 = n2.reshape((N2,1))
		n3 = np.arange(N3)
		k3 = n3.reshape((N3,1))

		#variable initializer
		self.n1 = self.add_weight('n1',
								shape=[n1.shape[0]],
								initializer=tf.constant_initializer(n1),
								dtype=tf.float32,
								trainable=False)
		self.k1 = self.add_weight('k1',
								shape=[k1.shape[0], k1.shape[1]],
								initializer=tf.constant_initializer(k1),
								dtype=tf.float32,
								trainable=False)
		self.n2 = self.add_weight('n2',
								shape=[n2.shape[0]],
								initializer=tf.constant_initializer(n2),
								dtype=tf.float32,
								trainable=False)
		self.k2 = self.add_weight('k2',
								shape=[k2.shape[0], k2.shape[1]],
								initializer=tf.constant_initializer(k2),
								dtype=tf.float32,
								trainable=False)
		self.n3 = self.add_weight('n3',
								shape=[n3.shape[0]],
								initializer=tf.constant_initializer(n3),
								dtype=tf.float32,
								trainable=False)
		self.k3 = self.add_weight('k3',
								shape=[k3.shape[0], k3.shape[1]],
								initializer=tf.constant_initializer(k3),
								dtype=tf.float32,
								trainable=False)

		self.N1 = self.add_weight('N1',
								shape=[1],
								initializer=tf.constant_initializer(N1),
								dtype=tf.float32,
								trainable=False)
		self.N2 = self.add_weight('N2',
								shape=[1],
								initializer=tf.constant_initializer(N2),
								dtype=tf.float32,
								trainable=False)
		self.N3 = self.add_weight('N3',
								shape=[1],
								initializer=tf.constant_initializer(N3),
								dtype=tf.float32,
								trainable=False)
		
	def call(self, x):
		z3 = 2*tf.tensordot((tf.cos(np.pi*(2*self.n3+1)*self.k3/(2*self.N3))), x, axes=[[1], [3]])
		z3 = tf.transpose(z3, [1,2,3,0])
		
		z2 = 2*tf.tensordot((tf.cos(np.pi*(2*self.n2+1)*self.k2/(2*self.N2))), z3, axes=[[1], [2]])
		z2 = tf.transpose(z2, [1,2,0,3])
		
		z1 = 2*tf.tensordot((tf.cos(np.pi*(2*self.n1+1)*self.k1/(2*self.N1))), z2, axes=[[1], [1]])
		z1 = tf.transpose(z1, [1,0,2,3])
		return z1

	def get_config(self):
		return {
			'N1': self.N1,
			'N2': self.N2,
			'N3': self.N3,
		}

	@classmethod
	def from_config(cls, config):
		return cls(**config)

	
	
"""
DCT3D - real Discrete Cosine Transform

Performs the discrete cosine transform

Example usage:
>>> import numpy as np
>>> import tensorflow as tf
>>> import tensorflow_probability as tfp
>>>
>>> x = tf.constant(np.expand_dims(np.random.rand(16,10),axis=-1), dtype=tf.float32)
>>> N = x.shape[1]
>>> irdft = irDFT(N, out=N*2)
>>> irdft(x)
"""
class iDCT3D(tf.keras.layers.Layer):

	def __init__(self, N1, N2, N3):

		super(iDCT3D, self).__init__()

		self.N1=N1
		self.N2=N2
		self.N3=N3

		n1 = np.arange(N1)
		k1 = n1.reshape((N1,1))
		n2 = np.arange(N2)
		k2 = n2.reshape((N2,1))
		n3 = np.arange(N3)
		k3 = n3.reshape((N3,1))

		#variable initializer
		self.n1 = self.add_weight('n1',
								shape=[n1.shape[0]],
								initializer=tf.constant_initializer(n1),
								dtype=tf.float32,
								trainable=False)
		self.k1 = self.add_weight('k1',
								shape=[k1.shape[0], k1.shape[1]],
								initializer=tf.constant_initializer(k1),
								dtype=tf.float32,
								trainable=False)
		self.n2 = self.add_weight('n2',
								shape=[n2.shape[0]],
								initializer=tf.constant_initializer(n2),
								dtype=tf.float32,
								trainable=False)
		self.k2 = self.add_weight('k2',
								shape=[k2.shape[0], k2.shape[1]],
								initializer=tf.constant_initializer(k2),
								dtype=tf.float32,
								trainable=False)
		self.n3 = self.add_weight('n3',
								shape=[n3.shape[0]],
								initializer=tf.constant_initializer(n3),
								dtype=tf.float32,
								trainable=False)
		self.k3 = self.add_weight('k3',
								shape=[k3.shape[0], k3.shape[1]],
								initializer=tf.constant_initializer(k3),
								dtype=tf.float32,
								trainable=False)

		self.N1 = self.add_weight('N1',
								shape=[1],
								initializer=tf.constant_initializer(N1),
								dtype=tf.float32,
								trainable=False)
		self.N2 = self.add_weight('N2',
								shape=[1],
								initializer=tf.constant_initializer(N2),
								dtype=tf.float32,
								trainable=False)
		self.N3 = self.add_weight('N3',
								shape=[1],
								initializer=tf.constant_initializer(N3),
								dtype=tf.float32,
								trainable=False)
		
		#remove this
		norm3 = np.ones((N1,N2,N3))
		norm3[:,:,1:] = 2
		norm2 = np.ones((N1,N2,N3))
		norm2[:,1:,:] = 2
		norm1 = np.ones((N1,N2,N3))
		norm1[1:,:,:] = 2
		
		self.norm1 = self.add_weight('norm1',
								shape=[N1,N2,N3],
								initializer=tf.constant_initializer(norm1),
								dtype=tf.float32,
								trainable=False)
		self.norm2 = self.add_weight('norm2',
								shape=[N1,N2,N3],
								initializer=tf.constant_initializer(norm2),
								dtype=tf.float32,
								trainable=False)
		self.norm3 = self.add_weight('norm3',
								shape=[N1,N2,N3],
								initializer=tf.constant_initializer(norm3),
								dtype=tf.float32,
								trainable=False)
		
	def call(self, x):
		z3 = (1/(2*self.N3))*tf.tensordot((tf.cos(np.pi*self.n3*(2*self.k3+1)/(2*self.N3))), x*self.norm3, 
										  axes=[[1], [3]])
		z3 = tf.transpose(z3, [1,2,3,0])
		
		z2 = (1/(2*self.N2))*tf.tensordot((tf.cos(np.pi*self.n2*(2*self.k2+1)/(2*self.N2))), z3*self.norm2, 
										  axes=[[1], [2]])
		z2= tf.transpose(z2, [1,2,0,3])
		
		z1 = (1/(2*self.N1))*tf.tensordot((tf.cos(np.pi*self.n1*(2*self.k1+1)/(2*self.N1))), z2*self.norm1, 
										  axes=[[1], [1]])
		
		return tf.transpose(z1, [1,0,2,3])

	def get_config(self):
		return {
			'N1': self.N1,
			'N2': self.N2,
			'N3': self.N3,
		}

	@classmethod
	def from_config(cls, config):
		return cls(**config)




"""
DCT3D - real Discrete Cosine Transform

Performs the discrete cosine transform

Example usage:
>>> import numpy as np
>>> import tensorflow as tf
>>> import tensorflow_probability as tfp
>>>
>>> x = tf.constant(np.expand_dims(np.random.rand(16,10),axis=-1), dtype=tf.float32)
>>> N = x.shape[1]
>>> irdft = irDFT(N, out=N*2)
>>> irdft(x)
"""
class padded_iDCT3D(tf.keras.layers.Layer):

	def __init__(self, in1, in2, in3, out1, out2, out3):

		super(padded_iDCT3D, self).__init__()
		
		assert out1 is not None
		assert out3 is not None
		assert out3 is not None
		
		self.in1 = in1
		self.in2 = in2
		self.in3 = in3
		
		self.out1 = out1
		self.out2 = out2
		self.out3 = out3
		
		self.idct3 = iDCT3D(out1, out2, out3)
		
	def call(self, x):
		
		paddings = [[0,0],
					[0, self.out1-self.in1],
				   [0, self.out2-self.in2],
				   [0, self.out3-self.in3]]
		
		return self.idct3(tf.pad(x, paddings))

	def get_config(self):
		return {
			"in1": self.in1,
			"in2": self.in2,
			"in3": self.in3,
			"out1": self.out1,
			"out2": self.out2,
			"out3": self.out3,
		}

	@classmethod
	def from_config(cls, config):
		return cls(**config)



"""
DCT3D - real Discrete Cosine Transform

Performs the discrete cosine transform

Example usage:
>>> import numpy as np
>>> import tensorflow as tf
>>> import tensorflow_probability as tfp
>>>
>>> x = tf.constant(np.expand_dims(np.random.rand(16,10),axis=-1), dtype=tf.float32)
>>> N = x.shape[1]
>>> irdft = irDFT(N, out=N*2)
>>> irdft(x)
"""
class variational_iDCT3D(tf.keras.layers.Layer):
	"""
	in1 - int - first dimension input

	If Gamma is used please cite arXiv:1805.08498 - Figurnov et al. 2019


	distribution variances
	"""
	def __init__(self, in1, in2, in3, out1, out2, out3, rand1, rand2, rand3, coefs_perturb=True, dependent=False, posterior_dimension=1, distribution=None, random_padding=False):

		super(variational_iDCT3D, self).__init__()

		assert out1 is not None
		assert out3 is not None
		assert out3 is not None


		assert (not dependent and posterior_dimension == 1) or dependent

		self.in1 = in1
		self.in2 = in2
		self.in3 = in3
		self.out1 = out1
		self.out2 = out2
		self.out3 = out3
		self.rand1 = rand1
		self.rand2 = rand2
		self.rand3 = rand3
		self.coefs_perturb = coefs_perturb
		self.dependent = dependent
		self.posterior_dimension = posterior_dimension
		self.distribution = distribution
		self.random_padding = random_padding

		if(distribution is None):
			distribution="Normal"#default

		loc_initializer=tf.initializers.GlorotUniform()
		scale_initializer=tf.initializers.Ones()
		constraint=tf.keras.constraints.NonNeg()
		
		if(self.coefs_perturb):
			self.normal= tfp.layers.default_mean_field_normal_fn(loc_constraint=constraint)(tf.float32, [self.in1, self.in2, self.in3], 'normal_posterior', True, self.add_weight)
			self.normal_prior = tfp.layers.default_multivariate_normal_fn(tf.float32, [self.in1, self.in2, self.in3], 'normal_prior', True, self.add_weight)
		if(self.dependent):
			self.w1 = self.add_weight('W1',
								shape=[self.in1*self.in2*self.in3, posterior_dimension],
								initializer=tf.initializers.GlorotUniform(),
								dtype=tf.float32,
								trainable=True)
			self.w2 = self.add_weight('W2',
								shape=[self.in1*self.in2*self.in3, posterior_dimension],
								initializer=tf.initializers.GlorotUniform(),
								dtype=tf.float32,
								trainable=True)
			self.w3 = self.add_weight('W3',
								shape=[self.in1*self.in2*self.in3, posterior_dimension],
								initializer=tf.initializers.GlorotUniform(),
								dtype=tf.float32,
								trainable=True)

		self.padded_idct3 = padded_iDCT3D(in1+rand1, in2+rand2, in3+rand3, out1, out2, out3)

		self.shape_normal1 = (self.rand1, self.in2, self.in3)
		self.shape_normal2 = (self.in1+self.rand1, self.rand2, self.in3)
		self.shape_normal3 = (self.in1+self.rand1, self.in2+self.rand2, self.rand3)

		if(self.distribution=="VonMises"):
			self.cartesian_loc = self.add_weight('cartesian_loc_posterior',
										shape=[posterior_dimension, self.shape_normal1[0]*self.shape_normal1[1]*self.shape_normal1[2]+self.shape_normal2[0]*self.shape_normal2[1]*self.shape_normal2[2]+self.shape_normal3[0]*self.shape_normal3[1]*self.shape_normal3[2]],
										initializer=loc_initializer,
										constraint=None,
										dtype=tf.float32,
										trainable=True)
			self.cartesian_scale = self.add_weight('cartesian_scale_posterior',
										shape=[posterior_dimension, self.shape_normal1[0]*self.shape_normal1[1]*self.shape_normal1[2]+self.shape_normal2[0]*self.shape_normal2[1]*self.shape_normal2[2]+self.shape_normal3[0]*self.shape_normal3[1]*self.shape_normal3[2]],
										initializer=scale_initializer,
										constraint=constraint,
										dtype=tf.float32,
										trainable=True)

		if(self.random_padding):
			self.random_pad1 = RandomizeFrequencies(self.in1, self.in1+self.rand1, dim=1)
			self.random_pad2 = RandomizeFrequencies(self.in2, self.in2+self.rand2, dim=2)
			self.random_pad3 = RandomizeFrequencies(self.in3, self.in3+self.rand3, dim=3)

	def call(self, x):

		rand_paddings1 = [[0,0],
					[0, self.rand1],
				   [0, 0],
				   [0, 0]]
		rand_paddings2 = [[0,0],
					[0, 0],
				   [0, self.rand2],
				   [0, 0]]
		rand_paddings3 = [[0,0],
					[0, 0],
				   [0, 0],
				   [0, self.rand3]]

		in_paddings1 = [[0, 0],
					[self.in1, 0],
				   [0, 0],
				   [0, 0]]
		in_paddings2 = [[0, 0],
					[0, 0],
				   [self.in2, 0],
				   [0, 0]]
		in_paddings3 = [[0, 0],
					[0, 0],
				   [0, 0],
				   [self.in3, 0]]
		
		#https://github.com/tensorflow/probability/blob/88d217dfe8be49050362eb14ba3076c0dc0f1ba6/tensorflow_probability/python/distributions/normal.py#L174
		if(self.distribution=="VonMises"):
			cartesian_dist = tfp.distributions.VonMises(self.cartesian_loc, self.cartesian_scale)
			rand_coefs=tf.cos(cartesian_dist.sample())
			rand_coefs1, rand_coefs2, rand_coefs3 = tf.split(rand_coefs, 
															[self.shape_normal1[0]*self.shape_normal1[1]*self.shape_normal1[2],
															self.shape_normal2[0]*self.shape_normal2[1]*self.shape_normal2[2],
															self.shape_normal3[0]*self.shape_normal3[1]*self.shape_normal3[2]], 
															axis=-1)

		if(self.dependent):
			x_cond1 = tf.squeeze(tf.matmul(tf.reshape(x, (tf.shape(x)[0], 1, tf.shape(x)[1]*tf.shape(x)[2]*tf.shape(x)[3],)), self.w1), axis=1)
			x_cond2 = tf.squeeze(tf.matmul(tf.reshape(x, (tf.shape(x)[0], 1, tf.shape(x)[1]*tf.shape(x)[2]*tf.shape(x)[3],)), self.w2), axis=1)
			x_cond3 = tf.squeeze(tf.matmul(tf.reshape(x, (tf.shape(x)[0], 1, tf.shape(x)[1]*tf.shape(x)[2]*tf.shape(x)[3],)), self.w3), axis=1)
			#attention?
			#x_cond1 = tf.nn.softmax(x_cond1)
			#x_cond2 = tf.nn.softmax(x_cond2)
			#x_cond3 = tf.nn.softmax(x_cond3)
			rand_coefs1 = tf.matmul(x_cond1, rand_coefs1)#shape = [None, F] = [Batch, F]
			rand_coefs2 = tf.matmul(x_cond2, rand_coefs2)#shape = [None, F] = [Batch, F]
			rand_coefs3 = tf.matmul(x_cond3, rand_coefs3)#shape = [None, F] = [Batch, F]

		rand_coefs1 = tf.reshape(rand_coefs1, (tf.shape(rand_coefs1)[0],)+self.shape_normal1)
		rand_coefs2 = tf.reshape(rand_coefs2, (tf.shape(rand_coefs2)[0],)+self.shape_normal2)
		rand_coefs3 = tf.reshape(rand_coefs3, (tf.shape(rand_coefs3)[0],)+self.shape_normal3)
		
		if(self.coefs_perturb):
			dist_normal = tfp.distributions.Normal(loc=self.normal.distribution.loc, scale=self.normal.distribution.scale)
			x = x*dist_normal.sample()
			
		if(self.random_padding):
			z = self.random_pad1(x,rand_coefs1)
			z = self.random_pad2(z,rand_coefs2)
			z = self.random_pad3(z,rand_coefs3)
		else:
			z = tf.pad(x, rand_paddings1, constant_values=1.0)*tf.pad(rand_coefs1, in_paddings1, constant_values=1.0)
			z = tf.pad(z, rand_paddings2, constant_values=1.0)*tf.pad(rand_coefs2, in_paddings2, constant_values=1.0)
			z = tf.pad(z, rand_paddings3, constant_values=1.0)*tf.pad(rand_coefs3, in_paddings3, constant_values=1.0)

		print(z.shape)
		return self.padded_idct3(z)
		
	def get_config(self):
		return {
			"in1": self.in1,
			"in2": self.in2,
			"in3": self.in3,
			"out1": self.out1,
			"out2": self.out2,
			"out3": self.out3,
			"rand1": self.rand1,
			"rand2": self.rand2,
			"rand3": self.rand3,
			"coefs_perturb": self.coefs_perturb,
			"dependent": self.dependent,
			"posterior_dimension": self.posterior_dimension,
			"distribution": self.distribution,
			"random_padding": self.random_padding,
		}

	@classmethod
	def from_config(cls, config):
		return cls(**config)		




"""
Spectral Dropout Layer - Khan et al. 2019 - https://www.sciencedirect.com/science/article/pii/S0893608018302715

tfp.distributions.Bernoulli(probs=p)

>>> import tensorflow as tf
>>> import tensorflow_probability as tfp
>>> layer = SpectralDropout(64,64,30,0.5)
>>> layer(tf.ones(1, 64,64,30))

"""
class SpectralDropout(tf.keras.layers.Layer):
	"""
	in1 - int - first dimension input
	"""
	def __init__(self, in1, in2, in3, probs=None, dtype=tf.float32):
		super(SpectralDropout, self).__init__()


		if(probs is None):
			probs=tf.constant(0.5, shape=(in1, in2, in3))
			self.probs=self.add_weight('probs',
								shape=[in1, in2, in3],
								initializer=tf.constant_initializer(probs.numpy()),
								constraint=tf.keras.constraints.NonNeg(),
								dtype=tf.float32,
								trainable=False)#can not be trained since Bernoulli sampling is not differentiable
		else:
			self.probs=tf.constant_initializer(probs, shape=(in1, in2, in3))

		self.mask_dist = tfp.distributions.Bernoulli(probs=self.probs, dtype=dtype)

	def call(self, X):
		return X*self.mask_dist.sample()




class RandomizeFrequencies(tf.keras.layers.Layer):
	"""
	Randomize the predicted frequencies and estimate the rest
	Just like in fMRI enhancement

	This is just like a padding, but instead of being reflected on the sides, 
	it specifies the positions of the X in the new representation in a random initialized order
	"""
	def __init__(self, in_shape, out_shape, dim):

		assert dim in [1,2,3], "This layer only operates for 3D representations"
		assert out_shape > in_shape, "The output shape has to be bigger than the input"

		super(RandomizeFrequencies, self).__init__()

		self.out_shape=out_shape
		self.shape1=np.sort(np.random.choice(np.arange(out_shape), size=in_shape, replace=False))
		self.dim=dim

	
	def call(self, X, C):
		"""
		list of all splits of both X and C

			>>> from layers.fft import RandomizeFrequencies
			>>> import tensorflow as tf
			>>> a = tf.random.uniform((1,2,2,2))
			>>> b = tf.random.uniform((1,2,2,2))
			>>> randomize = RandomizeFrequencies(2, 4, dim=1)
			>>> randomize(a,b)
			>>> randomize.shape1
			>>> a
			>>> b

		"""

		Z=None
		added=0

		for split in range(len(self.shape1)):
			if(Z is None):
				if(self.shape1[split]==0):
					if(self.dim==1):
						Z=X[:,split:split+1,:,:]
					elif(self.dim==2):
						Z=X[:,:,split:split+1,:]
					elif(self.dim==3):
						Z=X[:,:,:,split:split+1]
					else:
						raise NotImplementedError
				else:
					if(self.dim==1):
						Z=tf.concat([C[:,:self.shape1[split],:,:], X[:,split:split+1,:,:]], axis=self.dim)
					elif(self.dim==2):
						Z=tf.concat([C[:,:,:self.shape1[split],:], X[:,:,split:split+1,:]], axis=self.dim)
					elif(self.dim==3):
						Z=tf.concat([C[:,:,:,:self.shape1[split]], X[:,:,:,split:split+1]], axis=self.dim)
					else:
						raise NotImplementedError
					added+=self.shape1[split]

			else:
			
				diff=self.shape1[split]-self.shape1[split-1]-1
				if(self.dim==1):
					Z=tf.concat([Z,C[:,added:added+diff,:,:]], axis=self.dim)
				elif(self.dim==2):
					Z=tf.concat([Z,C[:,:,added:added+diff,:]], axis=self.dim)
				elif(self.dim==3):
					Z=tf.concat([Z,C[:,:,:,added:added+diff]], axis=self.dim)
				else:
					raise NotImplementedError

				added+=self.shape1[split]-self.shape1[split-1]-1

				if(self.dim==1):
					Z=tf.concat([Z,X[:,split:split+1,:,:]], axis=self.dim)
				elif(self.dim==2):
					Z=tf.concat([Z,X[:,:,split:split+1,:]], axis=self.dim)
				elif(self.dim==3):
					Z=tf.concat([Z,X[:,:,:,split:split+1]], axis=self.dim)
				else:
					raise NotImplementedError
			
		if(tf.shape(Z)[self.dim] < self.out_shape):
			if(self.dim==1):
				Z=tf.concat([Z,C[:,added:,:,:]], axis=self.dim)
			elif(self.dim==2):
				Z=tf.concat([Z,C[:,:,added:,:]], axis=self.dim)
			elif(self.dim==3):
				Z=tf.concat([Z,C[:,:,:,added:]], axis=self.dim)
			else:
				raise NotImplementedError
		return Z