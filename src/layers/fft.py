import tensorflow as tf

import tensorflow_probability as tfp

import numpy as np




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
	"""
	def __init__(self, in1, in2, in3, out1, out2, out3, rand1, rand2, rand3, coefs_perturb=True, dependent=True):

		super(variational_iDCT3D, self).__init__()

		assert out1 is not None
		assert out3 is not None
		assert out3 is not None

		self.in1 = in1
		self.in2 = in2
		self.in3 = in3

		self.rand1 = rand1
		self.rand2 = rand2
		self.rand3 = rand3
		self.coefs_perturb = coefs_perturb
		self.dependent = dependent

		if(self.coefs_perturb):
			self.normal= tfp.layers.default_mean_field_normal_fn()(tf.float32, [self.in1, self.in2, self.in3], 'normal_posterior', True, self.add_variable)
			self.normal_prior = tfp.layers.default_multivariate_normal_fn(tf.float32, [self.in1, self.in2, self.in3], 'normal_prior', True, self.add_variable)
		if(self.dependent):
			self.w = self.add_weight('W',
								shape=[self.in1*self.in2*self.in3, 1],
								initializer=tf.initializers.GlorotUniform(),
								dtype=tf.float32,
								trainable=True)

		self.padded_idct3 = padded_iDCT3D(in1+rand1, in2+rand2, in3+rand3, out1, out2, out3)

		self.normal1 = tfp.layers.default_mean_field_normal_fn()(tf.float32, [self.rand1, self.in2, self.in3], 'normal1_posterior', True, self.add_variable)
		self.normal2 = tfp.layers.default_mean_field_normal_fn()(tf.float32, [self.in1+self.rand1, self.rand2, self.in3], 'normal2_posterior', True, self.add_variable)
		self.normal3 = tfp.layers.default_mean_field_normal_fn()(tf.float32, [self.in1+self.rand1, self.in2+self.rand2, self.rand3], 'normal3_posterior', True, self.add_variable)
		self.normal1_prior = tfp.layers.default_multivariate_normal_fn(tf.float32, [self.rand1, self.in2, self.in3], 'normal1_prior', True, self.add_variable)
		self.normal2_prior = tfp.layers.default_multivariate_normal_fn(tf.float32, [self.in1+self.rand1, self.rand2, self.in3], 'normal2_prior', True, self.add_variable)
		self.normal3_prior = tfp.layers.default_multivariate_normal_fn(tf.float32, [self.in1+self.rand1, self.in2+self.rand2, self.rand3], 'normal3_prior', True, self.add_variable)
		
	def call(self, x):

		if(self.coefs_perturb):
			dist_normal = tfp.distributions.Normal(loc=self.normal.distribution.loc, scale=self.normal.distribution.scale)
			x = x*dist_normal.sample()
			#add kl divergence loss
			self.add_loss(tf.identity(tfp.distributions.kl_divergence(self.normal, self.normal_prior)))

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
		dist_normal1 = tfp.distributions.Normal(loc=self.normal1.distribution.loc, scale=self.normal1.distribution.scale)
		dist_normal2 = tfp.distributions.Normal(loc=self.normal2.distribution.loc, scale=self.normal2.distribution.scale)
		dist_normal3 = tfp.distributions.Normal(loc=self.normal3.distribution.loc, scale=self.normal3.distribution.scale)
		rand_coefs1 = dist_normal1.sample()#sample coefficients $c \sim \mathcal{N}(\mu,\sigma)$
		rand_coefs2 = dist_normal2.sample()#sample coefficients $c \sim \mathcal{N}(\mu,\sigma)$
		rand_coefs3 = dist_normal3.sample()#sample coefficients $c \sim \mathcal{N}(\mu,\sigma)$
		self.add_loss(tf.identity(tfp.distributions.kl_divergence(self.normal1, self.normal1_prior)))
		self.add_loss(tf.identity(tfp.distributions.kl_divergence(self.normal2, self.normal2_prior)))
		self.add_loss(tf.identity(tfp.distributions.kl_divergence(self.normal3, self.normal3_prior)))
		

		if(self.dependent):
			x_cond=tf.matmul(a=tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1]*tf.shape(x)[2]*tf.shape(x)[3],)), b=self.w)
			print(tf.math.multiply(rand_coefs1, x_cond))
			shape_coef1, shape_coef2, shape_coef3 = (tf.shape(rand_coefs1), tf.shape(rand_coefs2), tf.shape(rand_coefs3))
			rand_coefs1 = tf.reshape(tf.math.multiply(x_cond, tf.reshape(rand_coefs1, (shape_coef1[0]*shape_coef1[1]*shape_coef1[2],))), shape_coef1)
			rand_coefs2 = tf.reshape(tf.math.multiply(x_cond, tf.reshape(rand_coefs2, (shape_coef2[0]*shape_coef2[1]*shape_coef2[2],))), shape_coef2)
			rand_coefs3 = tf.reshape(tf.math.multiply(x_cond, tf.reshape(rand_coefs3, (shape_coef3[0]*shape_coef3[1]*shape_coef3[2],))), shape_coef3)
			
		z = tf.pad(x, rand_paddings1, constant_values=1.0)*tf.pad(rand_coefs1, in_paddings1, constant_values=1.0)
		z = tf.pad(z, rand_paddings2, constant_values=1.0)*tf.pad(rand_coefs2, in_paddings2, constant_values=1.0)
		return self.padded_idct3(tf.pad(z, rand_paddings3, constant_values=1.0)*tf.pad(rand_coefs3, in_paddings3, constant_values=1.0))
		"""
		z = tf.pad(x, rand_paddings1) + tf.pad(rand_coefs1, in_paddings1)
		z = tf.pad(z, rand_paddings2) + tf.pad(rand_coefs2, in_paddings2)
		return self.padded_idct3(tf.pad(z, rand_paddings3) + tf.pad(rand_coefs3, in_paddings3))
		"""
		




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