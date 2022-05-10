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
	def __init__(self, in1, in2, in3, out1, out2, out3, rand1, rand2, rand3, coefs_perturb=True, dependent=False, posterior_dimension=1, distribution=None):

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

		if(distribution is None):
			distribution="Normal"#default

		constraint=None
		loc_initializer=tf.initializers.GlorotUniform()
		scale_initializer=tf.initializers.Ones()
		if(self.distribution=="VonMisesFisher" or self.distribution=="VonMises"):
			constraint=tf.keras.constraints.NonNeg()

		if(self.coefs_perturb):
			self.normal= tfp.layers.default_mean_field_normal_fn(loc_constraint=constraint)(tf.float32, [self.in1, self.in2, self.in3], 'normal_posterior', True, self.add_weight)
			self.normal_prior = tfp.layers.default_multivariate_normal_fn(tf.float32, [self.in1, self.in2, self.in3], 'normal_prior', True, self.add_weight)
		if(self.dependent):
			self.w = self.add_weight('W',
								shape=[self.in1*self.in2*self.in3, posterior_dimension],
								initializer=tf.initializers.GlorotUniform(),
								dtype=tf.float32,
								trainable=True)

		self.padded_idct3 = padded_iDCT3D(in1+rand1, in2+rand2, in3+rand3, out1, out2, out3)

		self.shape_normal1 = (self.rand1, self.in2, self.in3)
		self.shape_normal2 = (self.in1+self.rand1, self.rand2, self.in3)
		self.shape_normal3 = (self.in1+self.rand1, self.in2+self.rand2, self.rand3)

		if(self.distribution=="VonMisesFisher"):
			self.angular_loc1 = self.add_weight('angular_loc1_posterior',
										shape=[posterior_dimension, self.shape_normal1[0]*self.shape_normal1[1]*self.shape_normal1[2], 2],
										initializer=loc_initializer,
										constraint=None,
										dtype=tf.float32,
										trainable=True)
			self.angular_scale1 = self.add_weight('angular_scale1_posterior',
										shape=[posterior_dimension, self.shape_normal1[0]*self.shape_normal1[1]*self.shape_normal1[2]],
										initializer=scale_initializer,
										constraint=constraint,
										dtype=tf.float32,
										trainable=True)


			self.angular_loc2 = self.add_weight('angular_loc2_posterior',
										shape=[posterior_dimension, self.shape_normal2[0]*self.shape_normal2[1]*self.shape_normal2[2], 2],
										initializer=loc_initializer,
										constraint=None,
										dtype=tf.float32,
										trainable=True)
			self.angular_scale2 = self.add_weight('angular_scale2_posterior',
										shape=[posterior_dimension, self.shape_normal2[0]*self.shape_normal2[1]*self.shape_normal2[2]],
										initializer=scale_initializer,
										constraint=constraint,
										dtype=tf.float32,
										trainable=True)

			self.angular_loc3 = self.add_weight('angular_loc3_posterior',
										shape=[posterior_dimension, self.shape_normal3[0]*self.shape_normal3[1]*self.shape_normal3[2], 2],
										initializer=loc_initializer,
										constraint=None,
										dtype=tf.float32,
										trainable=True)
			self.angular_scale3 = self.add_weight('angular_scale3_posterior',
										shape=[posterior_dimension, self.shape_normal3[0]*self.shape_normal3[1]*self.shape_normal3[2]],
										initializer=scale_initializer,
										constraint=constraint,
										dtype=tf.float32,
										trainable=True)

			self.real_angle = self.add_weight('angular_scale3_posterior',
										shape=[1,2],
										initializer=tf.constant_initializer(np.array([[1.,1.]])),
										dtype=tf.float32,
										trainable=False)

		if(self.distribution=="VonMises"):
			self.cartesian_loc1 = self.add_weight('cartesian_loc1_posterior',
										shape=[posterior_dimension, self.shape_normal1[0]*self.shape_normal1[1]*self.shape_normal1[2]],
										initializer=loc_initializer,
										constraint=None,
										dtype=tf.float32,
										trainable=True)
			self.cartesian_scale1 = self.add_weight('cartesian_scale1_posterior',
										shape=[posterior_dimension, self.shape_normal1[0]*self.shape_normal1[1]*self.shape_normal1[2]],
										initializer=scale_initializer,
										constraint=constraint,
										dtype=tf.float32,
										trainable=True)


			self.cartesian_loc2 = self.add_weight('cartesian_loc2_posterior',
										shape=[posterior_dimension, self.shape_normal2[0]*self.shape_normal2[1]*self.shape_normal2[2]],
										initializer=loc_initializer,
										constraint=None,
										dtype=tf.float32,
										trainable=True)
			self.cartesian_scale2 = self.add_weight('cartesian_scale2_posterior',
										shape=[posterior_dimension, self.shape_normal2[0]*self.shape_normal2[1]*self.shape_normal2[2]],
										initializer=scale_initializer,
										constraint=constraint,
										dtype=tf.float32,
										trainable=True)

			self.cartesian_loc3 = self.add_weight('cartesian_loc3_posterior',
										shape=[posterior_dimension, self.shape_normal3[0]*self.shape_normal3[1]*self.shape_normal3[2]],
										initializer=loc_initializer,
										constraint=None,
										dtype=tf.float32,
										trainable=True)
			self.cartesian_scale3 = self.add_weight('cartesian_scale3_posterior',
										shape=[posterior_dimension, self.shape_normal3[0]*self.shape_normal3[1]*self.shape_normal3[2]],
										initializer=scale_initializer,
										constraint=constraint,
										dtype=tf.float32,
										trainable=True)

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
		#dist1 = getattr(tfp.distributions, self.distribution)(self.loc1, self.scale1)
		#dist2 = getattr(tfp.distributions, self.distribution)(self.loc2, self.scale2)
		#dist3 = getattr(tfp.distributions, self.distribution)(self.loc3, self.scale3)
		#rand_coefs1 = dist1.sample()#sample coefficients $c \sim \mathcal{N}(\mu,\sigma)$
		#rand_coefs2 = dist2.sample()#sample coefficients $c \sim \mathcal{N}(\mu,\sigma)$
		#rand_coefs3 = dist3.sample()#sample coefficients $c \sim \mathcal{N}(\mu,\sigma)$
		if(self.distribution=="VonMises"):
			cartesian_dist1 = tfp.distributions.VonMises(self.cartesian_loc1, self.cartesian_scale1)
			cartesian_dist2 = tfp.distributions.VonMises(self.cartesian_loc2, self.cartesian_scale2)
			cartesian_dist3 = tfp.distributions.VonMises(self.cartesian_loc3, self.cartesian_scale3)
			rand_coefs1=cartesian_dist1.sample()#creating random coefficients with random angles and coordinates
			rand_coefs2=cartesian_dist2.sample()#creating random coefficients with random angles and coordinates
			rand_coefs3=cartesian_dist3.sample()#creating random coefficients with random angles and coordinates
		elif(self.distribution=="VonMisesFisher"):#learn the angles of the frequency space as well??
			angular_dist1 = tfp.distributions.VonMisesFisher(self.angular_loc1*2*np.pi, self.angular_scale1)
			angular_dist2 = tfp.distributions.VonMisesFisher(self.angular_loc2*2*np.pi, self.angular_scale2)
			angular_dist3 = tfp.distributions.VonMisesFisher(self.angular_loc3*2*np.pi, self.angular_scale3)
			rand_coefs1=tf.squeeze(tf.matmul(angular_dist1.sample(), tf.transpose(self.real_angle)), axis=-1)#filter real part of the 2D sphere
			rand_coefs2=tf.squeeze(tf.matmul(angular_dist2.sample(), tf.transpose(self.real_angle)), axis=-1)#filter real part of the 2D sphere
			rand_coefs3=tf.squeeze(tf.matmul(angular_dist3.sample(), tf.transpose(self.real_angle)), axis=-1)#filter real part of the 2D sphere
		else:
			raise NotImplementedError

		if(self.dependent):
			x_cond = tf.matmul(tf.reshape(x, (tf.shape(x)[0], 1, tf.shape(x)[1]*tf.shape(x)[2]*tf.shape(x)[3],)), self.w)
			x_cond = tf.squeeze(x_cond, axis=1)#shape = [None, H] = [Batch, dependent_dimension]
			#attention?
			x_cond = tf.nn.softmax(x_cond)
			rand_coefs1 = tf.matmul(x_cond, rand_coefs1)#shape = [None, F] = [Batch, F]
			rand_coefs2 = tf.matmul(x_cond, rand_coefs2)#shape = [None, F] = [Batch, F]
			rand_coefs3 = tf.matmul(x_cond, rand_coefs3)#shape = [None, F] = [Batch, F]

		rand_coefs1 = tf.reshape(rand_coefs1, (tf.shape(rand_coefs1)[0],)+self.shape_normal1)
		rand_coefs2 = tf.reshape(rand_coefs2, (tf.shape(rand_coefs2)[0],)+self.shape_normal2)
		rand_coefs3 = tf.reshape(rand_coefs3, (tf.shape(rand_coefs3)[0],)+self.shape_normal3)
			
		if(self.coefs_perturb):
			dist_normal = tfp.distributions.Normal(loc=self.normal.distribution.loc, scale=self.normal.distribution.scale)
			x = x*dist_normal.sample()
			
		z = tf.pad(x, rand_paddings1, constant_values=1.0)*tf.pad(rand_coefs1, in_paddings1, constant_values=1.0)
		z = tf.pad(z, rand_paddings2, constant_values=1.0)*tf.pad(rand_coefs2, in_paddings2, constant_values=1.0)
		return self.padded_idct3(tf.pad(z, rand_paddings3, constant_values=1.0)*tf.pad(rand_coefs3, in_paddings3, constant_values=1.0))
		
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