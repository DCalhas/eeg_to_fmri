import tensorflow as tf

import tensorflow_probability as tfp

import numpy as np


"""
irDFT - inverse real Discrete Fourier Transform

Performs the fourier transform but only on the real part and generates higher frequency features
for better resolution, gradients are propagated to the parameters of a multivariate normal distribution

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
class irDFT(tf.keras.layers.Layer):

	def __init__(self, N, out=0):

		super(irDFT, self).__init__()
		
		if(out):
			assert out > N
			out_n = np.arange(N, out)
			out_k = out_n.reshape((N-out,1))
		else:
			out=N

		n = np.arange(out)
		k = n.reshape((out,1))
		
		self._N = N
		self._out=out-N
		
		#variable initializer
		self.n = self.add_weight('n',
								shape=[n.shape[0]],
								initializer=tf.constant_initializer(n),
								dtype=tf.float32,
								trainable=False)
		self.k = self.add_weight('k',
								shape=[k.shape[0], k.shape[1]],
								initializer=tf.constant_initializer(k),
								dtype=tf.float32,
								trainable=False)
		if(out > N):
			self.mu=self.add_weight('mu',
									shape=[self._out],
									initializer=tf.constant_initializer(np.zeros(self._out)),
									dtype=tf.float32,
									trainable=True)
			self.sigma=self.add_weight('sigma',
									shape=[self._out],
									initializer=tf.constant_initializer(np.ones(self._out)),
									constraint=tf.keras.constraints.NonNeg(),
									dtype=tf.float32,
									trainable=True)
			self.normal = tfp.distributions.Normal(loc=self.mu, scale=self.sigma)
			self.normal_res = tfp.layers.default_multivariate_normal_fn(shape=(self._out), dtype=tf.float32, 
																		trainable=True, name="normal_freq",
																	   add_variable_fn=self.add_variable)
		else:
			self.normal_res = None
		self.N = self.add_weight('N',
								shape=[1],
								initializer=tf.constant_initializer(out),
								dtype=tf.float32,
								trainable=False)

	def call(self, x):
		z_paddings = tf.constant([[0, 0], [self._N, 0]])
		x_paddings = tf.constant([[0, 0], [0, self._out]])
		
		n_k = self.k*self.n

		if(self.normal_res is not None):
			if(x.shape[0]==None):
				return tf.einsum("AB,NB->NA", (-tf.cos(2.0*np.pi*n_k/self.N)),tf.pad(x, x_paddings))
			else:
				z = tf.pad(self.normal.sample(x.shape[0]), z_paddings)#z ~ q_theta
		
		return tf.einsum("AB,NB->NA", (-tf.cos(2.0*np.pi*n_k/self.N)),tf.pad(x, x_paddings)) + \
				tf.einsum("AB,NB->NA", (-tf.cos(2.0*np.pi*n_k/self.N)),z)



"""
irDFT - inverse real Discrete Fourier Transform

Performs the fourier transform but only on the real part and generates higher frequency features
for better resolution, gradients are propagated to the parameters of a multivariate normal distribution

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
class ir3DFT(tf.keras.layers.Layer):

	def __init__(self, N1, N2, N3, out1=0, out2=0, out3=0, parameter_decay=0.01):

		super(ir3DFT, self).__init__()

		if(out1):
			assert out1 > N1
			assert out2 > N2
			assert out3 > N3
			out_n1 = np.arange(N1, out1)
			out_k1 = out_n1.reshape((N1-out1,1))
			out_n2 = np.arange(N2, out2)
			out_k2 = out_n2.reshape((N2-out2,1))
			out_n3 = np.arange(N3, out3)
			out_k3 = out_n3.reshape((N3-out3,1))
		else:
			out1=N1
			out2=N2
			out3=N3

		n1 = np.arange(out1)
		k1 = n1.reshape((out1,1))
		n2 = np.arange(out2)
		k2 = n2.reshape((out2,1))
		n3 = np.arange(out3)
		k3 = n3.reshape((out3,1))

		self._N1 = N1
		self._out1=out1-N1
		self._N2 = N2
		self._out2=out2-N2
		self._N3 = N3
		self._out3=out3-N3

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
		if(out1 > N1):
			self.mu1=self.add_weight('mu1',
									shape=[self._out1, self._N2, self._N3],
									initializer=tf.constant_initializer(np.zeros((self._out1, self._N2, self._N3))),
									dtype=tf.float32,
									trainable=True)
			self.sigma1=self.add_weight('sigma1',
									shape=[self._out1, self._N2, self._N3],
									initializer=tf.constant_initializer(np.ones((self._out1, self._N2, self._N3))),
									constraint=tf.keras.constraints.NonNeg(),
									dtype=tf.float32,
									trainable=True)
			self.normal1 = tfp.distributions.Normal(loc=self.mu1, scale=self.sigma1)

			self.add_loss(lambda: parameter_decay*tf.norm(self.mu1, ord=1))
			self.add_loss(lambda: parameter_decay*tf.norm(self.sigma1, ord=2))
		if(out2 > N2):
			self.mu2=self.add_weight('mu2',
									shape=[self._out1+self._N1, self._out2, self._N3],
									initializer=tf.constant_initializer(np.ones((self._out1+self._N1, self._out2, self._N3))),
									dtype=tf.float32,
									trainable=True)
			self.sigma2=self.add_weight('sigma2',
									shape=[self._out1+self._N1, self._out2, self._N3],
									initializer=tf.constant_initializer(np.ones((self._out1+self._N1, self._out2, self._N3))),
									constraint=tf.keras.constraints.NonNeg(),
									dtype=tf.float32,
									trainable=True)
			self.normal2 = tfp.distributions.Normal(loc=self.mu2, scale=self.sigma2)
			self.add_loss(lambda: parameter_decay*tf.norm(self.mu2, ord=1))
			self.add_loss(lambda: parameter_decay*tf.norm(self.sigma2, ord=2))
		if(out3 > N3):
			self.mu3=self.add_weight('mu3',
									shape=[self._out1+self._N1, self._out2+self._N2, self._out3],
									initializer=tf.constant_initializer(np.ones((self._out1+self._N1, self._out2+self._N2, self._out3))),
									dtype=tf.float32,
									trainable=True)
			self.sigma3=self.add_weight('sigma3',
									shape=[self._out1+self._N1, self._out2+self._N2, self._out3],
									initializer=tf.constant_initializer(np.ones((self._out1+self._N1, self._out2+self._N2, self._out3))),
									constraint=tf.keras.constraints.NonNeg(),
									dtype=tf.float32,
									trainable=True)
			self.normal3 = tfp.distributions.Normal(loc=self.mu3, scale=self.sigma3)
			self.add_loss(lambda: parameter_decay*tf.norm(self.mu3, ord=1))
			self.add_loss(lambda: parameter_decay*tf.norm(self.sigma3, ord=2))

		self.N1 = self.add_weight('N1',
								shape=[1],
								initializer=tf.constant_initializer(out1),
								dtype=tf.float32,
								trainable=False)
		self.N2 = self.add_weight('N2',
								shape=[1],
								initializer=tf.constant_initializer(out2),
								dtype=tf.float32,
								trainable=False)
		self.N3 = self.add_weight('N3',
								shape=[1],
								initializer=tf.constant_initializer(out3),
								dtype=tf.float32,
								trainable=False)
		self.direction = tfp.distributions.Uniform(low=-0.5, high=0.5)

		self.not_built = True


	def call(self, x):
		z1_paddings = tf.constant([[self._N1, 0], [0, 0], [0, 0]])
		z2_paddings = tf.constant([[0, 0], [self._N2, 0], [0, 0]])
		z3_paddings = tf.constant([[0, 0], [0, 0], [self._N3, 0]])
		x1_paddings = tf.constant([[0, 0], [0, self._out1], [0, 0], [0, 0]])
		x2_paddings = tf.constant([[0, 0], [0, 0], [0, self._out2], [0, 0]])
		x3_paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, self._out3]])

		n_k1 = self.k1*self.n1
		n_k2 = self.k2*self.n2
		n_k3 = self.k3*self.n3

		if(self.not_built):
			self.not_built=False
			z1 = tf.einsum("AB,NBCD->NACD", (tf.cos(2.0*np.pi*n_k1/self.N1)),tf.pad(x, x1_paddings))
			z1 = tf.einsum("AC,NBCD->NBAD", (tf.cos(2.0*np.pi*n_k2/self.N2)),tf.pad(z1, x2_paddings))
			return tf.einsum("AD,NBCD->NBCA", (tf.cos(2.0*np.pi*n_k3/self.N3)),tf.pad(z1, x3_paddings))

		z1 = tf.pad(self.normal1.sample(), z1_paddings)#z ~ q_theta
		z2 = tf.pad(self.normal2.sample(), z2_paddings)#z ~ q_theta
		z3 = tf.pad(self.normal3.sample(), z3_paddings)#z ~ q_theta

		r1 = tf.einsum("AB,NBCD->NACD", (tf.cos(2.0*np.pi*n_k1/self.N1)),tf.pad(x, x1_paddings))
		z1 = tf.einsum("AB,BCD->ACD", (tf.cos(2.0*np.pi*n_k1/self.N1)),z1)
		r1 = r1+z1
		
		r2 = tf.einsum("AC,NBCD->NBAD", (tf.cos(2.0*np.pi*n_k2/self.N2)),tf.pad(r1, x2_paddings))
		z2 = tf.einsum("AC,BCD->BAD", (tf.cos(2.0*np.pi*n_k2/self.N2)),z2)
		r2 = r2+z2
		
		r3 = tf.einsum("AD,NBCD->NBCA", (tf.cos(2.0*np.pi*n_k3/self.N3)),tf.pad(r2, x3_paddings))
		z3 = tf.einsum("AD,BCD->BCA", (tf.cos(2.0*np.pi*n_k3/self.N3)),z3)
		return r3+z3

"""
r3DFT - real Discrete Fourier Transform

Performs the discrete fourier transform

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
class r3DFT(tf.keras.layers.Layer):

	def __init__(self, N1, N2, N3):

		super(r3DFT, self).__init__()

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
		n_k1 = self.k1*self.n1
		n_k2 = self.k2*self.n2
		n_k3 = self.k3*self.n3

		z1 = tf.einsum("AB,NBCD->NACD", (tf.cos(2.0*np.pi*n_k1/self.N1)),x)
		z1 = tf.einsum("AC,NBCD->NBAD", (tf.cos(2.0*np.pi*n_k2/self.N2)),z1)
		return tf.einsum("AD,NBCD->NBCA", (tf.cos(2.0*np.pi*n_k3/self.N3)),z1)



"""
r3DFT - real Discrete Fourier Transform

Performs the discrete fourier transform

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
class DFT3(tf.keras.layers.Layer):

    def __init__(self, N1, N2, N3):

        super(DFT3, self).__init__()

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
        n_k1 = self.k1*self.n1
        n_k2 = self.k2*self.n2
        n_k3 = self.k3*self.n3

        r1 = tf.einsum("AB,NBCD->NACD", (tf.cos(2.0*np.pi*n_k1/self.N1)),x)
        i1 = tf.einsum("AB,NBCD->NACD", (-tf.sin(2.0*np.pi*n_k1/self.N1)),x)
        
        r1 = tf.einsum("AC,NBCD->NBAD", (tf.cos(2.0*np.pi*n_k2/self.N2)),r1)
        i1 = tf.einsum("AC,NBCD->NBAD", (-tf.sin(2.0*np.pi*n_k2/self.N2)),i1)
        
        r1 = tf.einsum("AD,NBCD->NBCA", (tf.cos(2.0*np.pi*n_k2/self.N2)),r1)
        i1 = tf.einsum("AD,NBCD->NBCA", (-tf.sin(2.0*np.pi*n_k2/self.N2)),i1)
        
        return tf.concat([tf.expand_dims(r1,axis=-1), tf.expand_dims(i1,axis=-1)], axis=-1)