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