import tensorflow as tf

"""
Topographical_Attention:

	Int channels
	Int features reduce over number of features

"""
class Topographical_Attention(tf.keras.layers.Layer):

	def __init__(self, channels, features, organize_channels=False, regularizer=None, seed=None, **kwargs):

		self.channels=channels
		self.features=features
		self.organize_channels=organize_channels
		self.regularizer=regularizer
		self.seed=seed

		super(Topographical_Attention, self).__init__(**kwargs)

	def build(self, input_shape):
		self.A = self.add_weight('A',
								#shape=[self.channels,self.channels,self.features],
								shape=[self.channels,self.features],
								regularizer=self.regularizer,
								initializer=tf.initializers.GlorotUniform(seed=self.seed),
								dtype=tf.float32,
								trainable=True)
		
	def call(self, X):
		"""
		The defined topographical attention mechanism has an extra step:

			instead of performing the element wise product,
			one reduces the feature dimension,
			so instead of the attention weight matrix being of shape NxN
			it has shape NxNxF
			the F refers to the feature dimension that is reduced
		""" 

		c = tf.tensordot(X, self.A, axes=[[2], [1]])
		#c = tf.einsum('NCF,CMF->NCM', X, self.A)
		W = tf.nn.softmax(c, axis=-1)#dimension that is reduced in the next einsum, is the one that sums to one
		self.attention_scores=W

		if(self.organize_channels):
			self.losses=[self.organize_regularization(W)]#### COMPUTE GRADIENTS W.R.T. self.A, acitivity regularizer is not possible
		#tf.print(tf.reduce_sum(self.organize_regularization(W)))

		return tf.linalg.matmul(W, X), self.attention_scores
		#return tf.einsum('NMF,NCM->NCF', X, W), self.attention_scores

	def lrp(self, x, y):
		#store attention scores
		self.call(x)

		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(x)
			
			z = tf.tensordot(x, self.attention_scores, axes=[[1], [2]])+1e-9
			#z = tf.einsum('NMF,NCM->NCF', x, self.attention_scores)+1e-9

			s = y/tf.reshape(z, y.shape)
			s = tf.reshape(s, z.shape)
			
			c = tape.gradient(tf.reduce_sum(z*s.numpy()), x)
			R = x*c

		return R

	def lrp_attention(self, x, y):
		#store attention scores
		self.call(x)

		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(self.attention_scores)

			z = tf.tensordot(x, self.attention_scores, axes=[[1], [2]])+1e-9
			#z = tf.einsum('NMF,NCM->NCF', x, self.attention_scores)+1e-9

			s = y/tf.reshape(z, y.shape)
			s = tf.reshape(s, z.shape)
			
			c = tape.gradient(tf.reduce_sum(z*s.numpy()), self.attention_scores)
			R = self.attention_scores*c

		return R

	def organize_regularization(self,x):
		"""
		minimization of this term allows the matrix to be heterogeneous row and column wise, that is every channel is selected and it reorders
		Example in numpy:

		>>> a=np.array([[0.99,0.005,0.005],[0.005,0.99,0.005],[0.005,0.005,0.99]])
		>>> b=np.array([[0.005,0.99,0.005],[0.005,0.99,0.005],[0.005,0.005,0.99]])
		>>> c=np.array([[0.7,0.1,0.2],[0.005,0.99,0.005],[0.005,0.005,0.99]])
		>>> 
		>>> -np.log(np.sum(a, axis=0))
		<<< array([-0., -0., -0.])
		>>> -np.log(np.sum(b, axis=0))
		<<< array([ 4.19970508, -0.68561891, -0.        ])
		>>> -np.log(np.sum(c, axis=0))
		<<< array([ 0.34249031, -0.09075436, -0.17814619])

		a: is the optimal objective that minimizes this term
		b: is the wrong and has the highest penalty for selecting a channel more than once
		c: is the suboptimal objective that still has room to improve in terms of 
		"""

		return -tf.math.log(tf.reduce_sum(x, axis=-2)+1e-9)#it is the -2 axis because we need to know if a channel is being selected more than once

	def get_config(self):
		return {
			'channels': self.channels,
			'features': self.features,
			'organize_channels': organize_channels,
			'seed': self.seed,
		}

	@classmethod
	def from_config(cls, config):
		return cls(**config)
