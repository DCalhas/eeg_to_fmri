import tensorflow as tf

"""
Topographical_Attention:

	Int channels
	Int features reduce over number of features

"""
class Topographical_Attention(tf.keras.layers.Layer):

	def __init__(self, channels, features, seed=None, **kwargs):

		self.channels=channels
		self.features=features
		self.seed=seed
		
		super(Topographical_Attention, self).__init__(**kwargs)

		self.A = self.add_weight('A',
								#shape=[self.channels,self.channels,self.features],
								shape=[self.channels,self.features],
								initializer=tf.initializers.GlorotUniform(seed=seed),
								dtype=tf.float32,
								trainable=True)

	"""
	The defined topographical attention mechanism has an extra step:

		instead of performing the element wise product,
		one reduces the feature dimension,
		so instead of the attention weight matrix being of shape NxN
		it has shape NxNxF
		the F refers to the feature dimension that is reduced
	"""
	@tf.function(input_signature=[tf.TensorSpec([None,64,1340], tf.float32)])
	def call(self, X):

		c = tf.tensordot(X, self.A, axes=[[2], [1]])
		#c = tf.einsum('NCF,CMF->NCM', X, self.A)
		W = tf.nn.softmax(c, axis=-1)#dimension that is reduced in the next einsum, is the one that sums to one
		self.attention_scores = W

		#sum over M all M channels are multiplied by the attention scores over axis M that is normalized 
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

	def get_config(self):
		config = {
			'channels': self.channels,
			'features': self.features,
			'seed': self.seed,
		}
		base_config = super(Topographical_Attention, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	@classmethod
	def from_config(cls, config):
		return cls(**config)
