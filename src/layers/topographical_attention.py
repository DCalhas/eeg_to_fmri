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


	#def compute_output_signature(self, input_signature):
	#	return [tf.TensorSpec(shape=(None, self.channels, self.features), dtype=tf.float32),
	#			tf.TensorSpec(shape=(None, self.channels, self.channels), dtype=tf.float32)]

	"""
	The defined topographical attention mechanism has an extra step:

		instead of performing the element wise product,
		one reduces the feature dimension,
		so instead of the attention weight matrix being of shape NxN
		it has shape NxNxF
		the F refers to the feature dimension that is reduced
	"""
	def call(self, X):

		return X
		
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
		return {
			'channels': self.channels,
			'features': self.features,
			'seed': self.seed
		}

	@classmethod
	def from_config(cls, config):
		return cls(**config)
