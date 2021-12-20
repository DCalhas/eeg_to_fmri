import tensorflow as tf



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
	print(layer.name)
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