import tensorflow as tf


"""
Resnet-18 block that has implemented 
the backward step for LRP - Layer-Wise
Relevance Backpropagation 

Example usage:
	>>> import tensorflow as tf
	>>> import resnet_block 
	>>> layer = resnet_block.ResBlock(tf.keras.layers.Conv3D, (5,5,5), (1,1,1), 1, maxpool=False, seed=42)
	>>> x = tf.ones((1,10,10,10,1))
	>>> layer.lrp(x, layer(x))
"""
class ResBlock(tf.keras.layers.Layer):
	"""
		inputs:
			* x - Tensor
			* kernel_size - tuple
			* stride_size - tuple
			* n_channels - int
			* maxpool - bool
			* batch_norm - bool
			* weight_decay - float
			* skip_connections - bool
			* maxpool_k - tuple
			* maxpool_s - tuple
			* seed - int
	"""
	def __init__(self, operation, kernel_size, stride_size, n_channels,
						maxpool=True, batch_norm=True, 
						weight_decay=0.000, skip_connections=True,
						maxpool_k=None, maxpool_s=None,
						seed=None):
		super(ResBlock, self).__init__()

		self.set_layers(operation, kernel_size, stride_size, n_channels,
						maxpool=maxpool, batch_norm=batch_norm, 
						weight_decay=weight_decay, skip_connections=skip_connections,
						maxpool_k=maxpool_k, maxpool_s=maxpool_s, seed=seed)

	def set_layers(self, operation, kernel_size, stride_size, n_channels,
						maxpool=True, batch_norm=True, 
						weight_decay=0.000, skip_connections=True,
						maxpool_k=None, maxpool_s=None, seed=None):

		self.left_layers = []
		self.right_layers = []
		self.join_layers = []

		self.left_layers += [operation(filters=n_channels, kernel_size=kernel_size, strides=stride_size,
										kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
										bias_regularizer=tf.keras.regularizers.L2(weight_decay),
										kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
										padding="valid")]
		if(maxpool):
			self.left_layers += [tf.keras.layers.MaxPool3D(pool_size=maxpool_k, strides=maxpool_s)]
		if(batch_norm):
			self.left_layers += [tf.keras.layers.BatchNormalization()]
		self.left_layers += [tf.keras.layers.Dense(1)]

		self.left_layers += [operation(filters=n_channels, kernel_size=3, strides=1,
										kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
										bias_regularizer=tf.keras.regularizers.L2(weight_decay),
										kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
										padding="same")]
		if(batch_norm):
			self.left_layers += [tf.keras.layers.BatchNormalization()]
		self.left_layers += [tf.keras.layers.Dense(1)]


		self.right_layers += [operation(filters=n_channels, kernel_size=kernel_size, strides=stride_size,
											kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
											bias_regularizer=tf.keras.regularizers.L2(weight_decay),
											kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
											padding="valid")]
		if(maxpool):
			self.right_layers += [tf.keras.layers.MaxPool3D(pool_size=maxpool_k, strides=maxpool_s)]
		if(batch_norm):
			self.right_layers += [tf.keras.layers.BatchNormalization()]
		
		self.join_layers += [tf.keras.layers.Add()]
		self.join_layers += [tf.keras.layers.Dense(1)]


	def call(self, x):

		self.left_activations = []
		self.right_activations = []
		self.join_activations = []

		#left pass
		z_left = self.left_layers[0](x)
		self.left_activations += [z_left]
		for layer in range(1, len(self.left_layers)):
			z_left = self.left_layers[layer](z_left)
			self.left_activations += [z_left]

		#right pass
		z_right = self.right_layers[0](x)
		self.right_activations += [z_right]
		for layer in range(1, len(self.right_layers)):
			z_right = self.right_layers[layer](z_right)
			self.right_activations += [z_right]

		#join pass
		z = self.join_layers[0]([z_left, z_right])
		self.join_activations += [z]
		z = self.join_layers[1](z)
		self.join_activations += [z]
		
		return z

	def lrp(self, x, y):
		R_join = [None]*(len(self.join_layers)-1) + \
						[y]
		R_left = [None]*(len(self.left_layers))
		R_right = [None]*(len(self.right_layers))

		#begin with join block
		for layer in range(len(self.join_layers)-1)[::-1]:
			with tf.GradientTape(watch_accessed_variables=False) as tape:
				tape.watch(self.join_activations[layer])
				z = self.join_layers[layer+1](self.join_activations[layer])+1e-9
				
				if(z.shape != R_join[layer+1].shape):
					z = tf.flatten(z)

				s = R_join[layer+1]/z
				s = tf.reshape(s, z.shape)
				c = tape.gradient(tf.reduce_sum(z*s), self.join_activations[layer])
				R_join[layer] = self.join_activations[layer]*c

		R_left[-1] = R_join[0]
		R_right[-1] = R_join[0]
		
		#begin with join block
		for layer in range(1, len(self.left_layers)-1)[::-1]:
			if("batch" in self.left_layers[layer].name):
				R_left[layer] = R_left[layer+1]
				continue
			with tf.GradientTape(watch_accessed_variables=False) as tape:
				tape.watch(self.left_activations[layer])
				z = self.left_layers[layer+1](self.left_activations[layer])+1e-9
				
				if(z.shape != R_left[layer+1].shape):
					z = tf.flatten(z)

				s = R_left[layer+1]/z
				s = tf.reshape(s, z.shape)
				c = tape.gradient(tf.reduce_sum(z*s), self.left_activations[layer])
				R_left[layer] = self.left_activations[layer]*c

		#begin with join block
		for layer in range(1, len(self.right_layers)-1)[::-1]:
			if("batch" in self.right_layers[layer].name):
				R_left[layer] = R_left[layer+1]
				continue
			with tf.GradientTape(watch_accessed_variables=False) as tape:
				tape.watch(self.right_activations[layer])
				z = self.right_layers[layer+1](self.right_activations[layer])+1e-9
				
				if(z.shape != R_right[layer+1].shape):
					z = tf.flatten(z)

				s = R_right[layer+1]/z
				s = tf.reshape(s, z.shape)
				c = tape.gradient(tf.reduce_sum(z*s), self.right_activations[layer])
				R_right[layer] = self.right_activations[layer]*c
		
		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(x)
			
			z = self.left_layers[0](x)+1e-9
			s = R_left[1]/tf.reshape(z, R_left[1].shape)
			s = tf.reshape(s, z.shape)

			c = tape.gradient(tf.reduce_sum(z*s), x)
			R_left[0] = x*c

		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(x)
			
			z = self.right_layers[0](x)+1e-9
			s = R_right[1]/tf.reshape(z, R_right[1].shape)
			s = tf.reshape(s, z.shape)

			c = tape.gradient(tf.reduce_sum(z*s), x)
			R_right[0] = x*c

		
		#sum of the modulos, this breaks negative feature importance
		return R_left[0]+R_right[0]