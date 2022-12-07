import tensorflow as tf

OPTIMIZER=tf.keras.optimizers.Adam


def optimizer(name, input_shape, model, lr):
	if(name=="PathAdam"):
		return PathOptimizer(input_shape, model, lr)
	elif(name=="Adam"):
		return tf.keras.optimizers.Adam(lr)
	else:
		raise NotImplementedError


class PathOptimizer(OPTIMIZER):
	"""
	This class implements the tensorflow optimizer proposed in https://arxiv.org/abs/1506.02617

	Example:
	>>> import tensorflow as tf
	>>> 
	>>> model=tf.keras.Sequential([tf.keras.layers.Dense(2), tf.keras.layers.Dense(2)])
	>>> input_shape=(10,1)
	>>> x = tf.keras.initializers.GlorotUniform()(input_shape)
	>>> model.build(input_shape)
	>>> 
	>>> #assert computations of gradients
	>>> with tf.GradientTape() as tape:
	>>> 	tape.watch(model.trainable_variables)
	>>> 	y = model(x)
	>>> gradients=tape.gradient(y,model.trainable_variables)
	>>> 
	>>> #clone model and assign its l1 weights
	>>> path_model=tf.keras.models.clone_model(model)
	>>> for param in range(len(model.trainable_variables)):
	>>> 	path_model.trainable_variables[param].assign(tf.abs(model.trainable_variables[param]))
	>>> 
	>>> #compute scale
	>>> with tf.GradientTape() as tape:
	>>> 	tape.watch(path_model.trainable_variables)
	>>> 	y = tf.reduce_sum(path_model(tf.ones(input_shape)))
	>>> path_norm=tape.gradient(y, path_model.trainable_variables)
	>>> 
	>>> #compute ratio
	>>> sgd_norm=0.
	>>> pathsgd_norm=0.
	>>> model_params = model.trainable_variables
	>>> path_params = model.trainable_variables
	>>> for param in range(len(model_params)):
	>>> 	sgd_norm += tf.norm(gradients[param], ord=1)
	>>> 	pathsgd_norm += tf.norm(gradients[param]/path_norm[param], ord=1)
	>>> ratio = ( sgd_norm / pathsgd_norm ) ** 1
	>>> 
	>>> print("Gradients before:", gradients)
	>>> #gradient update
	>>> for param in range(len(model_params)):
	>>> 	gradients[param]=(gradients[param]/path_norm[param])*ratio
	>>> 
	>>> print("Gradients before:", gradients)
	"""

	def __init__(self, input_shape, model, lr, name="PathOptimizer", p=1, **kwargs):

		self.model=model
		self.path_norm=None
		self.ratio=None
		self.input_shape=input_shape
		self.p=p

		super(PathOptimizer, self).__init__(lr, name=name, **kwargs)


	def apply_gradients(self, grads_and_vars, name=None, **kwargs,):
		"""
		Example: 
		>>> import tensorflow as tf
		>>> from path_sgd import PathOptimizer
		>>> 
		>>> model=tf.keras.Sequential([tf.keras.layers.Dense(2), tf.keras.layers.Dense(2)])
		>>> input_shape=(10,1)
		>>> x = tf.keras.initializers.GlorotUniform()(input_shape)
		>>> model.build(input_shape)
		>>> 
		>>> with tf.GradientTape() as tape:
		>>> 	tape.watch(model.trainable_variables)
		>>> 	y = model(x)
		>>> 
		>>> gradients=tape.gradient(y,model.trainable_variables)
		>>> optimizer=PathOptimizer(input_shape, model, 0.01)
		>>> optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		"""

		self.compute_path_norm()
		
		gradients = list(list(zip(*grads_and_vars))[0])

		if(self.ratio is None or type(self.model).__name__=="ViewLatentContrastiveClassifier"):
			#compute ratio
			sgd_norm=0.
			pathsgd_norm=0.
			for param in range(len(self.model.trainable_variables)):
				sgd_norm += tf.norm(gradients[param], ord=self.p)
				pathsgd_norm += tf.norm(gradients[param]/self.path_norm[param], ord=self.p)
			self.ratio = ( sgd_norm / pathsgd_norm ) ** (1/self.p)

		
		for param in range(len(self.model.trainable_variables)):
			gradients[param]=(gradients[param]/self.path_norm[param])*self.ratio

		return super().apply_gradients(zip(gradients, self.model.trainable_variables), name=name)

	def compute_path_norm(self,):

		#clone model and assign its l1 weights	
		path_model=type(self.model).from_config(self.model.get_config())
		#in the special case of ViewLatentContrastiveClassifier we have to do this, so we do not have two flowsS
		if(type(path_model).__name__=="ViewLatentContrastiveClassifier"):
			path_model.training=False

		input_shape_tensor=None
		#build input
		if(type(self.input_shape) is list):
			input_shape_tensor=tuple(tf.ones(input_shape) for input_shape in self.input_shape)
			path_model.build(*tuple(input_shape for input_shape in self.input_shape))
		else:
			input_shape_tensor=(tf.ones(self.input_shape),)
			path_model.build(self.input_shape)

		for param in range(len(self.model.trainable_variables)):
			if(self.p==1):
				path_model.trainable_variables[param].assign((self.model.trainable_variables[param]**2)**0.5)
			else:
				path_model.trainable_variables[param].assign(self.model.trainable_variables[param]**self.p)

		#compute scale
		with tf.GradientTape() as tape:
			tape.watch(path_model.trainable_variables)
			y = path_model(*input_shape_tensor)
			if(type(y) is list):
				y = tf.reduce_sum([tf.reduce_sum(y_i) for y_i in y])
			else:
				y = tf.reduce_sum(y)

		self.path_norm=tape.gradient(y, path_model.trainable_variables)

		del path_model