import tensorflow as tf

import GPyOpt

from utils import train, tf_config


class Bayesian_Optimization:
					
	def __init__(self, iterations, model_class, input_shape):
		
		self.iterations = iterations
		self.model_class = model_class
		self.optimizer = tf.keras.optimizers.Adam
		self.input_shape = input_shape
				
	def set_hyperparameters(self, hyperparameters):
		
		self.hyperparameters = hyperparameters
		
	def set_data(self, X_train, X_val=None):
	
		self.X_train = X_train
		self.X_val = X_val
		
	def optimize(self, hyperparameters):
		hyperparameters = (self.input_shape,) + tuple(hyperparameters[0])
		
		with tf.device('/CPU:0'):
			model = self.model_class.build(*hyperparameters)
			X_train = tf.data.Dataset.from_tensor_slices(self.X_train).batch(int(hyperparameters[10]))
			X_val = tf.data.Dataset.from_tensor_slices(self.X_val).batch(1)

		optimizer = self.optimizer(float(hyperparameters[1]))
		loss_fn = tf.keras.losses.MAE

		#train
		train_loss, val_loss = train.train(X_train, model, optimizer, 
										   loss_fn, epochs=int(hyperparameters[9]), 
										   X_val=X_val, verbose=True)

		#optionally plot validation loss to analyze learning curve

		return val_loss[-1]

	def run(self):
		optimizer = GPyOpt.methods.BayesianOptimization(f=self.optimize, 
														domain=self.hyperparameters, 
														model_type="GP_MCMC", 
														acquisition_type="EI_MCMC")

		print("Started Optimization Process")
		optimizer.run_optimization(max_iter=self.iterations)