import tensorflow as tf
import tensorflow_probability as tfp

"""
Loss combinating aleatoric and epistemic_uncertainty
"""
def combined_log_loss(y_true, y_pred):
	variance = tf.math.square(y_pred[1])+1e-9
	
	return tf.reduce_mean((tf.exp(-tf.math.log(variance))*(y_pred[0] - y_true)**2)/2 + (tf.math.log(variance))/2, axis=(1,2,3))

def combined_original_loss(y_true, y_pred):
	variance = tf.math.square(y_pred[1])+1e-9
	
	return tf.reduce_mean(((1/variance)*(y_pred[0] - y_true)**2)/2 + (tf.math.log(variance))/2, axis=(1,2,3))

def combined_square_loss(y_true, y_pred):
	variance = tf.math.square(y_pred[1])
	
	return tf.reduce_mean(((1/(variance+1e-9))*(y_pred[0] - y_true)**2)/2 + variance/2, axis=(1,2,3))

def combined_log_abs_loss(y_true, y_pred):
	variance = tf.math.abs(y_pred[1])
	
	return tf.reduce_mean(((1/variance+1e-9)*(y_pred[0] - y_true)**2)/2 + (tf.math.log(variance))/2, axis=(1,2,3))

def combined_abs_loss(y_true, y_pred):
	variance = tf.math.abs(y_pred[1])
	
	return tf.reduce_mean(((1/variance+1e-9)*(y_pred[0] - y_true)**2)/2 + variance/2, axis=(1,2,3))

def combined_abs_diff_log_loss(y_true, y_pred):
	variance = tf.math.abs(y_pred[1])
	
	return tf.reduce_mean((-variance*(y_pred[0] - y_true)**2)/2 + (tf.math.log(variance))/2, axis=(1,2,3))

def combined_abs_diff_loss(y_true, y_pred):
	variance = tf.math.abs(y_pred[1])
	
	return tf.reduce_mean((-variance*(y_pred[0] - y_true)**2)/2 + variance/2, axis=(1,2,3))

def combined_abs_non_balanced_loss(y_true, y_pred):
	variance = tf.math.abs(y_pred[1])
	
	return tf.reduce_mean(((variance-variance**2)*(y_pred[0] - y_true)**2)/2 + (variance**2)/2, axis=(1,2,3))

def combined_abs_balanced_loss(y_true, y_pred):
	variance = tf.math.abs(y_pred[1])
	
	return tf.reduce_mean(((variance-variance**2)*(y_pred[0] - y_true)**2)/2 + (variance**2-variance)/2, axis=(1,2,3))


class extended_balance:
	def __init__(self, K):
		self.K = K

	def combined_abs_non_balanced_loss(self, y_true, y_pred):
		variance = tf.math.abs(self.K*y_pred[1])
		
		return tf.reduce_mean(((variance-variance**2)*(y_pred[0] - y_true)**2)/2 + (variance**2)/4, axis=(1,2,3))

	def combined_abs_balanced_loss(self, y_true, y_pred):
		variance = tf.math.abs(self.K*y_pred[1])
		
		return tf.reduce_mean(((variance-variance**2)*(y_pred[0] - y_true)**2)/2 + (variance**2-variance)/4, axis=(1,2,3))



"""
Computing \sigma_{i}^{2}
"""
def aleatoric_uncertainty(model, X, T=10):
	
	y_std = tf.zeros(X.shape)
	
	for i in range(T):
		y_t = model(X, training=False, T=T)
		y_std = y_std + tf.math.square(y_t[1])
		
	return y_std/T

"""
Computing Var(y*)
"""
def epistemic_uncertainty(model, X, T=10):
    
    y_square = tf.zeros(X.shape)
    
    for i in range(T):
        y_t = model(X, training=False, T=T)
        
        y_square = y_square + y_t[0]
        
    y_square = - tf.math.square((1/T)*y_square)
    
    y_hat = tf.zeros(X.shape)
    
    for i in range(T):
        y_t = model(X, training=False, T=T)
        
        y_hat = y_hat + tf.math.square(y_t[0]) + tf.math.square(y_t[1])
        
    y_hat = y_square + (1/T)*y_hat
        
    return y_hat