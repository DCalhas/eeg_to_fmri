import tensorflow as tf

"""
Loss combinating aleatoric and epistemic_uncertainty
"""
def combined_loss(y_true, y_pred):    
    D = y_true.shape[1]*y_true.shape[2]*y_true.shape[3]   
    
    variance = tf.math.square(y_pred[1])
    
    return (1/D)* tf.reduce_sum((tf.exp(-tf.math.log(variance))*(y_pred[0] - y_true)**2)/2 + (tf.math.log(variance))/2)

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