import tensorflow as tf

import gc

import numpy as np

from utils import print_utils

def apply_gradient(model, optimizer, loss_fn, x, y):
    with tf.GradientTape() as tape:
        loss = loss_fn(y, model(x))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return tf.reduce_mean(loss)

def train_step(model, x, optimizer, loss_fn):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    if(tf.is_tensor(x)):
        return apply_gradient(model, optimizer, loss_fn, x, x)
    else:
        return apply_gradient(model, optimizer, loss_fn, *x)

def evaluate(X, model, loss_fn):
    loss = 0.0
    n_batches = 0
    for batch_x in X.repeat(1):
        if(type(batch_x) is tuple):
            loss += tf.reduce_mean(loss_fn(batch_x[1], model(batch_x[0], training=False))).numpy()
        else:
            loss += tf.reduce_mean(loss_fn(batch_x, model(batch_x, training=False))).numpy()
        n_batches += 1
    
    return loss/n_batches

def evaluate_parameters(X, model):
    parameters = [0.0, 0.0]
    n_batches = 0
    for batch_x in X.repeat(1):
        if(type(batch_x) is tuple):
            prediction = model(batch_x[0], training=False)
        else:
            prediction = model(batch_x, training=False)
        
        parameters[0] += tf.reduce_mean(prediction[1]).numpy()
        if(len(prediction) > 2):
            parameters[1] += tf.reduce_mean(prediction[2]).numpy()

        n_batches += 1
    
    return (parameters[0]/n_batches, parameters[1]/n_batches)

def evaluate_l2loss(X, model):
    l2loss = 0.0
    n_batches = 0
    for batch_x in X.repeat(1):
        if(type(batch_x) is tuple):
            prediction = model(batch_x[0], training=False)
        else:
            prediction = model(batch_x, training=False)
        
        l2loss += tf.reduce_mean((batch_x - prediction[0])**2).numpy()
        
        n_batches += 1
    
    return l2loss/n_batches


def evaluate_additional(X, model, additional_losses):
    losses = np.zeros(len(additional_losses))
    n_batches = 0
    for batch_x in X.repeat(1):
        for loss_fn in additional_losses:
            if(type(batch_x) is tuple):
                prediction = model(batch_x[0], training=False)
            else:
                prediction = model(batch_x, training=False)
            
            losses[i] += tf.reduce_mean(loss_fn(batch_x, prediction[0])).numpy()
            
            n_batches += 1
        
    return (losses/n_batches).tolist()

def train(train_set, model, opt, loss_fn, epochs=10, val_set=None, additional_losses=[], file_output=None, verbose=False, verbose_batch=False):
    val_loss = []
    train_loss = []

    parameters_history = []
    l2loss_history = []
    additional_losses_history = []

    for epoch in range(epochs):

        loss = 0.0
        n_batches = 0
        
        for batch_set in train_set.repeat(1):
            batch_loss = train_step(model, batch_set, opt, loss_fn).numpy()
            loss += batch_loss
            n_batches += 1
            gc.collect()

            print_utils.print_message("Batch ... with loss: " + str(batch_loss), file_output=file_output, verbose=verbose_batch)

        if(val_set is not None):
            val_loss.append(evaluate(val_set, model, loss_fn))
            parameters_history.append(evaluate_parameters(val_set, model))
            l2loss_history.append(evaluate_l2loss(val_set, model))
            additional_losses_history.append(evaluate_additional(val_set, model, additional_losses))

        train_loss.append(loss/n_batches)

        print_utils.print_message("Epoch " + str(epoch+1) + " with loss: " + str(train_loss[-1]), file_output=file_output, verbose=verbose)

    return train_loss, val_loss, parameters_history, l2loss_history, additional_losses_history