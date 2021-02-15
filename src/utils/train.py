import tensorflow as tf

import gc

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
            loss += tf.reduce_mean(loss_fn(batch_x[1], model(batch_x[0]))).numpy()
        else:
            loss += tf.reduce_mean(loss_fn(batch_x, model(batch_x))).numpy()
        n_batches += 1
    
    return loss/n_batches


def train(train_set, model, opt, loss_fn, epochs=10, val_set=None, file_output=None, verbose=False):
    val_loss = []
    train_loss = []

    for epoch in range(epochs):

        loss = 0.0
        n_batches = 0
        
        for batch_set in train_set.repeat(1):
            loss += train_step(model, batch_set, opt, loss_fn).numpy()
            n_batches += 1
            gc.collect()

        if(val_set is not None):
            val_loss.append(evaluate(val_set, model, loss_fn))
        train_loss.append(loss/n_batches)

        print_utils.print_message("Epoch " + str(epoch+1) + " with loss: " + str(train_loss[-1]), file_output=file_output, verbose=verbose)

    return train_loss, val_loss