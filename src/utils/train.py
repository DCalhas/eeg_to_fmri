import tensorflow as tf

import gc


def train_step(model, x, optimizer, loss_fn):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    @tf.function
    def apply_gradient(model, x, optimizer, loss_fn):
        with tf.GradientTape() as tape:
            loss = loss_fn(x, model(x))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return tf.reduce_mean(loss)

    return apply_gradient(model, x, optimizer, loss_fn)

def evaluate(X, model, loss_fn):
    loss = 0.0
    n_batches = 0
    for batch_x in X.repeat(1):
        loss += tf.reduce_mean(loss_fn(batch_x, model(batch_x))).numpy()
        n_batches += 1
    
    return loss/n_batches


def train(X_train, model, opt, loss_fn, epochs=10, X_val=None, verbose=False):
    val_loss = []
    train_loss = []

    for epoch in range(epochs):

        loss = 0.0
        n_batches = 0
        
        for batch_x in X_train.repeat(1):
            loss += train_step(model, batch_x, opt, loss_fn).numpy()
            n_batches += 1
            gc.collect()

        val_loss.append(evaluate(X_val, model, loss_fn))
        train_loss.append(loss/n_batches)

        print("Epoch ", epoch+1, " with loss: ", train_loss[-1], flush=True)
        
    return train_loss, val_loss