"""GPU check (seconds): posthoc.bn_update, compiled with per-step momentum, ends at the exact average of the batch statistics."""
import sys
import numpy as np
from utils import process_utils
process_utils.process_setup_tensorflow(2000, seed=0)
import tensorflow as tf
import posthoc

i = tf.keras.Input((5,)); bn = tf.keras.layers.BatchNormalization()
h = tf.keras.layers.Dense(4)(i); m = tf.keras.Model(i, bn(h))
class Holder: pass
model = Holder(); model.decoder = m
eeg = np.random.RandomState(0).normal(3, 2, size=(160, 5)).astype("float32")
posthoc.bn_update(model, eeg, np.random.RandomState(1), n_batches=10, batch=16)
idx = np.random.RandomState(1).choice(160, size=160, replace=False)
pre = m.layers[1](eeg).numpy()
expected = np.mean([pre[np.sort(idx[s:s+16])].mean(0) for s in range(0, 160, 16)], 0)
ok = np.allclose(bn.moving_mean.numpy(), expected, atol=1e-4) and abs(bn.momentum-0.99) < 1e-9
print("moving_mean", np.round(bn.moving_mean.numpy(), 4), "expected", np.round(expected, 4), "momentum", bn.momentum)
print("BN_UPDATE_OK" if ok else "BN_UPDATE_FAILED")
sys.exit(0 if ok else 1)
