"""Diagnostic: GPU memory across repeated stochastic decoder calls (Flipout evaluation OOM)."""
import argparse, numpy as np, experiment
parser = experiment.add_model_args(argparse.ArgumentParser())
opt = parser.parse_args()
from utils import process_utils
process_utils.process_setup_tensorflow(opt.gpu_mem, seed=0)
import tensorflow as tf
model, _ = experiment.build_model(opt, (None, 63, 134, 10, 1), (None, 64, 64, 30, 1), n_train=3000)
x = tf.random.normal((8, 63, 134, 10, 1))
mb = lambda: tf.config.experimental.get_memory_info("GPU:0")["current"]/2**20
for mode in ["decoder", "decoder_tf_function"]:
	f = model.decoder if mode == "decoder" else tf.function(lambda e: model.decoder(e, training=False))
	print(mode, "start MB", round(mb()), flush=True)
	for i in range(100):
		out = f(x) if mode != "decoder" else f(x, training=False)
		if((i+1) % 20 == 0):
			print(mode, "call", i+1, "current MB", round(mb()), flush=True)
	del out
