import tensorflow as tf

import tensorflow_probability as tfp

import os

import numpy as np

import random

from utils import memory_utils

def setup_tensorflow(memory_limit, device="CPU", run_eagerly=False, set_primary_memory=True):
	gpu = tf.config.experimental.list_physical_devices(device)[0]
	tf.config.set_soft_device_placement(True)
	tf.config.log_device_placement=True
	if(device=="GPU"):
		tf.config.experimental.set_memory_growth(gpu, True)
		tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])

	if(run_eagerly):
		tf.config.run_functions_eagerly(True)

	if(set_primary_memory):
		memory_utils.limit_CPU_memory(1024*1024*1024*8)#4GB

def set_seed(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)
	os.environ['TF_DETERMINISTIC_OPS'] = '1'