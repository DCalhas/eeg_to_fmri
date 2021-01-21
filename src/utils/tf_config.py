import tensorflow as tf

def setup_tensorflow(memory_limit):
	gpu = tf.config.experimental.list_physical_devices("GPU")[0]
	tf.config.set_soft_device_placement(True)
	tf.config.log_device_placement=True
	tf.config.experimental.set_memory_growth(gpu, True)
	tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])