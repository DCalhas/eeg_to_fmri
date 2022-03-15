import tensorflow as tf

from models.eeg_to_fmri import pretrained_EEG_to_fMRI

class LinearClassifier(tf.keras.Model):
	"""
	
	
	"""
	def __init__(self, n_classes=2, regularizer=None):
		super(LinearClassifier, self).__init__()
		
		self.flatten = tf.keras.layers.Flatten()
		self.linear = tf.keras.layers.Dense(n_classes, kernel_regularizer=regularizer)
		
	def call(self, X):
		return self.linear(self.flatten(X))


"""
classifier of synthesized EEG view

"""

class view_EEG_classifier(tf.keras.Model):

    """
    Inputs:
        - EEG_to_fMRI: model
        - tupel: input_shape, eeg input shape
    """
    def __init__(self, model, input_shape, activation=None, regularizer=None, seed=None):
        super(view_EEG_classifier, self).__init__()
        
        self.view = pretrained_EEG_to_fMRI(model, input_shape, activation=activation, regularizer=regularizer, seed=seed)
        self.clf = LinearClassifier(regularizer=regularizer)
        self.training=True
    
    def build(self, input_shape):
        self.view.build(input_shape)
        self.clf.build(self.view.decoder.output_shape)
    
    def call(self, X):
        z = self.view(X)

        if(self.training):
            return [self.clf(z[1])]+z
        return self.clf(z[1])
        