import tensorflow as tf

from models.eeg_to_fmri import pretrained_EEG_to_fMRI

class LinearClassifier(tf.keras.Model):
    """
    
    
    """
    def __init__(self, n_classes=2, regularizer=None):
        super(LinearClassifier, self).__init__()

        self.training=True
        
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
    def __init__(self, model, input_shape, activation=None, regularizer=None, feature_selection=False, segmentation_mask=False, seed=None):
        super(view_EEG_classifier, self).__init__()

        self.training=True
        
        self.view = pretrained_EEG_to_fMRI(model, input_shape, activation=activation, regularizer=regularizer, feature_selection=feature_selection, segmentation_mask=segmentation_mask, seed=seed)
        self.clf = LinearClassifier(regularizer=regularizer)

    def build(self, input_shape):
        self.view.build(input_shape)
        self.clf.build(self.view.decoder.output_shape)
    
    def call(self, X):
        z = self.view(X)

        if(self.training):
            return [self.clf(z[0])]+z
        return self.clf(z[0])
        