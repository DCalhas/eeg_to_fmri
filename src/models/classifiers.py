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


class PolynomialClassifier(tf.keras.Model):
    """
    
    """

    def __init__(self, n_classes=2, degree=3, regularizer=None):
        super(PolynomialClassifier, self).__init__()

        self.training=True
        self.degree=degree
        
        self.flatten = tf.keras.layers.Flatten()
        self.linear = tf.keras.layers.Dense(n_classes, use_bias=False, kernel_regularizer=regularizer)
        
    def call(self, X):
        X = tf.expand_dims(X, -1)
        x = [X**0, X]

        for p in range(self.degree-1):
            x+=[x[-1]*X]

        return self.linear(self.flatten(tf.concat(x, -1)))



class view_EEG_classifier(tf.keras.Model):
    """
    classifier of synthesized EEG view

    """
    
    def __init__(self, model, input_shape, activation=None, regularizer=None, feature_selection=False, segmentation_mask=False, seed=None):
        """
        Inputs:
            - EEG_to_fMRI: model
            - tupel: input_shape, eeg input shape
        """
        super(view_EEG_classifier, self).__init__()

        self.training=True
        
        self.view = pretrained_EEG_to_fMRI(model, input_shape, activation=activation, regularizer=regularizer, feature_selection=feature_selection, segmentation_mask=segmentation_mask, seed=seed)
        #self.clf = LinearClassifier()
        self.clf = PolynomialClassifier()
        #sigma layers
        self.flatten=tf.keras.layers.Flatten()
        self.dense=tf.keras.layers.Dense(1, activation=tf.keras.activations.exponential)

    def build(self, input_shape):
        self.view.build(input_shape)
        if(self.view.aleatoric):
            self.clf.build(self.view.q_decoder.output_shape[:-1]+(2,))#additional dimension for aleatoric uncertainty
        else:
            self.clf.build(self.view.q_decoder.output_shape)
    
    def call(self, X):
        z = self.view(X)
        logits=self.clf(z[0])

        sigma_1 = self.dense(self.flatten(logits))

        if(self.training):
            return [logits]+z+[sigma_1]
        return logits
        


class ContrastiveClassifier(tf.keras.Model):

    def __init__(self, model, input_shape, dimension, activation=None, regularizer=None, feature_selection=False, segmentation_mask=False, seed=None):

        self.view=pretrained_EEG_to_fMRI(model, input_shape, activation=activation, regularizer=regularizer, feature_selection=feature_selection, segmentation_mask=segmentation_mask, seed=seed)

        self.flatten1 = tf.keras.layers.Flatten()
        self.linear = tf.keras.layers.Dense(dimension, kernel_regularizer=regularizer)

    def build(self, input_shape):
        self.view.build(input_shape)
        if(self.view.aleatoric):
            self.clf.build(self.view.q_decoder.output_shape[:-1]+(2,))#additional dimension for aleatoric uncertainty
        else:
            self.clf.build(self.view.q_decoder.output_shape)


    def call(self, x1, x2):

        z1 = self.view(x1)[0]
        z2 = self.view(x2)[0]

        z1=self.flatten1(z1)
        z1=self.linear(z1)

        z2=self.flatten1(z2)
        z2=self.linear(z2)

        return (z1**2-z2**2)**(1/2)

