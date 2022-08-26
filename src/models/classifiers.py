import tensorflow as tf

import tensorflow_probability as tfp

from models.eeg_to_fmri import pretrained_EEG_to_fMRI

from layers.bayesian import DenseVariational

class LinearClassifier(tf.keras.Model):
    """
    
    
    """
    def __init__(self, n_classes=2, regularizer=None, variational=False):
        super(LinearClassifier, self).__init__()

        self.training=True
        
        self.flatten = tf.keras.layers.Flatten()
        if(variational):
            self.linear = DenseVariational(n_classes)
        else:
            self.linear = tf.keras.layers.Dense(n_classes, kernel_regularizer=regularizer)
        
    def call(self, X):
        return self.linear(self.flatten(X))


class PolynomialClassifier(tf.keras.Model):
    """
    
    """

    def __init__(self, n_classes=2, degree=3, regularizer=None, variational=False):
        super(PolynomialClassifier, self).__init__()

        self.training=True
        self.degree=degree
        
        self.flatten = tf.keras.layers.Flatten()

        if(variational):
            self.linear = DenseVariational(n_classes, use_bias=False)
        else:
            self.linear = tf.keras.layers.Dense(n_classes, use_bias=False, kernel_regularizer=regularizer)
        
    def call(self, X):
        X = tf.expand_dims(X, -1)
        x = [X**0, X]

        for p in range(self.degree-1):
            x+=[x[-1]*X]

        return self.linear(self.flatten(tf.concat(x, -1)))



class ViewClassifier(tf.keras.Model):
    """
    classifier of synthesized EEG view

    """
    
    def __init__(self, model, input_shape, degree=1, activation=None, regularizer=None, feature_selection=False, segmentation_mask=False, variational=False, seed=None):
        """
        Inputs:
            - EEG_to_fMRI: model
            - tupel: input_shape, eeg input shape
        """
        super(ViewClassifier, self).__init__()

        self.training=True
        
        self.view = pretrained_EEG_to_fMRI(model, input_shape, activation=activation, regularizer=regularizer, feature_selection=feature_selection, segmentation_mask=segmentation_mask, seed=seed)
        if(degree==1):
            self.clf = LinearClassifier(variational=variational)
        else:
            self.clf = PolynomialClassifier(degree=degree, variational=variational)
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


class Contrastive(tf.keras.Model):
        
    def __init__(self, model, input_shape, dimension, degree=1, activation=None, regularizer=None, feature_selection=False, segmentation_mask=False, variational=False, seed=None):

        super(Contrastive, self).__init__()

        self.view=pretrained_EEG_to_fMRI(model, input_shape, activation=activation, regularizer=regularizer, feature_selection=feature_selection, segmentation_mask=segmentation_mask, seed=seed).eeg_encoder
        
        self.flatten = tf.keras.layers.Flatten()
        
        if(variational):
            self.linear = DenseVariational(dimension)
        else:
            self.linear = tf.keras.layers.Dense(dimension, kernel_regularizer=regularizer)

        self.dot = tf.keras.layers.Dot(axes=1, normalize=True)

    def build(self, input_shape):
        self.view.build(input_shape)

    def call(self, X, training=False):

        x=tf.split(X, 2, axis=1)
        x1, x2=(tf.squeeze(x[0], axis=1), tf.squeeze(x[1], axis=1))

        z1 = self.view(x1)
        z2 = self.view(x2)

        s1=self.flatten(z1)
        s1=self.linear(s1)

        s2=self.flatten(z2)
        s2=self.linear(s2)

        return 1.-self.dot([s1,s2])

class ViewContrastiveClassifier(tf.keras.Model):

    def __init__(self, model, input_shape, dimension, latent_clf=False, degree=1, activation=None, regularizer=None, feature_selection=False, segmentation_mask=False, variational=False, seed=None):

        super(ViewContrastiveClassifier, self).__init__()

        self.latent_clf=latent_clf

        self.view=pretrained_EEG_to_fMRI(model, input_shape, activation=activation, regularizer=regularizer, feature_selection=feature_selection, segmentation_mask=segmentation_mask, seed=seed)
        if(self.latent_clf):
            self.view=self.view.eeg_encoder

        if(degree==1):
            self.clf = LinearClassifier(variational=variational)
        else:
            self.clf = PolynomialClassifier(degree=degree, variational=variational)

        self.flatten = tf.keras.layers.Flatten()
        self.linear = tf.keras.layers.Dense(dimension, kernel_regularizer=regularizer)

        self.dot = tf.keras.layers.Dot(axes=1, normalize=True)

    def build(self, input_shape):
        self.view.build(input_shape)
        if(not self.latent_clf):
            if(self.view.aleatoric):
                self.clf.build(self.view.q_decoder.output_shape[:-1]+(2,))#additional dimension for aleatoric uncertainty
            else:
                self.clf.build(self.view.q_decoder.output_shape)
        else:
            self.clf.build(self.view.output_shape)

    def call(self, X, training=False):
        if(training):
            x=tf.split(X, 2, axis=1)
            x1, x2=(tf.squeeze(x[0], axis=1), tf.squeeze(x[1], axis=1))

            z1 = self.view(x1)
            z2 = self.view(x2)

            if(not self.latent_clf):
                z1,z2=(z1[0],z2[0])
            
            s1=self.flatten(z1)
            s1=self.linear(s1)

            s2=self.flatten(z2)
            s2=self.linear(s2)

            return [1.-self.dot([s1,s2]), self.clf(z1), self.clf(z2)]

        z=self.view(X)
        if(not self.latent_clf):
            z=z[0]
        return self.clf(z)



class ViewLatentContrastiveClassifier(tf.keras.Model):

    def __init__(self, model, input_shape, degree=1, activation=None, regularizer=None, feature_selection=False, segmentation_mask=False, siamese_projection=False, siamese_projection_dimension=10, variational=False, seed=None):

        super(ViewLatentContrastiveClassifier, self).__init__()

        self.siamese_projection=siamese_projection

        self.view=pretrained_EEG_to_fMRI(model, input_shape, activation=activation, regularizer=regularizer, feature_selection=feature_selection, segmentation_mask=segmentation_mask, latent_contrastive=True, seed=seed)
        
        if(degree==1):
            self.clf = LinearClassifier(variational=variational)
        else:
            self.clf = PolynomialClassifier(degree=degree, variational=variational)

        self.flatten = tf.keras.layers.Flatten()
        
        self.dot = tf.keras.layers.Dot(axes=1, normalize=True)

    def build(self, input_shape):
        self.view.build(input_shape)
        if(self.view.aleatoric):
            self.clf.build(self.view.q_decoder.output_shape[:-1]+(2,))#additional dimension for aleatoric uncertainty
        else:
            self.clf.build(self.view.q_decoder.output_shape)


    def call(self, X, training=False):
        if(training):
            x=tf.split(X, 2, axis=1)
            x1, x2=(tf.squeeze(x[0], axis=1), tf.squeeze(x[1], axis=1))

            z1 = self.view(x1)#returns a list of [fmri view, latent_eeg]
            z2 = self.view(x2)

            s1=self.flatten(z1[1])
            s2=self.flatten(z2[1])

            if(self.siamese_projection):
                ss1=self.flatten(z1[0])
                ss2=self.flatten(z2[0])
                similarity=self.dot([ss1,ss2])
                return [1.-self.dot([s1,s2])+1.-similarity, self.clf(z1[0]), self.clf(z2[0])]

            return [1.-self.dot([s1,s2]), self.clf(z1[0]), self.clf(z2[0])]

        return self.clf(self.view(X)[0])