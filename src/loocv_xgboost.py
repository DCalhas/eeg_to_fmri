import tensorflow as tf

import argparse

import xgboost

from utils import data_utils, preprocess_data, tf_config, train, losses_utils, lrp, viz_utils, fmri_utils

from layers import topographical_attention

from models import classifiers, eeg_to_fmri

from regularizers import path_sgd

memory_limit=3000
interval_eeg=10

tf_config.set_seed(seed=3)#02 20
tf_config.setup_tensorflow(device="GPU", memory_limit=memory_limit, run_eagerly=True)

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

import pickle

dataset="10"
dataset_synthesis="01"
raw_eeg=False

if(dataset=="10"):
	n_individuals=43

parser = argparse.ArgumentParser()
parser.add_argument('fold',
					choices=['metrics', 'residues', 'uncertainty', 'mean_residues', 'quality', 'attention_graph', 'mean_attention_graph', 'lrp_eeg_channels', 'lrp_eeg_fmri'],
					help="Fold to run")
opt = parser.parse_args()
fold=opt.fold

dataset_clf_wrapper = preprocess_data.Dataset_CLF_CV(dataset, standardize_eeg=True, f_resample=2, eeg_limit=True, eeg_f_limit_h=135, eeg_f_limit_l=0, load=True)
train_data, test_data = dataset_clf_wrapper.split(fold)
X_train, y_train=train_data
X_test, y_test=test_data

network="/home/ist_davidcalhas/eeg_to_fmri/networks/deterministic"

learning_rate=1e-2
optimizer=tf.keras.optimizers.Adam(learning_rate)
activation=tf.keras.activations.linear
variational=True
aleatoric=False
train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train[:,1])).shuffle(X_train.shape[0]).batch(8)
test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test[:,1])).batch(1)

linearCLF=classifiers.ViewLatentLikelihoodClassifier(path_network=network, input_shape=X_train.shape[1:], activation=activation, regularizer=tf.keras.regularizers.L2(l=2.), variational=variational, aleatoric=aleatoric)
loss_fn=losses_utils.SeparationEntropyLoss()

linearCLF.build(X_train.shape)
#train neural network
train.train(train_set, linearCLF, optimizer, loss_fn, epochs=10, val_set=None, u_architecture=False, verbose=True, verbose_batch=True)

#train xgboost
#transform data of fMRI predicted views
X_test_views=np.empty((0,)+linearCLF.view.q_decoder.output_shape[1:], dtype=np.float32)
X_train_views=np.empty((0,)+linearCLF.view.q_decoder.output_shape[1:], dtype=np.float32)
for x,y in test_set.repeat(1):
	X_test_views=np.append(X_test_views, linearCLF.view.q_decoder(x), axis=0)
for x,y in train_set.repeat(1):
	X_train_views=np.append(X_train_views, linearCLF.view.q_decoder(x), axis=0)
#flatten representations
X_test_views=np.reshape(X_test_views, (X_test_views.shape[0], X_test_views.shape[1]*X_test_views.shape[2]*X_test_views.shape[3]))
X_train_views=np.reshape(X_train_views, (X_train_views.shape[0], X_train_views.shape[1]*X_train_views.shape[2]*X_train_views.shape[3]))

xgbclf=xgboost.XGBClassifier()
xgbclf.fit(X_train_views, y_train[:,1:2])
y_pred=xgbclf.predict(X_test_views)

y_true=y_test[:,1]
y_pred=y_pred.astype(np.float32)
#write results
save_file="loocv_xgboost.pickle"
with open(save_file, "rb") as f:
    results=pickle.load(f)
y_pred_list, y_true_list=results
with open(save_file, "wb") as f:
    pickle.dump((y_pred_list+list(y_pred), y_true_list+list(y_true)), f)