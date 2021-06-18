import sys
import argparse
import gc
import os
import time

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from utils import tf_config, preprocess_data, search_algorithms, train, bnn_utils, outlier_utils, viz_utils
from models import fmri_ae, eeg_to_fmri, uniconv_fmri, bnn_fmri_ae
from layers import locally_connected

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold

losses=["combined_original_loss", "combined_log_loss", 
        "combined_abs_diff_loss", "combined_abs_non_balanced_loss", 
        "combined_abs_balanced_loss", "gamma_prior_loss"]
parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['01', '02'], help="Which dataset to load")
parser.add_argument('loss_fn', choices=losses, 
                                help="Which loss function to use in training session.")
parser.add_argument('-gpu_mem', default=4000, type=int, help="GPU memory limit")
parser.add_argument('-batch_size', default=1, type=int, help="Batch size to use in training.")
parser.add_argument('-learning_rate', default=0.001, type=float, help="Learning rate for training session.")
parser.add_argument('-epochs', default=10, type=int, help="Number epochs for training session.")
parser.add_argument('-n_channels', default=4, type=int, help="Number of channels for the convolutional layers.")
parser.add_argument('-latent_dimension', default=5, type=int, help="Latent dimension specification (3D, repeat integer provided).")
parser.add_argument('-kernel_size_x', default=9, type=int, help="Kernel size of 1st dimension.")
parser.add_argument('-kernel_size_y', default=9, type=int, help="Kernel size of 2nd dimension.")
parser.add_argument('-kernel_size_z', default=4, type=int, help="Kernel size of 3rd dimension.")
parser.add_argument('-stride_size_x', default=1, type=int, help="Stride size of 1st dimension.")
parser.add_argument('-stride_size_y', default=1, type=int, help="Stride size of 2nd dimension.")
parser.add_argument('-stride_size_z', default=1, type=int, help="Stride size of 3rd dimension.")
parser.add_argument('-n_stacks', default=3, type=int, help="Number of Resnet stacks to consider until reaching latent dimension.")
parser.add_argument('-skip_connections', action="store_true", help="Include skip connections.")
parser.add_argument('-maxpool', action="store_true", help="Include maxpool layers.")
parser.add_argument('-batch_norm', action="store_true", help="Include Batch Normalization layers.")
parser.add_argument('-MAP', action="store_true", help="Maximum a posteriori estimation, standard is maximum likelihood estimation of parameter sigma^2.")
parser.add_argument('-local_attention', action="store_true", help="Local attention at the latent dimension.")
parser.add_argument('-local', action="store_true", help="Use of convolutions or locally connected layers for downsampling.")
parser.add_argument('-outfilter', default=0, type=int, help="0: means no filter at the end; 1:conv 1x1 filter variational layer; 2: locally connected 1x1 filter variational layer.")
parser.add_argument('-interval_eeg', default=6, type=int, help="EEG signal shift relative to fMRI, studies show it is around 5/6 seconds.")
parser.add_argument('-n_individuals', default=8, type=int, help="Number of individuals to consider from the dataset provided.")
parser.add_argument('-seed', default=42, type=int, help="Seed for random generation.")
parser.add_argument('-iqr', action="store_true", help="Perform IQR preprocessing to remove out of distribution data.")
parser.add_argument('-out_file', default="out.txt", type=str, help="Output file")
opt = parser.parse_args()

dataset = opt.dataset
loss_fn = getattr(bnn_utils, opt.loss_fn)
loss_index=losses.index(opt.loss_fn)
gpu_mem = opt.gpu_mem
batch_size = opt.batch_size
learning_rate = opt.learning_rate
epochs = opt.epochs
n_channels = opt.n_channels
latent_dimension=(opt.latent_dimension,opt.latent_dimension,opt.latent_dimension)
kernel_size=(opt.kernel_size_x,opt.kernel_size_y,opt.kernel_size_z)
stride_size=(opt.stride_size_x,opt.stride_size_y,opt.stride_size_z)
n_stacks=opt.n_stacks
skip_connections=opt.skip_connections
maxpool=opt.maxpool
batch_norm=opt.batch_norm
MAP=opt.MAP
local_attention=opt.local_attention
local=opt.local
outfilter=opt.outfilter
interval_eeg = opt.interval_eeg
n_individuals = opt.n_individuals
seed = opt.seed
iqr = opt.iqr
out_file = opt.out_file

"""
Define setting to create directory
"""
if(MAP):
    if(iqr):
        setting=dataset+"_"+loss_fn.__name__+"_lr_"+str(learning_rate)+"_MAP_"+"_local_attention_"+str(int(local_attention))+"_local_"+str(int(local))+"_outfilter_"+str(int(outfilter))+"_iqr"
    else:
        setting=dataset+"_"+loss_fn.__name__+"_lr_"+str(learning_rate)+"_MAP_"+"_local_attention_"+str(int(local_attention))+"_local_"+str(int(local))+"_outfilter_"+str(int(outfilter))
else:
    if(iqr):
        setting=dataset+"_"+loss_fn.__name__+"_lr_"+str(learning_rate)+"_local_attention_"+str(int(local_attention))+"_local_"+str(int(local))+"_outfilter_"+str(int(outfilter))+"_iqr"
    else:
        setting=dataset+"_"+loss_fn.__name__+"_lr_"+str(learning_rate)+"_local_attention_"+str(int(local_attention))+"_local_"+str(int(local))+"_outfilter_"+str(int(outfilter))

tf_config.set_seed(seed=seed)
tf_config.setup_tensorflow(device="GPU", memory_limit=gpu_mem)


print("I: Loading data...")
"""
Load data
"""
with tf.device('/CPU:0'):
    train_data, val_data = preprocess_data.dataset(dataset, n_individuals=n_individuals, 
                                            interval_eeg=interval_eeg, 
                                            ind_volume_fit=False,
                                            standardize_fmri=True,
                                            iqr=False,
                                            verbose=True)
    _, train_x=train_data
    _, val_x=val_data
    
    train_x = train_x.astype('float32')
    val_x = val_x.astype('float32')

print("I: Finished loading data")
print("I: Starting IQR Preprocesing...")

"""
IQR preprocessing of outliers
"""
if(iqr):
    iqr = outlier_utils.IQR()
    iqr.fit(train_x)
    train_x = iqr.transform(train_x, channels_last=False)

print("I: Finished IQR Preprocesing")

"""
Specification of data
"""
optimizer = tf.keras.optimizers.Adam(learning_rate)
train_set = tf.data.Dataset.from_tensor_slices((train_x, train_x)).batch(batch_size)
dev_set = tf.data.Dataset.from_tensor_slices((val_x, val_x)).batch(1)

"""
Setup bayesian model
"""
model = bnn_fmri_ae.create_bayesian_model(train_x.shape[1:], train_x.shape[1:], latent_dimension,
                    kernel_size, stride_size, n_channels,
                    maxpool=maxpool, batch_norm=batch_norm,
                    skip_connections=skip_connections, n_stacks=n_stacks, 
                    local=local, local_attention=local_attention, MAP=MAP, outfilter=outfilter)

print("I: Starting training session...")
"""
Training session
"""
train_loss, val_loss, parameters_history, l2loss_history, additional_losses_history = train.train(train_set, model, optimizer, 
                                                           loss_fn, epochs=epochs, 
                                                           val_set=dev_set, additional_losses=[bnn_utils.epistemic_log_loss, bnn_utils.epistemic_original_loss],
                                                           verbose=True, verbose_batch=False)

print("I: Training session finished")
print("I: Gathering epistemic and aleatoric uncertainty plots...")

"""
Save plots of epistemic and aleatoric uncertainty
"""
for volume in range(val_x.shape[0]):
    viz_utils.plot_epistemic_aleatoric_uncertainty(setting, model, val_x, volume, 30, 30, 15, T=10)


"""
Plot gamma function evolution
"""
if(MAP):
    viz_utils.gamma_epoch_plot(setting, parameters_history, epochs=epochs)

"""
Original loss convergence
"""
losses_history=np.zeros((10,4))
losses_history[:,0]= np.array(val_loss)
losses_history[:,1]= additional_losses_history[:,0]
losses_history[:,2]= additional_losses_history[:,1]
losses_history[:,3]= np.array(l2loss_history)
viz_utils.uncertainty_losses_plot(setting, losses_history, loss_index, epochs=epochs)
