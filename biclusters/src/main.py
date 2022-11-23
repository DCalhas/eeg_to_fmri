import sys

from pathlib import Path

import os

os.environ["EEG_FMRI"]=str(Path.home())+"/eeg_to_fmri"
os.environ["EEG_FMRI_DATASETS"]="/mnt/datasets"

#to be replaced by eeg_to_fmri package
from eeg_to_fmri.utils import tf_config, viz_utils

from eeg_to_fmri.data import preprocess_data

from eeg_to_fmri.layers import fft

import numpy as np

import tensorflow as tf

import bicpy

import patterns

import to_bicpams

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['ground_truth', 'pred'], help="Which labels to run BicPAMS on, predictions or ground_truth")
parser.add_argument('dataset', choices=['10', '11'], help="Which dataset to work on")
parser.add_argument('-format_file', default="arff", type=str, help="File format to give to BicPAMS")
parser.add_argument('-background_cutoff', action="store_true", help="Build a brain mask in the original fMRI resolution")
parser.add_argument('-background_cutoff_low', action="store_true", help="Build a brain mask in a downsampled fMRI resolution")
parser.add_argument('-resolution', default=(10,10,5), type=tuple, help="Resolution where the brain mask is built")
parser.add_argument('-threshold', default=0.5, type=float, help="Threshold to build the brain mask")
parser.add_argument('-min_biclusters', default=100, type=int, help="Minimum biclusters to stop searching for more")
parser.add_argument('-min_lift', default=1.2, type=float, help="Minimum lift to stop searching for more")
parser.add_argument('-nr_labels', default=5, type=int, help="Number of bins to discretize the data")
parser.add_argument('-min_columns', default=3, type=int, help="Minimum columns to stop searching for more")


opt = parser.parse_args()

dataset=opt.dataset
mode=opt.mode
format_file=opt.format_file
background_cutoff=opt.background_cutoff
background_cutoff_low=opt.background_cutoff_low
threshold=opt.threshold
resolution=opt.resolution
min_biclusters=opt.min_biclusters
min_lift=opt.min_lift
nr_labels=opt.nr_labels
min_columns=opt.min_columns

bicpams_parameters=bicpy.DEFAULT_PARAMS
bicpams_parameters['min_biclusters']=min_biclusters
bicpams_parameters['min_lift']=min_lift
bicpams_parameters['nr_labels']=nr_labels
bicpams_parameters['min_columns']=min_columns

assert not (background_cutoff and background_cutoff_low)

setting=dataset+"_"+mode
if(background_cutoff):
	setting+="_original_masked"
elif(background_cutoff):
	setting+="_low_masked_"+str(resolution[0])+"x"+str(resolution[1])+"x"+str(resolution[2])

#set seed and tensorflow GPU memory
tf_config.set_seed(seed=42)#02 20
tf_config.setup_tensorflow(device="GPU", memory_limit=1500)

#define labels path and view
path_labels="../../metrics/10_synth_01_style_prior_bayesian/"
view="fmri"

#load data
X_view = np.load(path_labels+"views.npy")
y_true = np.load(path_labels+"y_true.npy")
y_pred = np.load(path_labels+"y_pred.npy")

views_downsampled, brain_mask = to_bicpams.downsample(X_view, resolution, threshold=threshold, cutoff=background_cutoff, cutoff_low=background_cutoff_low,)

with open("./view_"+mode+"."+format_file, "w") as f:
	if(mode=="ground_truth"):
		f.write(to_bicpams.build_arff(views_downsampled, y_true, y_pred, resolution, format_file, cutoff_low=background_cutoff_low, brain_mask=brain_mask)[0])	
	elif(mode=="pred"):
		f.write(to_bicpams.build_arff(views_downsampled, y_true, y_pred, resolution, format_file, cutoff_low=background_cutoff_low, brain_mask=brain_mask)[1])
	f.close()

bicpy.run(bicpams_parameters, "./view_"+mode+"."+format_file)