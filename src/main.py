import argparse

import numpy as np

from utils import metrics, process_utils


parser = argparse.ArgumentParser()
parser.add_argument('mode',
                    choices=['metrics', 'quality'],
                    help="What to compute")
parser.add_argument('dataset', choices=['01', '02'], help="Which dataset to load")
parser.add_argument('-versbose', action="store_true", help="Verbose")
parser.add_argument('-seed', default=42, type=int, help="Seed for random generator")
parser.add_argument('-gpu_mem', default=4000, type=int, help="GPU memory limit")
opt = parser.parse_args()

mode=opt.mode
dataset=opt.dataset
verbose=opt.verbose
seed=opt.seed
gpu_mem=opt.gpu_mem


#load data
raw_eeg=False#time or frequency features? raw-time nonraw-frequency
resampling=False
if(dataset=="01"):
    n_volumes=300-3
    n_individuals=10
    n_individuals_train=8
if(dataset=="02"):
    n_volumes=170-3
#parametrize the interval eeg?
interval_eeg=6

train_data, test_data = process_utils.load_data_eeg_fmri(dataset, n_individuals, n_volumes, interval_eeg, memory_limit, return_test=True)


if(mode=="metrics"):

	metrics.rmse(test_data, model)
	metrics.ssim(test_data, model)