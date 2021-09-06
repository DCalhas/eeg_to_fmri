import GPyOpt

import argparse

import gc

from utils import process_utils

import multiprocessing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('setup', choices=['latent_fmri'], help="Which setuo to run")
    parser.add_argument('dataset', choices=['01', '02'], help="Which dataset to load")
    parser.add_argument('-gpu_mem', default=4000, type=int, help="GPU memory limit")
    opt = parser.parse_args()

    setup=opt.setup
    memory_limit=opt.gpu_mem
    dataset=opt.dataset

raw_eeg=False#time or frequency features? raw-time nonraw-frequency
resampling=False
if(dataset=="01"):
    n_volumes=300-3
if(dataset=="02"):
    n_volumes=170-3
n_individuals=10
n_individuals_train=8
#parametrize the interval eeg?
interval_eeg=6

load_data_args = (dataset, n_individuals, n_individuals_train, n_volumes, interval_eeg, memory_limit)

def latent_fmri(theta):
    score = multiprocessing.Value('d', 0)

    #unroll hyperparameters
    learning_rate=float(theta[:, 0])
    weight_decay = float(theta[:, 1])
    batch_size=int(theta[:, 2])
    latent_dimension=(int(theta[:, 3]),int(theta[:, 3]),int(theta[:, 3]))
    n_channels=int(theta[:, 4])
    max_pool=bool(theta[:, 5])
    batch_norm=bool(theta[:, 6])
    skip_connections=bool(theta[:, 7])
    n_stacks=int(theta[:, 8])
    outfilter=int(theta[:, 9])
    
    n_stacks=3
    #n_channels=16
    #batch_size=16
    #latent_dimension=(5,5,5)

    kernel_size = (9,9,4)#fixed
    stride_size = (1,1,1)#fixed
    local=True#fixed

    cross_val_args = (learning_rate, weight_decay, 
                        kernel_size, stride_size,
                        batch_size, latent_dimension,
                        n_channels, max_pool, 
                        batch_norm, skip_connections, 
                        n_stacks,outfilter)

    process_utils.launch_process(getattr(process_utils, "cross_validation_"+setup), (score,) + cross_val_args+load_data_args)

    return score.value


if __name__ == "__main__":

    theta_space = getattr(process_utils, "theta_"+setup)()

    optimizer = GPyOpt.methods.BayesianOptimization(f=globals()[setup], 
                                                    domain=theta_space, 
                                                    model_type="GP_MCMC", 
                                                    acquisition_type="EI_MCMC")

    print("Started Optimization Process")
    optimizer.run_optimization(max_iter=1)

    print(optimizer.fx_opt)
    optimized_hyperparameters = optimizer.x_opt