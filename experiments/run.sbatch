#!/bin/bash
#SBATCH --partition=dev
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/david.calhas/eeg_to_fmri/uncertainty/logs/slurm-%j.out
# Generic wrapper: sbatch experiments/run.sbatch <command...>, run from the uncertainty repo root.
# TF 2.9 + CUDA 11 pip libs; only A100 (sm_80) is guaranteed to work with this stack.
source ~/eeg_to_fmri/envs/tf29_activate.sh
export EEG_FMRI=/shared/home/david.calhas/eeg_to_fmri/uncertainty/repo
export EEG_FMRI_DATASETS=/shared/home/david.calhas/eeg_to_fmri/data/
cd $EEG_FMRI/src
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$@"
