import numpy as np

from pathlib import Path

from os import listdir

import sys

home = str(Path.home())

metrics_path=home+"/eeg_to_fmri/metrics/"

for metric in ["rmse", "ssim"]:
	print(metric, ":")
	for directory in sorted(listdir(metrics_path)):
		if(sys.argv[1] in directory):
			print(directory, end=": $")
			print("{:.4f}".format(round(np.mean(np.load(metrics_path+directory+"/metrics/"+metric+"_seed_"+str(sys.argv[2])+".npy")), 4)), end=" \pm ")
			print("{:.4f}".format(round(np.std(np.load(metrics_path+directory+"/metrics/"+metric+"_seed_"+str(sys.argv[2])+".npy")), 4)), end="$\n")