"""
CHUR-XP1 (ds002336) replication: BOLD preprocessing of every run with the shared pipeline (e2f.data.noddi_bold.preprocess_bold,
read-only import; TR 2 s), written to uncertainty/cache/xp1_e2f (not to the shared data directory). Registration to MNI
then uses register_xp2_bold.py -dataset ds002336 -deriv cache/xp1_e2f -out cache/xp1_mni -runs <tasks>.
CPU-only software (FSL, nilearn): CPU Slurm job (-p amd), ~/eeg_to_fmri/envs/torch/bin/python.

python experiments/preprocess_xp1_bold.py -n_parallel 4 -n_jobs 8
"""
import argparse
import glob
import os
import sys

from joblib import Parallel, delayed

sys.path.insert(0, os.path.expanduser("~/eeg_to_fmri/repo"))
from e2f.data.noddi_bold import preprocess_bold

DATA = os.path.expanduser("~/eeg_to_fmri/data/ds002336")
OUT = os.path.expanduser("~/eeg_to_fmri/uncertainty/cache/xp1_e2f")
parser = argparse.ArgumentParser()
parser.add_argument('-n_parallel', default=4, type=int)
parser.add_argument('-n_jobs', default=8, type=int)
parser.add_argument('-tasks', default="motorloc,eegNF,fmriNF,eegfmriNF")
opt = parser.parse_args()
os.makedirs(OUT, exist_ok=True)
subjects = sorted(os.path.basename(d) for d in glob.glob(os.path.join(DATA, "sub-xp1*")))


def run(sub, task):
	bold = os.path.join(DATA, sub, "func", "%s_task-%s_bold.nii.gz" % (sub, task))
	if(not os.path.exists(bold)):
		return sub, task, "missing"
	out = os.path.join(OUT, "%s_task-%s_bold_e2f.npz" % (sub, task))
	preprocess_bold(bold, 2.0, out, n_jobs=opt.n_jobs, t1_path=os.path.join(DATA, sub, "anat", sub+"_T1w.nii.gz"))
	return sub, task, "done"


for st in Parallel(n_jobs=opt.n_parallel)(delayed(run)(s, t) for s in subjects for t in opt.tasks.split(",")):
	print("I:", *st, flush=True)
