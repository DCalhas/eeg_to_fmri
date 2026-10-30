"""
BOLD preprocessing of ds002338 (CHUR-XP2) MIpost runs for the ROI ridge arm, with the shared pipeline
e2f.data.noddi_bold.preprocess_bold (paper S, read-only import): flirt motion correction, Friston-24 + FD spikes,
detrend, 0.01 Hz high-pass, z-score, 6 mm smoothing, EPI->T1->MNI registration, Harvard-Oxford parcels.

CPU-only software (FSL, nilearn): runs as a CPU Slurm job (-p amd), with ~/eeg_to_fmri/envs/torch/bin/python.
Output: data/ds002338/derivatives_e2f/<sub>_task-MIpost_bold_e2f.npz, a QC table, and .DONE when all subjects exist.

python experiments/preprocess_xp2_bold.py -n_parallel 4 -n_jobs 8
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.expanduser("~/eeg_to_fmri/repo"))
from e2f.data.noddi_bold import preprocess_bold

DATA = os.path.expanduser("~/eeg_to_fmri/data/ds002338")
OUT = os.path.join(DATA, "derivatives_e2f")
MOTOR = ["ctx:Precentral Gyrus", "ctx:Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)", "ctx:Postcentral Gyrus"]

parser = argparse.ArgumentParser()
parser.add_argument('-n_parallel', default=4, type=int, help="subjects in parallel")
parser.add_argument('-n_jobs', default=8, type=int, help="flirt processes per subject")
parser.add_argument('-tasks', default="MIpost", type=str, help="comma separated: MIpost, MIpre, 1dNF_run-01, 1dNF_run-02, 1dNF_run-03")
opt = parser.parse_args()

subjects = sorted(os.path.basename(d) for d in glob.glob(os.path.join(DATA, "sub-xp2*")))
def run(sub, task):
	bold = os.path.join(DATA, sub, "func", sub+"_task-"+task+"_bold.nii.gz")
	if(not os.path.exists(bold)):
		return sub, None
	t1 = os.path.join(DATA, sub, "anat", sub+"_T1w.nii.gz")
	out = os.path.join(OUT, sub+"_task-"+task+"_bold_e2f.npz")
	preprocess_bold(bold, 1.0, out, n_jobs=opt.n_jobs, t1_path=t1)
	return sub, out

os.makedirs(OUT, exist_ok=True)
tasks = opt.tasks.split(",")
runs = Parallel(n_jobs=opt.n_parallel)(delayed(run)(s, t) for s in subjects for t in tasks)
done = [(s, o) for s, o in runs if o is not None and o.endswith("_task-MIpost_bold_e2f.npz")]
print("I: preprocessed", sum(o is not None for _, o in runs), "runs; missing", [(s, t) for (s, o), t in zip(runs, [t for _ in subjects for t in tasks]) if o is None], flush=True)
if(not done):
	print("PREPROCESS_XP2_DONE (no MIpost in this call)"); sys.exit(0)

#registration QC: in-mask voxels of the motor parcels per subject (partial 16-slice coverage)
qc = {}
for sub, out in done:
	with np.load(out) as f:
		names = list(f["parcel_names"]); cort = f["atlas_cort"]; mask = f["mask"]
		qc[sub] = {"n_parcels": len(names), "n_volumes": int(f["parcels"].shape[0]), "n_spikes": int(f["n_spikes"]),
				"mean_fd": float(np.mean(f["fd"])), "motor_present": [m for m in MOTOR if m in names],
				"mask_voxels": int(mask.sum())}
		print(sub, json.dumps(qc[sub]), flush=True)
with open(os.path.join(OUT, "qc.json"), "w") as f:
	json.dump(qc, f, indent=1)
if(len(done) == len(subjects)):
	with open(os.path.join(OUT, ".DONE"), "w") as f:
		f.write("uncertainty session, preprocess_bold (rebuild-2026) with t1_path\n")
print("PREPROCESS_XP2_DONE", len(done), "subjects")
