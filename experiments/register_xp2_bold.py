"""
Registered BOLD of ds002338 MIpre/MIpost on a common 3 mm MNI grid (review #1: the deep arm used native-space volumes,
so voxel v was not the same tissue across subjects or runs). Reuses the per-volume motion matrices mc_mats (volume k ->
run mean) of the shared preprocessing (data/ds002338/derivatives_e2f, preprocess_bold), but NOT its epi2mni: that
EPI->T1 flirt started from the FSL voxel frame, and on XP2's oblique 16-slice slab it can converge to a wrong rotation
(sub-xp201: slab rotated ~45 deg, 2026-10-28 QC). Here EPI->T1 starts from the scanner world coordinates (same
session, valid sforms) and is only refined (6 dof, corratio, -nosearch); T1->MNI152 is a 12-dof flirt. Matrices are
saved in xfm/ and QC images (registered mean EPI and T1 over the MNI brain outline) in qc/.

Outputs in uncertainty/cache/xp2_mni (FSL flirt -applyisoxfm 3 on the MNI152 2 mm reference, i.e. a 61x73x61 grid):
	<sub>_<run>_raw.npy    (T, 61, 73, 61) float16: raw volumes, motion-corrected and registered in one interpolation
	                       (epi2mni @ mc_k), for the deep models (the loader derives the run-z-score / fluct targets)
	<sub>_<run>_clean.npy  (T, 61, 73, 61) float16: the cleaned BOLD of the shared preprocessing (Friston-24, FD spikes,
	                       0.01 Hz high-pass, 6 mm smoothing, z-scored) registered with epi2mni, for the ROI analyses
	atlas_cort_3mm.npy, mni_brain_3mm.npy, labels_cort.json; .DONE when every subject/run exists

CPU-only software (FSL): run as a CPU Slurm job (-p amd).
python experiments/register_xp2_bold.py -n_jobs 32
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.expanduser("~/eeg_to_fmri/repo"))
from e2f.data.noddi_bold import _fsl, _labels, FSLDIR

DATA = os.path.expanduser("~/eeg_to_fmri/data/ds002338")
OUT = os.path.expanduser("~/eeg_to_fmri/uncertainty/cache/xp2_mni")
DERIV = os.path.join(DATA, "derivatives_e2f")
MNI = os.path.join(FSLDIR, "data/standard/MNI152_T1_2mm.nii.gz")

parser = argparse.ArgumentParser()
parser.add_argument('-n_jobs', default=32, type=int)
parser.add_argument('-runs', default="MIpre,MIpost")
parser.add_argument('-dataset', default="ds002338", help="ds002336 (CHUR-XP1) with -deriv/-out for the replication")
parser.add_argument('-deriv', default=None, help="directory of the preprocess_bold npz files (default data/<dataset>/derivatives_e2f)")
parser.add_argument('-out', default=None, help="output directory (default cache/xp2_mni)")
parser.add_argument('-subjects', default=None)
opt = parser.parse_args()
DATA = os.path.expanduser("~/eeg_to_fmri/data/"+opt.dataset)
DERIV = opt.deriv or os.path.join(DATA, "derivatives_e2f")
OUT = opt.out or OUT
os.makedirs(os.path.join(OUT, "xfm"), exist_ok=True); os.makedirs(os.path.join(OUT, "qc"), exist_ok=True)


def fsl_frame(img):
	"""FSL scaled-voxel frame of an image: voxel -> scaled mm, with the x flip FSL applies to neurological storage."""
	n = img.shape[0]; z = np.array(img.header.get_zooms()[:3], float)
	S = np.diag(np.r_[z, 1.])
	if(np.linalg.det(img.affine[:3, :3]) > 0):
		S[0, 0] = -z[0]; S[0, 3] = (n-1)*z[0]
	return S


def world_init(img_in, img_ref):
	"""FSL matrix in -> ref implied by the two sforms (identity in scanner world coordinates)."""
	return fsl_frame(img_ref) @ np.linalg.inv(img_ref.affine) @ img_in.affine @ np.linalg.inv(fsl_frame(img_in))


def subject_t1_to_mni(sub, work):
	m = os.path.join(OUT, "xfm", sub+"_t12mni.mat")
	if(not os.path.exists(m)):
		t1 = os.path.join(DATA, sub, "anat", sub+"_T1w.nii.gz")
		_fsl("flirt", "-in", t1, "-ref", MNI, "-omat", m, "-dof", "12")
	return np.loadtxt(m)


def epi_to_mni(sub, run, img, raw, work):
	m = os.path.join(OUT, "xfm", "%s_%s_epi2mni.mat" % (sub, run))
	if(not os.path.exists(m)):
		t1 = os.path.join(DATA, sub, "anat", sub+"_T1w.nii.gz")
		mean = os.path.join(work, "mean.nii.gz"); nib.save(nib.Nifti1Image(raw.mean(-1), img.affine, img.header), mean)
		init, e2t = os.path.join(work, "init.mat"), os.path.join(work, "e2t.mat")
		np.savetxt(init, world_init(img, nib.load(t1)))
		_fsl("flirt", "-in", mean, "-ref", t1, "-init", init, "-omat", e2t, "-dof", "6", "-cost", "corratio", "-nosearch")
		np.savetxt(m, subject_t1_to_mni(sub, work) @ np.loadtxt(e2t))
	return np.loadtxt(m)


def qc_image(sub, run, R):
	import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
	brain = np.load(os.path.join(OUT, "mni_brain_3mm.npy")) > 0
	m = np.mean(R[::20], 0); zs = np.where(m.any((0, 1)))[0]
	fig, ax = plt.subplots(2, 5, figsize=(15, 6))
	for i, z in enumerate(np.linspace(zs[0], zs[-1], 5).astype(int)):
		ax[0, i].imshow(m[:, :, z].T, origin="lower", cmap="gray"); ax[0, i].contour(brain[:, :, z].T, [0.5], colors="r", linewidths=0.6); ax[0, i].set_title("z=%d" % z)
	for i, y in enumerate(np.linspace(15, 60, 5).astype(int)):
		ax[1, i].imshow(m[:, y, :].T, origin="lower", cmap="gray"); ax[1, i].contour(brain[:, y, :].T, [0.5], colors="r", linewidths=0.6); ax[1, i].set_title("y=%d" % y)
	fig.suptitle("%s %s" % (sub, run)); fig.savefig(os.path.join(OUT, "qc", "%s_%s.png" % (sub, run)), dpi=60); plt.close(fig)


def to_mni(vol, img, mat, work, k, interp="trilinear"):
	vin, vout, m = (os.path.join(work, "%s%05d%s" % (p, k, s)) for p, s in (("in", ".nii.gz"), ("out", ".nii.gz"), ("m", ".mat")))
	nib.save(nib.Nifti1Image(vol.astype(np.float32), img.affine, img.header), vin); np.savetxt(m, mat)
	_fsl("flirt", "-in", vin, "-ref", MNI, "-applyisoxfm", "3", "-init", m, "-interp", interp, "-out", vout)
	out = np.asarray(nib.load(vout).dataobj, np.float32)
	for f in (vin, vout, m):
		os.remove(f)
	return out


def static_files(work):
	"""HO cortical atlas and MNI brain mask on the 3 mm grid (identity transform, nearest neighbour)."""
	eye = os.path.join(work, "eye.mat"); np.savetxt(eye, np.eye(4))
	for name, src, interp in (("atlas_cort_3mm", "data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz", "nearestneighbour"),
							  ("mni_brain_3mm", "data/standard/MNI152_T1_2mm_brain_mask.nii.gz", "nearestneighbour")):
		o = os.path.join(work, name+".nii.gz")
		_fsl("flirt", "-in", os.path.join(FSLDIR, src), "-ref", MNI, "-applyisoxfm", "3", "-init", eye, "-interp", interp, "-out", o)
		np.save(os.path.join(OUT, name+".npy"), np.asarray(nib.load(o).dataobj).astype(np.int16))
	with open(os.path.join(OUT, "labels_cort.json"), "w") as f:
		json.dump(_labels(os.path.join(FSLDIR, "data/atlases/HarvardOxford-Cortical.xml")), f)


def one_run(sub, run):
	raw_out, clean_out = (os.path.join(OUT, "%s_%s_%s.npy" % (sub, run, k)) for k in ("raw", "clean"))
	if(os.path.exists(raw_out) and os.path.exists(clean_out)):
		return sub, run, "cached"
	nii = os.path.join(DATA, sub, "func", "%s_task-%s_bold.nii.gz" % (sub, run))
	der = os.path.join(DERIV, "%s_task-%s_bold_e2f.npz" % (sub, run))
	if(not (os.path.exists(nii) and os.path.exists(der))):
		return sub, run, "missing"
	img = nib.load(nii); raw = np.asarray(img.dataobj, np.float32)
	with np.load(der, allow_pickle=True) as f:
		mc, clean = f["mc_mats"], f["bold"]
	assert clean.shape[0] == raw.shape[-1] == mc.shape[0], (clean.shape, raw.shape, mc.shape)
	work = tempfile.mkdtemp(prefix="reg_%s_%s_" % (sub, run), dir=OUT)
	try:
		e2m = epi_to_mni(sub, run, img, raw, work)
		R = Parallel(n_jobs=opt.n_jobs)(delayed(to_mni)(raw[..., k], img, e2m @ mc[k], work, k) for k in range(raw.shape[-1]))
		R = np.stack(R).astype(np.float16)
		np.save(raw_out+".tmp.npy", R); os.replace(raw_out+".tmp.npy", raw_out)
		qc_image(sub, run, R.astype(np.float32))
		C = Parallel(n_jobs=opt.n_jobs)(delayed(to_mni)(clean[k].astype(np.float32), img, e2m, work, 100000+k) for k in range(clean.shape[0]))
		np.save(clean_out+".tmp.npy", np.stack(C).astype(np.float16)); os.replace(clean_out+".tmp.npy", clean_out)
	finally:
		shutil.rmtree(work, ignore_errors=True)
	return sub, run, "done"


work = tempfile.mkdtemp(prefix="static_", dir=OUT)
static_files(work); shutil.rmtree(work)
subs = sorted(os.path.basename(d) for d in glob.glob(os.path.join(DATA, "sub-xp*")))
if(opt.subjects):
	subs = [s for s in subs if s in opt.subjects.split(",")]
status = [one_run(s, r) for s in subs for r in opt.runs.split(",")]
for st in status:
	print("I:", *st, flush=True)
if(not opt.subjects and all(st[2] != "missing" for st in status)):
	open(os.path.join(OUT, ".DONE"), "w").write("%d runs\n" % len(status))
