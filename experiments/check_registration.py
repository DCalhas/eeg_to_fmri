"""
Registration check for register_xp2_bold.py (2026-10-31: the registered mean EPI extends ~30 mm above the MNI brain mask
in every subject). Separates the two steps of epi2mni = t12mni @ e2t:
	T1 -> MNI   the subject's T1 resampled with t12mni, against the MNI152 template and its brain mask
	EPI -> T1   the run's mean EPI resampled into T1 space with e2t = inv(t12mni) @ epi2mni, against the T1
and reports how far the 6-dof refinement moved the slab from the world-coordinate start, and the superior extent of
brain signal in MNI space for the T1 and for the EPI. QC images and a JSON summary go to -out.

CPU-only (FSL): run as a CPU Slurm job (-p amd).
python experiments/check_registration.py -out ~/eeg_to_fmri/uncertainty/regcheck
"""
import argparse
import json
import os
import sys
import tempfile

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.expanduser("~/eeg_to_fmri/repo"))
from e2f.data.noddi_bold import _fsl, FSLDIR

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', default="ds002338")
parser.add_argument('-run', default="MIpost")
parser.add_argument('-xfm', default=os.path.expanduser("~/eeg_to_fmri/uncertainty/cache/xp2_mni/xfm"))
parser.add_argument('-out', required=True)
opt = parser.parse_args()
DATA = os.path.expanduser("~/eeg_to_fmri/data/"+opt.dataset)
MNI = os.path.join(FSLDIR, "data/standard/MNI152_T1_2mm.nii.gz")
MASK = os.path.join(FSLDIR, "data/standard/MNI152_T1_2mm_brain_mask.nii.gz")
os.makedirs(opt.out, exist_ok=True)


def fsl_frame(img):
	n = img.shape[0]; z = np.array(img.header.get_zooms()[:3], float)
	S = np.diag(np.r_[z, 1.])
	if(np.linalg.det(img.affine[:3, :3]) > 0):
		S[0, 0] = -z[0]; S[0, 3] = (n-1)*z[0]
	return S


def world_init(img_in, img_ref):
	return fsl_frame(img_ref) @ np.linalg.inv(img_ref.affine) @ img_in.affine @ np.linalg.inv(fsl_frame(img_in))


def apply(src, ref, mat, work, name, interp="trilinear"):
	m, o = os.path.join(work, name+".mat"), os.path.join(work, name+".nii.gz")
	np.savetxt(m, mat)
	_fsl("flirt", "-in", src, "-ref", ref, "-applyxfm", "-init", m, "-interp", interp, "-out", o)
	return np.asarray(nib.load(o).dataobj, np.float32)


def top_mm(vol, affine, frac=0.2):
	"""Highest world z (mm) of voxels above frac of the 99th percentile."""
	on = vol > frac*np.percentile(vol[vol > 0], 99) if np.any(vol > 0) else vol > 0
	ijk = np.argwhere(on)
	if(not len(ijk)):
		return None
	return float((affine @ np.c_[ijk, np.ones(len(ijk))].T)[2].max())


mni_img = nib.load(MNI); mni = np.asarray(mni_img.dataobj, np.float32); mask = np.asarray(nib.load(MASK).dataobj) > 0
summary = {"mni_brain_top_mm": top_mm(mask.astype(np.float32), mni_img.affine, 0.5), "mni_head_top_mm": top_mm(mni, mni_img.affine)}
subs = sorted(d for d in os.listdir(DATA) if d.startswith("sub-"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
for sub in subs:
	t12mni_p = os.path.join(opt.xfm, sub+"_t12mni.mat"); e2m_p = os.path.join(opt.xfm, "%s_%s_epi2mni.mat" % (sub, opt.run))
	nii = os.path.join(DATA, sub, "func", "%s_task-%s_bold.nii.gz" % (sub, opt.run)); t1 = os.path.join(DATA, sub, "anat", sub+"_T1w.nii.gz")
	if(not all(os.path.exists(p) for p in (t12mni_p, e2m_p, nii, t1))):
		print("I: skip", sub, flush=True); continue
	work = tempfile.mkdtemp(prefix="regcheck_"+sub+"_", dir=opt.out)
	t12mni, e2m = np.loadtxt(t12mni_p), np.loadtxt(e2m_p); e2t = np.linalg.inv(t12mni) @ e2m
	img = nib.load(nii); mean = os.path.join(work, "mean.nii.gz")
	nib.save(nib.Nifti1Image(np.asarray(img.dataobj, np.float32).mean(-1), img.affine, img.header), mean)
	t1_img = nib.load(t1); t1_vol = np.asarray(t1_img.dataobj, np.float32)
	init = world_init(img, t1_img)
	#how far the refinement moved the slab centre (mm, in the T1 FSL frame)
	c = np.r_[(np.array(img.shape[:3])-1)/2.*np.array(img.header.get_zooms()[:3]), 1.]
	shift = float(np.linalg.norm((e2t @ c)[:3]-(init @ c)[:3]))
	t1_mni = apply(t1, MNI, t12mni, work, "t1mni")
	epi_mni = apply(mean, MNI, e2m, work, "epimni")
	epi_t1 = apply(mean, t1, e2t, work, "epit1")
	epi_t1_init = apply(mean, t1, init, work, "epit1init")
	r = float(np.corrcoef(t1_mni[mask], mni[mask])[0, 1])
	summary[sub] = {"refine_shift_mm": shift, "t1_mni_corr_in_brain": r, "t1_top_mm": top_mm(t1_mni, mni_img.affine),
		"epi_top_mm": top_mm(epi_mni, mni_img.affine)}
	print("I:", sub, json.dumps(summary[sub]), flush=True)
	fig, ax = plt.subplots(2, 4, figsize=(16, 8))
	xm, ym = mni.shape[0]//2, mni.shape[1]//2+5
	for j, (vol, title) in enumerate([(t1_mni, "T1 in MNI"), (epi_mni, "EPI in MNI")]):
		ax[0, 2*j].imshow(vol[xm].T, origin="lower", cmap="gray"); ax[0, 2*j].contour(mask[xm].T, [0.5], colors="r", linewidths=0.7); ax[0, 2*j].set_title(title+", sagittal")
		ax[0, 2*j+1].imshow(vol[:, ym].T, origin="lower", cmap="gray"); ax[0, 2*j+1].contour(mask[:, ym].T, [0.5], colors="r", linewidths=0.7); ax[0, 2*j+1].set_title(title+", coronal")
	xt, yt = t1_vol.shape[0]//2, t1_vol.shape[1]//2
	for j, (vol, title) in enumerate([(epi_t1, "EPI on T1 (final)"), (epi_t1_init, "EPI on T1 (world init)")]):
		ax[1, 2*j].imshow(t1_vol[xt].T, origin="lower", cmap="gray"); ax[1, 2*j].contour(vol[xt].T, [0.3*np.percentile(vol, 99)], colors="y", linewidths=0.7); ax[1, 2*j].set_title(title+", sagittal x")
		ax[1, 2*j+1].imshow(t1_vol[:, yt].T, origin="lower", cmap="gray"); ax[1, 2*j+1].contour(vol[:, yt].T, [0.3*np.percentile(vol, 99)], colors="y", linewidths=0.7); ax[1, 2*j+1].set_title(title+", coronal")
	fig.suptitle(sub+" "+opt.run); fig.savefig(os.path.join(opt.out, sub+".png"), dpi=55); plt.close(fig)
	for f in os.listdir(work):
		os.remove(os.path.join(work, f))
	os.rmdir(work)
json.dump(summary, open(os.path.join(opt.out, "summary.json"), "w"), indent=1)
