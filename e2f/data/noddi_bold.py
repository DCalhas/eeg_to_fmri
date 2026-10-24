"""RQ1b fair target: preprocessed NODDI BOLD (CPU job: FSL flirt + nilearn).

Per subject (300 volumes, TR 2.16 s, native 64x64x30):
  1. rigid motion correction: every volume registered to the temporal mean with FSL flirt (6 dof, corratio);
     mcflirt is not part of the Frida FSL install. Motion parameters (3 rotations, 3 translations) and
     framewise displacement (Power 2012, 50 mm sphere) are derived from the matrices.
  2. EPI brain mask (nilearn compute_epi_mask on the corrected mean); spatial smoothing 6 mm FWHM.
  3. nuisance regression + filtering: Friston-24 motion model + FD > 0.5 mm spike regressors, linear detrend, high-pass 0.01 Hz,
     voxelwise z-score (nilearn.signal.clean), inside the mask.
  4. parcels: corrected mean EPI -> MNI152 2 mm (flirt 12 dof, corratio); inverse affine warps the
     Harvard-Oxford cortical (thr25) + subcortical (thr25, grey-matter labels only) atlases to native space
     (nearest neighbour). Parcel time series = mean of the cleaned voxels in each parcel; plus the global signal.
Output runs/noddi_bold_v1/<sub>.npz: bold (300, 64, 64, 30) float16 (0 outside mask), mask, parcels (300, R),
parcel_names, motion (300, 6), fd (300,), plus the atlas in native space.
"""
import os, glob, subprocess, tempfile, shutil
import numpy as np
import nibabel as nib

from e2f.data import noddi

FSLDIR = os.path.expanduser("~/fsl")
OUT = os.path.expanduser("~/eeg_to_fmri/runs/noddi_bold_v3")  # v3: Friston-24 + FD>0.5 mm spikes; v2: centre-based motion
TR = noddi.TR
HO_SUB_EXCLUDE = {1, 2, 3, 12, 13, 14}  # L/R cerebral white matter, cerebral cortex, lateral ventricle


def _fsl(*args):
	env = dict(os.environ, FSLDIR=FSLDIR, FSLOUTPUTTYPE="NIFTI_GZ", PATH=f"{FSLDIR}/bin:" + os.environ["PATH"])
	r = subprocess.run([os.path.join(FSLDIR, "bin", args[0])] + list(args[1:]), env=env, capture_output=True, text=True)
	if r.returncode != 0:
		raise RuntimeError(f"{args[0]} failed: {r.stderr[-2000:]}")
	return r.stdout


def _rigid_params(M, centre_mm):
	"""4x4 flirt matrix (scaled-voxel mm coordinates with the origin at the image corner) -> rotations (rad, x/y/z)
	and translations (mm) of the volume centre. flirt rotates about the corner, so the raw M[:3, 3] mixes rotation
	into translation (mm-scale artefacts for 0.01-rad rotations); the centre displacement R c + t - c does not."""
	R = M[:3, :3]
	rx = np.arctan2(R[2, 1], R[2, 2]); ry = np.arcsin(-np.clip(R[2, 0], -1, 1)); rz = np.arctan2(R[1, 0], R[0, 0])
	return np.r_[rx, ry, rz, R @ centre_mm + M[:3, 3] - centre_mm]


def framewise_displacement(motion, radius=50.0):
	d = np.abs(np.diff(motion, axis=0))
	return np.r_[0.0, (d[:, :3] * radius).sum(1) + d[:, 3:].sum(1)]


def motion_correct(img, work, n_jobs=8):
	from joblib import Parallel, delayed
	data = np.asarray(img.dataobj, dtype=np.float32)
	ref = os.path.join(work, "ref.nii.gz")
	nib.save(nib.Nifti1Image(data.mean(-1), img.affine, img.header), ref)
	def one(k):
		vin, vout, mat = (os.path.join(work, f"v{k:03d}{s}") for s in (".nii.gz", "_mc.nii.gz", ".mat"))
		nib.save(nib.Nifti1Image(data[..., k], img.affine, img.header), vin)
		_fsl("flirt", "-in", vin, "-ref", ref, "-out", vout, "-omat", mat, "-dof", "6", "-cost", "corratio", "-interp", "trilinear")
		return np.asarray(nib.load(vout).dataobj, dtype=np.float32), np.loadtxt(mat)
	res = Parallel(n_jobs=n_jobs)(delayed(one)(k) for k in range(data.shape[-1]))
	vols = np.stack([r[0] for r in res], -1)
	centre = (np.array(img.shape[:3]) - 1) / 2 * np.array(img.header.get_zooms()[:3])
	mats = np.stack([r[1] for r in res])
	motion = np.stack([_rigid_params(m, centre) for m in mats])
	return nib.Nifti1Image(vols, img.affine, img.header), motion, mats


def native_atlas(mean_img, work):
	mean_f = os.path.join(work, "mean_mc.nii.gz"); nib.save(mean_img, mean_f)
	mni = os.path.join(FSLDIR, "data/standard/MNI152_T1_2mm_brain.nii.gz")
	e2m, m2e = os.path.join(work, "epi2mni.mat"), os.path.join(work, "mni2epi.mat")
	_fsl("flirt", "-in", mean_f, "-ref", mni, "-omat", e2m, "-dof", "12", "-cost", "corratio")
	np.savetxt(m2e, np.linalg.inv(np.loadtxt(e2m)))  # convert_xfm -inverse (not in the Frida FSL install)
	ho = os.path.join(FSLDIR, "data/atlases/HarvardOxford")
	out = {}
	for name, f in (("cort", "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz"), ("sub", "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz")):
		o = os.path.join(work, f"{name}_native.nii.gz")
		_fsl("flirt", "-in", os.path.join(ho, f), "-ref", mean_f, "-applyxfm", "-init", m2e, "-interp", "nearestneighbour", "-out", o)
		out[name] = np.asarray(nib.load(o).dataobj).astype(int)
	return out, np.loadtxt(e2m)


def _labels(xml):
	import xml.etree.ElementTree as ET
	return {int(l.get("index")) + 1: l.text for l in ET.parse(xml).getroot().iter("label")}


def build(sub, force=False, n_jobs=8):
	f = os.path.join(OUT, f"{sub}.npz")
	if os.path.exists(f) and not force:
		return f
	from nilearn import image, masking, signal  # only needed to build (envs without nilearn can read the cache)
	os.makedirs(OUT, exist_ok=True)
	work = tempfile.mkdtemp(prefix=f"bold_{sub}_", dir=OUT)
	try:
		img = nib.load(glob.glob(os.path.join(noddi.ROOT, "fMRI", sub, "*nw_mepi_rest_with_cross.nii.gz"))[0])
		assert img.shape[-1] == 300, img.shape
		mc, motion, mats = motion_correct(img, work, n_jobs)
		mean_img = image.mean_img(mc)
		mask = masking.compute_epi_mask(mean_img)
		sm = image.smooth_img(mc, 6.0)
		X = masking.apply_mask(sm, mask)  # (300, V)
		# Friston-24: params, derivatives and the squares of both (Friston et al. 1996), plus FD spike regressors
		# for volumes with FD > 0.5 mm (Power et al. 2014)
		dm = np.r_[np.zeros((1, 6)), np.diff(motion, axis=0)]
		fd = framewise_displacement(motion)
		spikes = np.eye(len(fd))[:, fd > 0.5]
		conf = np.c_[motion, dm, motion ** 2, dm ** 2, spikes]
		Xc = signal.clean(X, detrend=True, standardize="zscore_sample", confounds=conf, high_pass=0.01, t_r=TR)
		bold = np.zeros(img.shape[:3] + (300,), np.float32)
		bold[np.asarray(mask.dataobj, bool)] = Xc.T
		bold = np.moveaxis(bold, -1, 0)
		atlas, e2m = native_atlas(mean_img, work)
		ho = os.path.join(FSLDIR, "data/atlases")
		names_c, names_s = _labels(os.path.join(ho, "HarvardOxford-Cortical.xml")), _labels(os.path.join(ho, "HarvardOxford-Subcortical.xml"))
		m = np.asarray(mask.dataobj, bool)
		parcels, pnames = [], []
		for lab, nm in names_c.items():
			sel = (atlas["cort"] == lab) & m
			if sel.sum() >= 5:
				parcels.append(bold[:, sel].mean(1)); pnames.append(f"ctx:{nm}")
		for lab, nm in names_s.items():
			if lab in HO_SUB_EXCLUDE:
				continue
			sel = (atlas["sub"] == lab) & m
			if sel.sum() >= 5:
				parcels.append(bold[:, sel].mean(1)); pnames.append(f"sub:{nm}")
		parcels.append(bold[:, m].mean(1)); pnames.append("global")
		tmp = f + ".tmp.npz"
		np.savez(tmp, bold=bold.astype(np.float16), mask=m, parcels=np.stack(parcels, 1).astype(np.float32), parcel_names=np.array(pnames),
				 motion=motion.astype(np.float32), fd=fd.astype(np.float32), mc_mats=mats.astype(np.float32), n_spikes=int(spikes.shape[1]),
				 atlas_cort=atlas["cort"].astype(np.int16), atlas_sub=atlas["sub"].astype(np.int16), epi2mni=e2m, affine=img.affine)
		os.replace(tmp, f)
	finally:
		shutil.rmtree(work, ignore_errors=True)
	return f


if __name__ == "__main__":
	import sys
	n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
	for s in noddi.subjects():
		print(s, build(s, n_jobs=n_jobs), flush=True)
