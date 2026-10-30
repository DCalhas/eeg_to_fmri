"""Registration QC of the XP2 BOLD derivatives: labelled fraction of the EPI mask, labels present, slab extent."""
import glob, os, numpy as np
D = os.path.expanduser("~/eeg_to_fmri/data/ds002338/derivatives_e2f")
for f in sorted(glob.glob(D+"/sub-*_e2f.npz")):
	with np.load(f) as z:
		m, c = z["mask"], z["atlas_cort"]; names = list(z["parcel_names"]); e2m = z["epi2mni"]
		zs = np.where(m.any((0, 1)))[0]
		print(os.path.basename(f)[:9], "mask %d labelled %.2f cort_labels_in_mask %d parcels %d mask_z %d-%d det(epi2mni) %.2f" % (m.sum(), np.mean(c[m] > 0), len(np.unique(c[m][c[m] > 0])), len(names), zs.min(), zs.max(), np.linalg.det(np.asarray(e2m)[:3, :3]) if np.all(np.isfinite(e2m)) else np.nan), flush=True)
