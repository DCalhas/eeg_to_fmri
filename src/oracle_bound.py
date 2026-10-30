"""
Oracle bound of the spectral relaxation.

A decoder that zero-fills the DCT spectrum outputs volumes in the span of the low-frequency DCT basis, so for
any EEG input its error is at least ||(I-P_N) y||, the energy of the target outside the kept block. With an
orthonormal DCT, P_N y is the target low-passed to the kept block. For each cutoff this script reports,
per subject:
  * rmse_floor - RMSE between the target and its low-pass (the best any relaxed model can do)
  * temporal_kept - fraction of each brain voxel's temporal variance (the dynamic signal EEG could explain)
                    preserved by the low-pass, median over voxels
  * spatial_kept - fraction of the volume's total energy preserved

python oracle_bound.py 05 -out oracle_05.json
"""
import argparse
import json

import numpy as np
from scipy.fft import dctn, idctn

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('-factors', default="1.25,1.5,2,3,4,8", type=str, help="Downsampling factors of the kept DCT block (2 = original setting)")
parser.add_argument('-out', required=True)
opt = parser.parse_args()

from utils import data_utils, fmri_utils
import evaluate_uq

factors = list(map(float, opt.factors.split(",")))
n_individuals = getattr(data_utils, "n_individuals_"+opt.dataset)
bold_shift = getattr(fmri_utils, "bold_shift_"+opt.dataset)
n_volumes = getattr(fmri_utils, "n_volumes_"+opt.dataset)

imgs = getattr(fmri_utils, "get_individuals_paths_"+opt.dataset)(resolution_factor=1, number_individuals=n_individuals)
out = {"factors": factors, "subjects": []}
for i, img in enumerate(imgs):
	y = img.get_fdata()[..., bold_shift:bold_shift+n_volumes]
	y = (y-y.mean())/y.std()  #per-subject scalar z-score, as in data_utils.get_data(ind_volume_fit=False)
	y = np.moveaxis(y, -1, 0)
	mask = evaluate_uq.otsu_mask(y.mean(0))
	Y = dctn(y, axes=(1, 2, 3), norm="ortho")
	row = {"subject": i, "brain_fraction": float(mask.mean())}
	for f in factors:
		keep = tuple(int(round(s/f)) for s in y.shape[1:])
		Yl = np.zeros_like(Y); Yl[:, :keep[0], :keep[1], :keep[2]] = Y[:, :keep[0], :keep[1], :keep[2]]
		yl = idctn(Yl, axes=(1, 2, 3), norm="ortho")
		res = y-yl
		dy, dyl = (y-y.mean(0))[:, mask], (yl-yl.mean(0))[:, mask]
		#temporal correlation between each voxel's fluctuation and its low-passed version
		r = np.sum(dy*dyl, 0)/np.sqrt(np.sum(dy**2, 0)*np.sum(dyl**2, 0)+1e-12)
		row[str(f)] = {"keep": keep, "n_coefficients": int(np.prod(keep)),
					"rmse_floor": float(np.sqrt(np.mean(res**2))), "rmse_floor_brain": float(np.sqrt(np.mean(res[:, mask]**2))),
					"spatial_kept": float(1-np.sum(res**2)/np.sum(y**2)),
					"temporal_kept_median": float(np.median(np.var(dyl, 0)/np.var(dy, 0))),
					"temporal_r_median": float(np.median(r))}
	out["subjects"].append(row)
	print(i, {k: (v["rmse_floor"], v["temporal_r_median"]) for k, v in row.items() if isinstance(v, dict)}, flush=True)

with open(opt.out, "w") as f:
	json.dump(out, f, indent=1)
