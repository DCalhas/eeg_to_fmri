"""
Where does the deep pipeline lose an EEG signal that a closed-form ridge recovers? (review #2, positive control)

On the injected fluctuation target (-target fluct -inject_snr s, preprocess_data.dataset_subjects) and the same folds,
from the same input features as linear_probe.py (band power of the deep models' STFT input window, 63 x 5 x 10), fit
to every voxel of the training mask:
	linear_laplace  linear map, the deep models' training loop: Adam at the backbone's learning rate, batch 4, 10 epochs,
	                Laplace likelihood with a learned per-voxel scale
	linear_mse      linear map, a conventional regression loop: Adam 1e-3, batch 64, 50 epochs, MSE
	mlp_laplace     one hidden layer (256, ReLU) with the deep models' loop
	mlp_mse         one hidden layer with the regression loop
Rungs 0 and 4 exist already: linear_probe.py (closed-form ridge, recovers the signal) and the full backbone
(train_cv.py inject* runs, does not). Reported per test subject: median voxel temporal r inside the injected box and
outside it, for the real test EEG and the shift/swap nulls (no refit).

python ablation_ladder.py 05 -target fluct -inject_snr 1 -variant linear_laplace -fold 0 -name ladder -out_dir RUNS
"""
import argparse
import json
import os

import numpy as np

import experiment

parser = experiment.add_model_args(argparse.ArgumentParser())
parser.add_argument('-variant', required=True, choices=["linear_laplace", "linear_mse", "mlp_laplace", "mlp_mse"])
opt = parser.parse_args()

from utils import process_utils
process_utils.process_setup_tensorflow(opt.gpu_mem, seed=opt.seed)
import tensorflow as tf
from evaluate_uq import eeg_null
from linear_probe_utils import features as _features, voxel_r
from utils import fmri_utils as _fu

LR_BACKBONE = 0.002980911194116198
features = lambda e: _features(e, getattr(_fu, "TR_"+opt.dataset))
splits, test, calib = experiment.load_split(opt)
mask = splits["mask"].reshape(-1)
box = splits["inject_mask"].reshape(-1)[mask] if splits.get("inject_mask") is not None else None
get = lambda name: (features(splits[name][0]), splits[name][1][..., 0].reshape(len(splits[name][1]), -1)[:, mask].astype(np.float32), splits[name][2])
Xtr, Ytr, _ = get("train"); Xte, Yte, ste = get("test")
mx, sx = Xtr.mean(0), Xtr.std(0)+1e-6
Xtr, Xte = (Xtr-mx)/sx, (Xte-mx)/sx
V = Ytr.shape[1]
laplace = opt.variant.endswith("laplace")
lr, batch, epochs = (LR_BACKBONE, 4, 10) if laplace else (1e-3, 64, 50)

inp = tf.keras.Input((Xtr.shape[1],))
h = tf.keras.layers.Dense(256, activation="relu")(inp) if opt.variant.startswith("mlp") else inp
mu = tf.keras.layers.Dense(V)(h)
model = tf.keras.Model(inp, mu)
log_b = tf.Variable(tf.zeros((V,)), name="log_b")
opt_ = tf.keras.optimizers.Adam(lr)
variables = model.trainable_variables+([log_b] if laplace else [])


@tf.function
def step(x, y):
	with tf.GradientTape() as tape:
		m = model(x, training=True)
		if(laplace):
			b = tf.exp(log_b)+1e-3
			loss = tf.reduce_mean(tf.abs(y-m)/b+tf.math.log(2*b))
		else:
			loss = tf.reduce_mean((y-m)**2)
	opt_.apply_gradients(zip(tape.gradient(loss, variables), variables))
	return loss


data = tf.data.Dataset.from_tensor_slices((Xtr, Ytr)).shuffle(len(Xtr), seed=opt.seed, reshuffle_each_iteration=True).batch(batch)
curve = []
for e in range(epochs):
	curve.append(float(np.mean([step(x, y).numpy() for x, y in data])))
predict = lambda X: model.predict(X, batch_size=256, verbose=0)


def per_subject(P, sel=None):
	out = {}
	for k in np.unique(ste):
		r = voxel_r(P[ste == k], Yte[ste == k])
		out[int(k)] = float(np.nanmedian(r if sel is None else r[sel]))
	return out


res = {"variant": opt.variant, "lr": lr, "batch": batch, "epochs": epochs, "train_curve": curve}
P = predict(Xte)
res["real"] = {"all": per_subject(P)}
if(box is not None):
	res["real"]["box"] = per_subject(P, box); res["real"]["outside"] = per_subject(P, ~box)
for null in ["shift", "swap"]:
	e, _ = eeg_null(splits["test"][0], ste, null, seed=0)
	Pn = predict((features(e)-mx)/sx)
	res[null] = {"all": per_subject(Pn)}
	if(box is not None):
		res[null]["box"] = per_subject(Pn, box)
out = experiment.run_dir(opt); os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "ladder.json"), "w") as f:
	json.dump(res, f, indent=1)
print("LADDER", opt.name, opt.variant, "fold", opt.fold, json.dumps({k: {kk: float(np.mean(list(vv.values()))) for kk, vv in v.items()} for k, v in res.items() if k in ("real", "shift", "swap")}), flush=True)
