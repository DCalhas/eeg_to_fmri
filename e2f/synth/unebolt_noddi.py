"""RQ1b with the state-of-the-art ROI-level synthesizer UnEBOLT (MIDL 2026; NeuroBOLT successor, Li et al.).

Uses the authors' model, LaBraM initialisation (labram-base.pth), loss (MSE + temporal + ROI correlation),
optimiser (AdamW, layer decay 0.65), schedule (cosine, 5 warm-up epochs) and EEG scaling (/100 on uV), imported
read-only from ~/eeg_to_fmri/baselines/UnEBOLT. Changes, all to the evaluation protocol:
  data      trigger-aligned NODDI EEG (57 common channels, uppercase 10-20 names), resampled 250 -> 200 Hz,
            16 s window preceding each volume onset (as the authors' epoching_seq2one); target = the RQ1b v3
            preprocessed Harvard-Oxford parcels (Friston-24 + spikes, high-pass, z-score), volumes k >= 20
  folds     data/noddi/folds.json LOSO; epoch selection on the val subjects (the authors' loop tracks test corr)
  metrics   parcel r and out-of-sample R^2 on the held-out subject; circular-shift null on the EEG windows
GPU job (envs/unebolt).
"""
import os, sys, json, argparse, time
import numpy as np
import torch
from scipy.signal import resample_poly

from e2f.data import noddi, preprocess as P
from e2f.synth.benchmark_noddi import make_folds
from e2f.synth import benchmark_bold as BB

UNE = os.path.expanduser("~/eeg_to_fmri/baselines/UnEBOLT")
RUNS = os.path.expanduser("~/eeg_to_fmri/runs/unebolt_noddi")
CH = [c.upper() for c in P.NODDI_CHANNELS]


def subject_data(sub, n_frames=20, win_s=16):
	d = noddi.load(sub)
	parcels = BB.features(sub, "motion")[1]  # common parcels, volumes k >= 20 (same set as the ridge benchmark)
	eeg = resample_poly(d["eeg"].astype(np.float64), 4, 5, axis=1) * 1e6  # 250 -> 200 Hz, V -> uV
	fs, w = 200, win_s * 200
	X = []
	for k in range(n_frames, 300):
		e = int(round(d["onsets"][k] * fs))
		assert e - w >= 0
		X.append(eeg[:, e - w:e])
	return np.stack(X).astype(np.float32), parcels.astype(np.float32)


def build(args, n_roi):
	sys.path.insert(0, UNE)
	import main as M
	import utils as U
	from optim_factory import create_optimizer, LayerDecayValueAssigner
	from customized_losses import CustomMSECorrLoss
	argv = sys.argv
	sys.argv = ["main.py", "--nb_eegchan", str(len(CH)), "--nb_roi", str(n_roi), "--epochs", str(args.epochs),
				"--batch_size", str(args.bs), "--pretrained_weights", os.path.join(UNE, "checkpoints", "labram-base.pth"),
				"--warmup_epochs", str(5 if args.epochs > 5 else 1)]  # authors' 5 warm-up epochs; 1 for short smoke runs
	try:
		a = M.get_args()  # the authors' parser reads sys.argv
	finally:
		sys.argv = argv
	model = M.get_models(a)
	a.window_size = (1, a.input_size // model.patch_size); a.patch_size = model.patch_size
	# labram-base.pth holds numpy scalars; keep torch.load(weights_only=True) and allow-list only those types
	import numpy.core.multiarray as nca
	import argparse
	safe = [nca.scalar, np.dtype, argparse.Namespace] + [type(np.dtype(t)) for t in ("float64", "float32", "int64", "int32", "bool")]
	with torch.serialization.safe_globals(safe):
		M.initialize_encoder(model, a)
	return M, U, a, model, create_optimizer, LayerDecayValueAssigner, CustomMSECorrLoss


@torch.no_grad()
def predict(model, X, input_chans, bs=128):
	from einops import rearrange
	model.eval()
	out = []
	for i in range(0, len(X), bs):
		e = torch.from_numpy(X[i:i + bs]).cuda().float() / 100
		e = rearrange(e, "B N (A T) -> B N A T", T=200)
		out.append(model(e, input_chans=input_chans).float().cpu().numpy())
	return np.concatenate(out)


def run_fold(fold, args):
	import engine
	tr = [subject_data(s) for s in fold["train"]]; va = [subject_data(s) for s in fold["val"]]; te = subject_data(fold["test"][0])
	Xtr, Ytr = np.concatenate([t[0] for t in tr]), np.concatenate([t[1] for t in tr])
	Xva, Yva = np.concatenate([t[0] for t in va]), np.concatenate([t[1] for t in va])
	M, U, a, model, create_optimizer, LDA, Loss = build(args, Ytr.shape[1])
	torch.manual_seed(args.seed); np.random.seed(args.seed)
	model.cuda()
	ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
	dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=True, drop_last=True, num_workers=2)
	steps = len(ds) // a.batch_size
	nl = model.get_num_layers()
	assigner = LDA(list(a.layer_decay ** (nl + 1 - i) for i in range(nl + 2)))
	opt = create_optimizer(a, model, skip_list=model.no_weight_decay(), get_num_layer=assigner.get_layer_id, get_layer_scale=assigner.get_scale)
	lr_s = U.cosine_scheduler(a.lr, a.min_lr, a.epochs, steps, warmup_epochs=a.warmup_epochs, warmup_steps=a.warmup_steps)
	wd_s = U.cosine_scheduler(a.weight_decay, a.weight_decay, a.epochs, steps)
	crit = Loss(loss_weights={"mse": a.loss_mse_weight, "temporal": a.loss_temporal_weight, "roi": a.loss_roi_weight})
	scaler = U.NativeScalerWithGradNormCount() if hasattr(U, "NativeScalerWithGradNormCount") else U.NativeScaler()
	input_chans = U.get_input_chans(CH)
	best = (-np.inf, None, -1)
	for ep in range(a.epochs):
		engine.train_one_epoch(model, crit, dl, opt, torch.device("cuda"), ep, scaler, a.clip_grad, None, None,
							   start_steps=ep * steps, lr_schedule_values=lr_s, wd_schedule_values=wd_s,
							   num_training_steps_per_epoch=steps, update_freq=1, ch_names=CH, is_binary=False)
		pv = predict(model, Xva, input_chans)
		s = BB.r2_cols(pv, Yva)[1] if args.select == "R2" else float(np.nanmean(BB.corr_cols(pv, Yva)))
		if s > best[0]:
			best = (s, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
		print("epoch", ep, "val", args.select, round(float(s), 4), flush=True)
	model.load_state_dict(best[1])
	Xte, Yte = te
	Yte_d = BB.detrend_cols(Yte)
	def score(X):
		Pp = BB.detrend_cols(predict(model, X, input_chans))  # detrended scoring (A20)
		r = BB.corr_cols(Pp, Yte_d); r2, r2g = BB.r2_cols(Pp, Yte_d)
		return r, r2, r2g
	r, r2, r2g = score(Xte)
	rng = np.random.default_rng(args.seed); T = len(Xte)
	nulls = [score(np.roll(Xte, int(rng.integers(30, T - 30)), axis=0)) for _ in range(args.n_shifts)]
	nr = [float(np.nanmean(n[0])) for n in nulls]; nr2 = [n[2] for n in nulls]
	swaps = [score(np.concatenate([t[0]] * int(np.ceil(T / len(t[0]))))[:T]) for t in va]  # other subjects' EEG
	sr = [float(np.nanmean(n[0])) for n in swaps]; sr2 = [n[2] for n in swaps]
	return {"best_epoch": best[2], "val_score": float(best[0]), "r_mean": float(np.nanmean(r)), "r_median": float(np.nanmedian(r)),
			"r2corr_mean": float(np.nanmean(r ** 2)), "R2_mean": float(np.nanmean(r2)), "R2_global": r2g, "R2_frac_pos": float(np.nanmean(r2 > 0)),
			"R2_max": float(np.nanmax(r2)), "r_per_parcel": r.tolist(), "R2_per_parcel": r2.tolist(), "shift_null_r": nr, "shift_null_R2_global": nr2,
			"p_shift": float((1 + sum(v >= np.nanmean(r) for v in nr)) / (1 + len(nr))),
			"p_shift_R2_descriptive_only": float((1 + sum(v >= r2g for v in nr2)) / (1 + len(nr2))),
			"swap_null_r": sr, "swap_null_R2_global": sr2, "detrended": True}


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--folds", nargs="*", type=int, default=None)
	ap.add_argument("--epochs", type=int, default=20)
	ap.add_argument("--bs", type=int, default=64)
	ap.add_argument("--select", default="R2", choices=["R2", "r"])
	ap.add_argument("--n_shifts", type=int, default=100)
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--tag", default="")
	args = ap.parse_args()
	assert torch.cuda.is_available(), "GPU-only rule"
	sys.path.insert(0, UNE)
	folds = make_folds()["fold_sets"]["loso"]["folds"]
	os.makedirs(RUNS, exist_ok=True)
	for i in (args.folds if args.folds is not None else range(len(folds))):
		out = os.path.join(RUNS, f"fold{i:02d}_seed{args.seed}{('_' + args.tag) if args.tag else ''}.json")
		if os.path.exists(out):
			continue
		t0 = time.time()
		r = run_fold(folds[i], args)
		r["minutes"] = (time.time() - t0) / 60
		print("unebolt", folds[i]["test"], {k: v for k, v in r.items() if not isinstance(v, list)}, flush=True)
		json.dump({"fold": folds[i], "args": vars(args), "results": r}, open(out, "w"), indent=1)
