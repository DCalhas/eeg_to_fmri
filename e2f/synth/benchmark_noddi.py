"""RQ1: EEG-to-fMRI synthesis fidelity on NODDI under a corrected protocol.

Protocol
  data      trigger-aligned EEG (e2f.data.noddi), 20 TR-long log|FFT| frames (0.46-45 Hz) per volume, no lag;
            fMRI target = volume min-max [0,1] (E2fNet convention, comparable SSIM/PSNR), shape (30, 64, 64)
  folds     data/noddi/folds.json "loso": test = 1 subject, val = next 2 (cyclic), train = rest.
            Model selection / early stopping on val only (ledger A8).
  arms      aligned (scanner triggers) | filestart (published protocol: frames indexed from the EEG file start)
            | ecg_only (ECG frames replicated over channels; ledger A1)
  models    template (no EEG) | ridge | cnn_tag (ours, PyTorch port) | e2fnet | spec2vol
  scoring   e2f.eval.fidelity: RMSE, SSIM-2D (E2fNet), voxel temporal r, plus own_mean and the circular-shift null (A9)
EEG features are standardized with training-subject statistics only.
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn

from e2f.data import noddi, preprocess as P
from e2f.eval import fidelity as F

BASE = os.path.expanduser("~/eeg_to_fmri/baselines")
RUNS = os.path.expanduser("~/eeg_to_fmri/runs/synth_noddi")
FOLDS = os.path.expanduser("~/eeg_to_fmri/data/noddi/folds.json")


# ---------------------------------------------------------------- data
def make_folds():
	subs = noddi.subjects()
	folds = []
	for i, s in enumerate(subs):
		val = [subs[(i + 1) % len(subs)], subs[(i + 2) % len(subs)]]
		folds.append({"test": [s], "val": val, "train": [u for u in subs if u != s and u not in val]})
	d = {"dataset": "noddi", "subjects": subs, "created": time.strftime("%Y-%m-%d"), "created_by": "paper S (rebuild-2026)",
		 "fold_sets": {"loso": {"description": "LOSO; val = next 2 subjects (cyclic) for model selection", "folds": folds},
					   "e2fnet_fixed": {"description": "E2fNet run_train.sh: test 43; val = 2 subjects for selection",
										"folds": [{"test": ["43"], "val": ["44", "45"], "train": [u for u in subs if u not in ("43", "44", "45")]}]}}}
	if not os.path.exists(FOLDS):
		json.dump(d, open(FOLDS, "w"), indent=1)
	return json.load(open(FOLDS))


def subject_arrays(sub, arm, n_frames=20):
	d = noddi.load(sub)
	if arm == "aligned":
		onsets = d["onsets"]
	elif arm == "filestart":
		onsets = np.arange(300) * noddi.TR  # EEG frame k <-> volume k from the file start, as in published pipelines
	elif arm == "ecg_only":
		onsets = d["onsets"]
	sig = d["ecg"] if arm == "ecg_only" else d["eeg"]
	X, idx, _ = noddi.tr_frames(sig, onsets, n_frames=n_frames)
	if arm == "ecg_only":
		X = np.repeat(X, len(P.NODDI_CHANNELS), axis=1)
	Y = d["fmri"][idx]
	lo, hi = Y.min(axis=(1, 2, 3), keepdims=True), Y.max(axis=(1, 2, 3), keepdims=True)
	Y = (Y - lo) / (hi - lo + 1e-10)
	return np.transpose(X, (0, 3, 1, 2)), np.transpose(Y, (0, 3, 1, 2)).astype(np.float32)  # X (V,frames,C,F); Y (V,30,64,64)


# ---------------------------------------------------------------- models
def build_model(name, x_shape):
	T, C, Fq = x_shape
	if name == "e2fnet":
		sys.path.insert(0, os.path.join(BASE, "E2fNet"))
		from models import create_unet, EEGEncoder, fMRIDecoder, EEG2fMRINet
		return EEG2fMRINet(EEGEncoder(in_channels=T, img_size=64), create_unet(256, 256), fMRIDecoder(256, 30))
	if name == "spec2vol":
		sys.path.insert(0, os.path.join(BASE, "Spec2VolCAMU-Net"))
		from Spectrogram2fMRI import Spectrogram2fMRI
		return Spectrogram2fMRI(in_dim=T, out_dim=30)
	if name == "cnn_tag":
		from e2f.models.cnn_tag import CNNTAG
		class Wrap(nn.Module):
			def __init__(self):
				super().__init__()
				self.m = CNNTAG(eeg_shape=(C, Fq, T), fmri_shape=(64, 64, 30))
			def forward(self, x):  # x (B,T,C,F) -> (B,1,C,F,T); out (B,1,64,64,30) -> (B,30,64,64)
				y, _ = self.m(x.permute(0, 2, 3, 1).unsqueeze(1))
				return y[:, 0].permute(0, 3, 1, 2)
		return Wrap()
	raise ValueError(name)


def ssim_mse_loss():
	from torchmetrics.image import StructuralSimilarityIndexMeasure
	ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
	mse = nn.MSELoss()
	def f(p, t):
		return 0.5 * (1 - ssim.to(p.device)(p, t)) + 0.5 * mse(p, t)
	return f


def train_torch(name, Xtr, Ytr, Xva, Yva, epochs, bs, lr, device, seed):
	torch.manual_seed(seed); np.random.seed(seed)
	model = build_model(name, Xtr.shape[1:]).to(device)
	opt = torch.optim.Adam(model.parameters(), lr=lr)
	loss_fn = ssim_mse_loss()
	best, best_state, best_ep = -np.inf, None, -1
	Xva_t = torch.from_numpy(Xva)
	for ep in range(epochs):
		model.train()
		perm = np.random.permutation(len(Xtr))  # shuffled across subjects (ledger C6)
		for i in range(0, len(perm), bs):
			b = perm[i:i + bs]
			x, y = torch.from_numpy(Xtr[b]).to(device), torch.from_numpy(Ytr[b]).to(device)
			opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
		pv = predict(model, Xva_t, device)
		score = F.ssim_2d_e2fnet(np.transpose(pv, (0, 2, 3, 1)), np.transpose(Yva, (0, 2, 3, 1)))
		if score > best:
			best, best_ep = score, ep
			best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
	model.load_state_dict(best_state)
	return model, {"val_ssim": best, "best_epoch": best_ep}


@torch.no_grad()
def predict(model, X, device, bs=64):
	model.eval()
	X = X if torch.is_tensor(X) else torch.from_numpy(X)
	return np.concatenate([model(X[i:i + bs].to(device)).float().cpu().numpy() for i in range(0, len(X), bs)])


def ridge_fit(Xtr, Ytr, Xva, Yva, n_x=256, n_y=256):
	from sklearn.decomposition import PCA
	from sklearn.linear_model import Ridge
	xs = Xtr.reshape(len(Xtr), -1); ys = Ytr.reshape(len(Ytr), -1)
	px = PCA(n_x, random_state=0).fit(xs); py = PCA(n_y, random_state=0).fit(ys)
	zx, zy = px.transform(xs), py.transform(ys)
	best = None
	for a in [1e1, 1e2, 1e3, 1e4, 1e5]:
		r = Ridge(alpha=a).fit(zx, zy)
		p = py.inverse_transform(r.predict(px.transform(Xva.reshape(len(Xva), -1)))).reshape(Yva.shape)
		s = F.ssim_2d_e2fnet(np.clip(np.transpose(p, (0, 2, 3, 1)), 0, 1), np.transpose(Yva, (0, 2, 3, 1)))
		if best is None or s > best[0]:
			best = (s, a, r)
	pred = lambda X: np.clip(py.inverse_transform(best[2].predict(px.transform(X.reshape(len(X), -1)))).reshape((len(X),) + Ytr.shape[1:]), 0, 1)
	return pred, {"val_ssim": best[0], "alpha": best[1]}


# ---------------------------------------------------------------- run
def run_fold(fold, arm, models, args):
	load = lambda subs: [subject_arrays(s, arm) for s in subs]
	tr, va, te = load(fold["train"]), load(fold["val"]), load(fold["test"])
	Xtr, Ytr = np.concatenate([a[0] for a in tr]), np.concatenate([a[1] for a in tr])
	Xva, Yva = np.concatenate([a[0] for a in va]), np.concatenate([a[1] for a in va])
	Xte, Yte = te[0]
	mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
	z = lambda X: ((X - mu) / sd).astype(np.float32)
	Xtr, Xva, Xte = z(Xtr), z(Xva), z(Xte)
	to_xyz = lambda V: np.transpose(V, (0, 2, 3, 1))  # (V,30,64,64) -> (V,64,64,30) for fidelity (Z last)
	target = to_xyz(Yte)
	res = {"template": F.score(to_xyz(Ytr).mean(0), target), "own_mean": F.score(target.mean(0), target)}
	device = "cuda" if torch.cuda.is_available() else "cpu"
	for name in models:
		t0 = time.time()
		if name == "ridge":
			pred_fn, info = ridge_fit(Xtr, Ytr, Xva, Yva)
		else:
			model, info = train_torch(name, Xtr, Ytr, Xva, Yva, args.epochs, args.bs, args.lr, device, args.seed)
			pred_fn = lambda X, model=model: np.clip(predict(model, X, device), 0, 1)
		nr = F.null_report(lambda X: to_xyz(pred_fn(X)), Xte, target, n_shifts=args.n_shifts, min_shift=30, rng=args.seed,
						   conventions=("rmse", "ssim_2d_e2fnet", "voxel_r"))
		res[name] = {**nr["real"], **info, "shift_null_mean": {k: float(np.mean(v)) for k, v in nr["shift_null"].items()},
					 "p_voxel_r": nr["p_voxel_r"], "minutes": (time.time() - t0) / 60}
		print(arm, fold["test"], name, json.dumps({k: v for k, v in res[name].items() if not isinstance(v, dict)}), flush=True)
	return res


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--arm", default="aligned", choices=["aligned", "filestart", "ecg_only"])
	ap.add_argument("--models", nargs="+", default=["ridge", "cnn_tag", "e2fnet", "spec2vol"])
	ap.add_argument("--fold_set", default="loso")
	ap.add_argument("--folds", nargs="*", type=int, default=None)
	ap.add_argument("--epochs", type=int, default=50)
	ap.add_argument("--bs", type=int, default=32)
	ap.add_argument("--lr", type=float, default=1e-3)
	ap.add_argument("--n_shifts", type=int, default=20)
	ap.add_argument("--seed", type=int, default=0)
	a = ap.parse_args()
	folds = make_folds()["fold_sets"][a.fold_set]["folds"]
	sel = a.folds if a.folds is not None else range(len(folds))
	os.makedirs(RUNS, exist_ok=True)
	for i in sel:
		out = os.path.join(RUNS, f"{a.arm}_{a.fold_set}_fold{i:02d}_seed{a.seed}.json")
		if os.path.exists(out):
			continue
		r = run_fold(folds[i], a.arm, a.models, a)
		json.dump({"fold": folds[i], "arm": a.arm, "args": vars(a), "results": r}, open(out, "w"), indent=1, default=float)
