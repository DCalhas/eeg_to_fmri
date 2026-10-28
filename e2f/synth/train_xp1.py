"""Synthesizers retrained on a second simultaneous EEG-fMRI dataset, XP1 (e2f.data.xp1), and applied to the clinical
cohorts (review point 4: is the diagnostic null specific to NODDI?). GPU jobs only.

Protocol mirrors the NODDI final models (benchmark_noddi.train_final / ntvit_final.train_final): 8 training
subjects, epoch selection on the val SSIM of 2 held-out subjects (sub-xp109, sub-xp110), same losses, optimisers,
epochs and batch sizes; inputs are the same 2.16 s spectral frames (20-frame history) or 43.2 s raw windows, so the
clinical EEG is featurised exactly as for the NODDI models. Outputs:
  runs/synth_xp1/xp1_<model>_seed0.pt, xp1_featstats.npz, xp1_<model>_val.json (val fidelity vs the training template)
  runs/synth_clf/<cohort>/xp1_<model>.npz (synthesized [mean, SD] features, synth_clf format)
"""
import os, json, argparse, copy
import numpy as np
import torch

from e2f.data import xp1
from e2f.eval import fidelity as F

RUNS = os.path.expanduser("~/eeg_to_fmri/runs/synth_xp1")
VAL = ["sub-xp109", "sub-xp110"]


def split():
	subs = [s for s in xp1.subjects() if any(os.path.exists(os.path.join(xp1.OUT, f"{s}_{t}.npz")) for t in xp1.TASKS)]
	return [s for s in subs if s not in VAL], [s for s in VAL if s in subs]


def val_report(pred, Yva, template):
	to_xyz = lambda V: np.transpose(V, (0, 2, 3, 1))
	conv = ("rmse", "ssim_2d_e2fnet", "voxel_r")
	return {"model": F.score(to_xyz(pred), to_xyz(Yva), conv) if pred.ndim == 4 else None,
			"template": F.score(to_xyz(template[None])[0], to_xyz(Yva), conv)}


def train_conv(names, epochs, bs, lr):
	from e2f.synth.benchmark_noddi import train_torch, predict
	tr, va = split()
	a = [xp1.load_subject(s) for s in tr]; b = [xp1.load_subject(s) for s in va]
	Xtr, Ytr = np.concatenate([x[0] for x in a]), np.concatenate([x[1] for x in a]); del a
	Xva, Yva = np.concatenate([x[0] for x in b]), np.concatenate([x[1] for x in b]); del b
	mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
	template = Ytr.mean(0)
	os.makedirs(RUNS, exist_ok=True)
	np.savez(os.path.join(RUNS, "xp1_featstats.npz"), mu=mu, sd=sd, template=template, train=np.array(tr), val=np.array(va),
			 n_train_volumes=len(Xtr), n_val_volumes=len(Xva))
	z = lambda X: ((X - mu) / sd).astype(np.float32)
	Xtr, Xva = z(Xtr), z(Xva)
	print("xp1 data", Xtr.shape, Ytr.shape, Xva.shape, flush=True)
	for name in names:
		ep = epochs if name != "spec2vol" else min(epochs, 15)
		model, info = train_torch(name, Xtr, Ytr, Xva, Yva, ep, bs, lr, "cuda", 0)
		pred = np.clip(predict(model, Xva, "cuda"), 0, 1)
		info["val"] = val_report(pred, Yva, template)
		torch.save({"state_dict": model.state_dict(), "info": info, "x_shape": Xtr.shape[1:], "arm": "xp1"}, os.path.join(RUNS, f"xp1_{name}_seed0.pt"))
		json.dump(info, open(os.path.join(RUNS, f"xp1_{name}_val.json"), "w"), indent=1, default=float)
		print("xp1 final", name, json.dumps(info, default=float), flush=True)


class LazyWindows(torch.utils.data.Dataset):
	"""43.2 s raw windows cut on the fly from per-run EEG (avoids materialising ~20 GB of windows)."""
	def __init__(self, items, Y):
		self.items, self.Y = items, Y  # items: list of (eeg (C, N) array, end sample)
		self.L = int(np.ceil(250 * 20 * 2.16))
	def __len__(self):
		return len(self.items)
	def __getitem__(self, i):
		eeg, e = self.items[i]
		return {"eegs": torch.from_numpy(np.ascontiguousarray(eeg[:, e - self.L:e])), "fmris": torch.from_numpy(self.Y[i])}


def ntvit_items(subs):
	items, Ys = [], []
	L = int(np.ceil(250 * 20 * 2.16))
	for s in subs:
		for t in xp1.TASKS:
			f = os.path.join(xp1.OUT, f"{s}_{t}.npz")
			if not os.path.exists(f):
				continue
			d = np.load(f); eeg = d["eeg"]
			ends = np.round(d["onsets"] * 250).astype(int)
			keep = [k for k, e in enumerate(ends) if e - L >= 0 and e <= eeg.shape[1]]
			Y = d["fmri"][keep]
			lo, hi = Y.min(axis=(1, 2, 3), keepdims=True), Y.max(axis=(1, 2, 3), keepdims=True)
			Ys.append(np.transpose((Y - lo) / (hi - lo + 1e-10), (0, 3, 1, 2)).astype(np.float32))
			items += [(eeg, ends[k]) for k in keep]
	return items, np.concatenate(Ys)


def train_ntvit(epochs, bs):
	import pytorch_lightning as pl
	from e2f.synth import ntvit_final as NT
	tr, va = split()
	pl.seed_everything(0)
	itr, Ytr = ntvit_items(tr); iva, Yva = ntvit_items(va)
	Wva = np.stack([eeg[:, e - LazyWindows(iva, Yva).L:e] for eeg, e in iva]).astype(np.float32)
	template = Ytr.mean(0)
	model = NT.build()
	best = {"ssim": -np.inf, "state": None, "epoch": -1}
	class Select(pl.Callback):
		def on_train_epoch_end(self, trainer, module):
			p = NT.predict(module, Wva)
			s = F.ssim_2d_e2fnet(np.transpose(p, (0, 2, 3, 1)), np.transpose(Yva, (0, 2, 3, 1)))
			print("epoch", trainer.current_epoch, "val SSIM", round(s, 4), flush=True)
			if s > best["ssim"]:
				best.update(ssim=s, state=copy.deepcopy({k: v.detach().cpu() for k, v in module.state_dict().items()}), epoch=trainer.current_epoch)
			module.train()
	trainer = pl.Trainer(devices=1, accelerator="gpu", precision="16-mixed", logger=False, max_epochs=epochs, callbacks=[Select()],
						 enable_checkpointing=False, enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0)
	trainer.fit(model, torch.utils.data.DataLoader(LazyWindows(itr, Ytr), batch_size=bs, shuffle=True, num_workers=0, pin_memory=True))
	model.load_state_dict(best["state"])
	pred = NT.predict(model.cuda(), Wva)
	info = {"val_ssim": best["ssim"], "best_epoch": best["epoch"], "val": val_report(pred, Yva, template)}
	os.makedirs(RUNS, exist_ok=True)
	torch.save({"state_dict": best["state"], "hparams": NT.HPARAMS, "info": info}, os.path.join(RUNS, "xp1_ntvit_seed0.pt"))
	json.dump(info, open(os.path.join(RUNS, "xp1_ntvit_val.json"), "w"), indent=1, default=float)
	print("xp1 final ntvit", json.dumps(info, default=float), flush=True)


def synthesize(names):
	from e2f.experiments.revision_gpu import load_model, synthesize_cohorts
	st = np.load(os.path.join(RUNS, "xp1_featstats.npz"))
	for name in names:
		model = load_model(name, path=os.path.join(RUNS, f"xp1_{name}_seed0.pt"))
		synthesize_cohorts(model, name, st["mu"], st["sd"], f"xp1_{name}")


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--models", nargs="+", default=["cnn_tag", "e2fnet", "spec2vol"])
	ap.add_argument("--ntvit", action="store_true")
	ap.add_argument("--synthesize", action="store_true")
	ap.add_argument("--epochs", type=int, default=50)
	ap.add_argument("--bs", type=int, default=32)
	ap.add_argument("--lr", type=float, default=1e-3)
	a = ap.parse_args()
	assert torch.cuda.is_available(), "GPU-only rule"
	if a.ntvit and not a.synthesize:
		train_ntvit(epochs=30, bs=32)
	elif a.models and not a.synthesize:
		train_conv(a.models, a.epochs, a.bs, a.lr)
	if a.synthesize:
		synthesize(a.models + (["ntvit"] if a.ntvit else []))
