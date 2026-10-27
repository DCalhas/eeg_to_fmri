"""NT-ViT (Lanzino et al., ECCVW 2024) as a fourth synthesizer for the classification benchmark (GPU job).

The authors' model is imported read-only from ~/eeg_to_fmri/baselines/ntvit, with the compatibility shims written by
the audit session (~/eeg_to_fmri/audit/ntvit_common.py, 2026-10-24; vendored below): thop/wandb stand-ins (MAC
counting and image logging only), the removed `verbose` argument of CosineAnnealingWarmRestarts, and the old default
data_range of torchmetrics' functional PSNR. Hyper-parameters are the README/train.py setting used by the audit
(domain matching on, 3 layers, h_dim 256, AdamW lr 5e-4, wd 1e-2, batch 128, 30 epochs, 16-mixed precision).
Changes from the published protocol (ledger A16/A18):
  input   trigger-aligned NODDI EEG (57 common channels by name, 0.5-45 Hz, 250 Hz) as raw waveforms, a 43.2 s window
          (20 TRs) ending at each volume onset, i.e. the same EEG as the other synthesizers; the authors used a
          simultaneous 2.16 s window (zero haemodynamic lag) cropped by an amplitude threshold from gradient-artefact EEG
  target  the volume min-max scaled to [0,1] (30, 64, 64)
  split   14 training subjects; epoch selected on val SSIM of subjects 49 and 50 (benchmark_noddi.FINAL_VAL)
"""
import os, sys, types, json, argparse, copy
import numpy as np
import torch

NTVIT = os.path.expanduser("~/eeg_to_fmri/baselines/ntvit")
FINAL = os.path.expanduser("~/eeg_to_fmri/runs/synth_noddi/final")
WIN_TR = 20


def _shims():
	thop = types.ModuleType("thop")
	def _no_profile(*a, **k):
		raise RuntimeError("thop not installed (MAC counting skipped)")
	thop.profile = _no_profile
	sys.modules.setdefault("thop", thop)
	wandb = types.ModuleType("wandb")
	class _Image:
		def __init__(self, *a, **k):
			raise RuntimeError("wandb not installed; run with plot_images=False")
	wandb.Image, wandb.log, wandb.login = _Image, (lambda *a, **k: None), (lambda *a, **k: None)
	sys.modules.setdefault("wandb", wandb)
	import torch.optim.lr_scheduler as _lrs
	class _CAWR(_lrs.CosineAnnealingWarmRestarts):
		def __init__(self, *a, verbose=None, **k):
			super().__init__(*a, **k)
	_lrs.CosineAnnealingWarmRestarts = _CAWR
	import torchmetrics.functional.image as _tmi
	_psnr = _tmi.peak_signal_noise_ratio
	def _psnr_old_default(preds, target, data_range=None, *a, **k):
		if data_range is None:
			data_range = (target.max() - target.min()).item()
		return _psnr(preds, target, data_range, *a, **k)
	_tmi.peak_signal_noise_ratio = _psnr_old_default
	_n = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
	os.cpu_count = lambda: _n
	if NTVIT not in sys.path:
		sys.path.insert(0, NTVIT)


HPARAMS = dict(fmris_shape=(30, 64, 64), eegs_seconds=WIN_TR * 2.16, eegs_channels=57, eegs_sampling_rate=250,
			   fmris_downsampling_factor=1, normalized_fmris=True, spectrogram_scale=2, fmri_patches_size=6, eeg_patches_size=8,
			   use_domain_matching=True, use_discriminator=False, alpha_disc=1.0, activation="selu", layers=3, h_dim=256,
			   dropout=0.0, input_noise=0.0, learning_rate=5e-4, weight_decay=1e-2, use_lr_scheduler=False, use_kl_loss=False,
			   use_cm_loss=False, use_disp_loss=False, plot_images=False)


def build(hparams=HPARAMS):
	_shims()
	from ntvit import NTViT
	return NTViT(**hparams)


def raw_windows(eeg, onsets, sfreq=250, n_tr=WIN_TR, tr=2.16):
	"""eeg (C, N) -> (V, C, L) raw windows of n_tr TRs ending at each onset; returns windows and kept volume indices."""
	L = int(np.ceil(sfreq * n_tr * tr))
	ends = np.round(np.asarray(onsets) * sfreq).astype(int)
	keep = [k for k, e in enumerate(ends) if e - L >= 0 and e <= eeg.shape[1]]
	return np.stack([eeg[:, ends[k] - L:ends[k]] for k in keep]).astype(np.float32), np.array(keep)


def subject_pairs(sub):
	from e2f.data import noddi
	d = noddi.load(sub)
	W, idx = raw_windows(d["eeg"], d["onsets"])
	Y = d["fmri"][idx]
	lo, hi = Y.min(axis=(1, 2, 3), keepdims=True), Y.max(axis=(1, 2, 3), keepdims=True)
	Y = np.transpose((Y - lo) / (hi - lo + 1e-10), (0, 3, 1, 2)).astype(np.float32)  # (V, 30, 64, 64)
	return W, Y


class Pairs(torch.utils.data.Dataset):
	def __init__(self, W, Y):
		self.W, self.Y = W, Y
	def __len__(self):
		return len(self.W)
	def __getitem__(self, i):
		return {"eegs": torch.from_numpy(self.W[i]), "fmris": torch.from_numpy(self.Y[i])}


@torch.no_grad()
def predict(model, W, device="cuda", bs=16):
	model.eval()
	out = []
	for i in range(0, len(W), bs):
		e = torch.from_numpy(W[i:i + bs]).to(device)
		with torch.autocast("cuda", dtype=torch.float16):
			o = model({"eegs": e, "fmris": torch.zeros((len(e),) + tuple(HPARAMS["fmris_shape"]), device=device)})
		out.append(o["fmris_rec"].float().clamp(0, 1).cpu().numpy())
	return np.concatenate(out)


def train_final(epochs=30, bs=128, seed=0):
	import pytorch_lightning as pl
	from e2f.data import noddi
	from e2f.synth.benchmark_noddi import FINAL_VAL
	from e2f.eval import fidelity as F
	assert torch.cuda.is_available(), "GPU-only rule"
	pl.seed_everything(seed)
	subs = noddi.subjects(); tr = [s for s in subs if s not in FINAL_VAL]
	a = [subject_pairs(s) for s in tr]; b = [subject_pairs(s) for s in FINAL_VAL]
	Wtr, Ytr = np.concatenate([x[0] for x in a]), np.concatenate([x[1] for x in a])
	Wva, Yva = np.concatenate([x[0] for x in b]), np.concatenate([x[1] for x in b])
	model = build()
	best = {"ssim": -np.inf, "state": None, "epoch": -1}
	class Select(pl.Callback):
		def on_train_epoch_end(self, trainer, module):
			p = predict(module, Wva)
			s = F.ssim_2d_e2fnet(np.transpose(p, (0, 2, 3, 1)), np.transpose(Yva, (0, 2, 3, 1)))
			print("epoch", trainer.current_epoch, "val SSIM", round(s, 4), flush=True)
			if s > best["ssim"]:
				best.update(ssim=s, state=copy.deepcopy({k: v.detach().cpu() for k, v in module.state_dict().items()}), epoch=trainer.current_epoch)
			module.train()
	trainer = pl.Trainer(devices=1, accelerator="gpu", precision="16-mixed", logger=False, max_epochs=epochs, callbacks=[Select()],
						 enable_checkpointing=False, enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0)
	trainer.fit(model, torch.utils.data.DataLoader(Pairs(Wtr, Ytr), batch_size=bs, shuffle=True, num_workers=4, pin_memory=True))
	os.makedirs(FINAL, exist_ok=True)
	torch.save({"state_dict": best["state"], "hparams": HPARAMS, "info": {"val_ssim": best["ssim"], "best_epoch": best["epoch"]}},
			   os.path.join(FINAL, f"aligned_ntvit_seed{seed}.pt"))
	print("final ntvit", best["ssim"], best["epoch"], flush=True)


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--epochs", type=int, default=30)
	ap.add_argument("--bs", type=int, default=128)
	a = ap.parse_args()
	train_final(a.epochs, a.bs)
