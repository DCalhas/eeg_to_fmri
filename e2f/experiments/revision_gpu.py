"""GPU part of the revision analyses (Brain Informatics mock review 2026-10-28). Run inside GPU Slurm jobs only.

  --clean      synthesized [mean, SD] features of every clinical subject from the NODDI final synthesizers (trained
               and untrained) using only artefact-free pseudo-TRs (revision_clf.clean_mask)      -> <m>[_untrained]_clean
  --cnn_ep50   CNN-TAG trained on the same 14 NODDI subjects for all 50 epochs (no epoch selection; the selected
               final model stopped after epoch 0), then synthesized for every cohort              -> cnn_tag_ep50
  --examples   volumes of the held-out NODDI validation subject 49: real, EEG-free template and each final
               synthesizer, for the fidelity figure                                               -> runs/revision/examples.npz
Features are written to runs/synth_clf/<cohort>/<tag>.npz in the format of synth_clf (X, y, sub).
"""
import os, json, argparse
import numpy as np
import torch

from e2f.data import noddi, cohorts as C
from e2f.experiments.revision_clf import clean_mask, HARM, SYN, OUT
from e2f.synth.benchmark_noddi import build_model, subject_arrays, FINAL_VAL, ssim_mse_loss, predict

FINAL = os.path.expanduser("~/eeg_to_fmri/runs/synth_noddi/final")
COHORTS = ["fep1", "fep2", "semmelweis", "pd", "fribourg"]
MODELS = ["cnn_tag", "e2fnet", "spec2vol", "ntvit"]


def load_model(name, final_dir=FINAL, arm="aligned", untrained=False, path=None):
	ck = torch.load(path or os.path.join(final_dir, f"{arm}_{name}_seed0.pt"), map_location="cpu", weights_only=False)
	if name == "ntvit":
		from e2f.synth import ntvit_final as NT
		model = NT.build(ck["hparams"])
	else:
		model = build_model(name, tuple(ck["x_shape"]))
	if not untrained:
		model.load_state_dict(ck["state_dict"])
	return model.cuda().eval()


@torch.no_grad()
def subject_volumes(model, name, eeg, mu, sd, mask_artefacts=False):
	"""Synthesized volumes (V', 30, 64, 64) for one recording; optionally only artefact-free pseudo-TRs."""
	onsets = np.arange(20, int(eeg.shape[1] / 250 / noddi.TR)) * noddi.TR
	if name == "ntvit":
		from e2f.synth import ntvit_final as NT
		W, keep = NT.raw_windows(eeg, onsets)
		on = onsets[keep]
		if mask_artefacts:
			ok, _, _ = clean_mask(eeg, on)
			W = W[ok] if ok.sum() >= 5 else W
		return torch.from_numpy(NT.predict(model, W)).cuda()
	X, idx, _ = noddi.tr_frames(eeg, onsets, n_frames=20)
	if mask_artefacts:
		ok, _, _ = clean_mask(eeg, onsets[idx])
		X = X[ok] if ok.sum() >= 5 else X
	X = ((np.transpose(X, (0, 3, 1, 2)) - mu) / sd).astype(np.float32)
	return torch.cat([model(torch.from_numpy(X[i:i + 64]).cuda()).clamp(0, 1) for i in range(0, len(X), 64)])


def synthesize_cohorts(model, name, mu, sd, tag, mask_artefacts=False, cohorts=COHORTS):
	for cohort in cohorts:
		feats, labels, subs = [], [], []
		for rec in C.COHORTS[cohort]():
			eeg = np.load(os.path.join(HARM, cohort, f"{rec['sub']}.npz"))["eeg"]
			Y = subject_volumes(model, name, eeg, mu, sd, mask_artefacts)
			feats.append(torch.cat([Y.mean(0).flatten(), Y.std(0).flatten()]).cpu().numpy().astype(np.float32))
			labels.append(rec["label"]); subs.append(rec["sub"])
		os.makedirs(os.path.join(SYN, cohort), exist_ok=True)
		tmp = os.path.join(SYN, cohort, f"{tag}.tmp.npz")
		np.savez(tmp, X=np.stack(feats), y=np.array(labels), sub=np.array(subs))
		os.replace(tmp, os.path.join(SYN, cohort, f"{tag}.npz"))
		print("synthesized", cohort, tag, np.stack(feats).shape, flush=True)


def train_last_epoch(name, Xtr, Ytr, epochs, bs=32, lr=1e-3, seed=0):
	torch.manual_seed(seed); np.random.seed(seed)
	model = build_model(name, Xtr.shape[1:]).cuda()
	opt = torch.optim.Adam(model.parameters(), lr=lr); loss_fn = ssim_mse_loss()
	for ep in range(epochs):
		model.train()
		perm = np.random.permutation(len(Xtr))
		for i in range(0, len(perm), bs):
			b = perm[i:i + bs]
			x, y = torch.from_numpy(Xtr[b]).cuda(), torch.from_numpy(Ytr[b]).cuda()
			opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
	return model.eval()


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--clean", action="store_true")
	ap.add_argument("--cnn_ep50", action="store_true")
	ap.add_argument("--examples", action="store_true")
	ap.add_argument("--models", nargs="+", default=MODELS)
	a = ap.parse_args()
	assert torch.cuda.is_available(), "GPU-only rule"
	st = np.load(os.path.join(FINAL, "aligned_featstats.npz"))
	mu, sd = st["mu"], st["sd"]
	if a.clean:
		for m in a.models:
			for untrained in (False, True):
				synthesize_cohorts(load_model(m, untrained=untrained), m, mu, sd, f"{m}{'_untrained' if untrained else ''}_clean", mask_artefacts=True)
	if a.cnn_ep50:
		subs = noddi.subjects(); tr = [s for s in subs if s not in FINAL_VAL]
		arr = [subject_arrays(s, "aligned") for s in tr]
		Xtr = ((np.concatenate([x[0] for x in arr]) - mu) / sd).astype(np.float32); Ytr = np.concatenate([x[1] for x in arr])
		model = train_last_epoch("cnn_tag", Xtr, Ytr, epochs=50)
		Xva, Yva = subject_arrays(FINAL_VAL[0], "aligned")
		from e2f.eval import fidelity as F
		pv = predict(model, ((Xva - mu) / sd).astype(np.float32), "cuda")
		info = {"val_ssim_49": F.ssim_2d_e2fnet(np.transpose(np.clip(pv, 0, 1), (0, 2, 3, 1)), np.transpose(Yva, (0, 2, 3, 1))), "epochs": 50}
		torch.save({"state_dict": model.state_dict(), "info": info, "x_shape": Xtr.shape[1:], "arm": "aligned"}, os.path.join(FINAL, "aligned_cnn_tag_ep50_seed0.pt"))
		print("cnn_tag_ep50", info, flush=True)
		del Xtr, Ytr
		synthesize_cohorts(model, "cnn_tag", mu, sd, "cnn_tag_ep50")
	if a.examples:
		from e2f.synth import ntvit_final as NT
		from e2f.eval import fidelity as F
		to_xyz = lambda V: np.transpose(V, (0, 2, 3, 1))
		conv = ("rmse", "ssim_2d_e2fnet", "voxel_r")
		val = {}
		for sub in FINAL_VAL:  # final models: fidelity on the two held-out NODDI subjects, all four synthesizers
			Xv, Yv = subject_arrays(sub, "aligned")
			out = {"real": Yv, "template": st["template"]}
			val[sub] = {"template": F.score(to_xyz(st["template"][None])[0], to_xyz(Yv), conv)}
			for m in MODELS:
				model = load_model(m)
				if m == "ntvit":
					W, Y = NT.subject_pairs(sub)
					out[m] = NT.predict(model, W); out["real_ntvit"] = Y
					val[sub][m] = F.score(to_xyz(out[m]), to_xyz(Y), conv)
					val[sub]["template_ntvit_volumes"] = F.score(to_xyz(st["template"][None])[0], to_xyz(Y), conv)
				else:
					out[m] = np.clip(predict(model, ((Xv - mu) / sd).astype(np.float32), "cuda"), 0, 1)
					val[sub][m] = F.score(to_xyz(out[m]), to_xyz(Yv), conv)
				del model; torch.cuda.empty_cache()
			if sub == FINAL_VAL[0]:
				np.savez_compressed(os.path.join(OUT, "examples.npz"), **{k: v.astype(np.float16) for k, v in out.items()})
		json.dump(val, open(os.path.join(OUT, "final_models_val_fidelity.json"), "w"), indent=1, default=float)
		print("final-model val fidelity", json.dumps(val, default=float), flush=True)
