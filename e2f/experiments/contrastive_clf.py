"""The 2023 contrastive fine-tuning method, re-evaluated on the clean clinical cohorts (GPU job).

Method (Calhas & Henriques 2023, medRxiv; PyTorch port in e2f.models.cnn_tag): the CNN-TAG synthesizer's encoder is
trained with a contrastive loss on the sinusoid (random-Fourier-feature) space so that same-label windows are close
and different-label windows are at margin m = 0.5, while a linear classifier (L1) is trained on the synthesized
fMRI view (the view is detached for the classifier, as in the original code); the voxel decoder (style vector,
dense, 1x1 conv) stays frozen at its NODDI-trained weights.
Variants:
  legacy2023   encoder, dense latent and RFF freshly initialised (what the 2023 code actually did)
  pretrained   encoder and RFF initialised from the NODDI-trained synthesizer (the learned mapping), then fine-tuned
Data: harmonized clinical EEG -> pseudo-TR frames (20 x 57 x 97, NODDI feature statistics), one instance per
synthesized volume; subject score = mean predicted probability over its windows.
Validation: repeated stratified 10-fold at the subject level (same splits as synth_clf for a given seed), a fixed
training budget (10 epochs, Adam lr 1e-3, batch 16 pairs, L1 1e-4: the 2023 defaults' centre, no tuning), AUC with
bootstrap CI and a label-permutation p (full CV per permutation, 1 repeat).
"""
import os, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

from e2f.data import noddi, cohorts as C
from e2f.eval import stats
from e2f.models.cnn_tag import LegacyClassificationView, legacy_contrastive_loss
from e2f.synth.benchmark_noddi import build_model

HARM = os.path.expanduser("~/eeg_to_fmri/runs/harmonized_v2")
FINAL = os.path.expanduser("~/eeg_to_fmri/runs/synth_noddi/final")
OUT = os.path.expanduser("~/eeg_to_fmri/runs/contrastive_clf")


def cohort_windows(cohort, mu, sd):
	Xs, y, subs = [], [], []
	for rec in C.COHORTS[cohort]():
		eeg = np.load(os.path.join(HARM, cohort, f"{rec['sub']}.npz"))["eeg"]
		onsets = np.arange(20, int(eeg.shape[1] / 250 / noddi.TR)) * noddi.TR
		X, _, _ = noddi.tr_frames(eeg, onsets, n_frames=20)
		X = ((np.transpose(X, (0, 3, 1, 2)) - mu) / sd).astype(np.float32)  # (V, 20, 57, 97)
		Xs.append(np.transpose(X, (0, 2, 3, 1))[:, None])  # (V, 1, 57, 97, 20) for the CNN-TAG encoder
		y.append(rec["label"]); subs.append(rec["sub"])
	return Xs, np.array(y), subs


def load_synth():
	ck = torch.load(os.path.join(FINAL, "aligned_cnn_tag_seed0.pt"), map_location="cpu", weights_only=False)
	w = build_model("cnn_tag", tuple(ck["x_shape"])); w.load_state_dict(ck["state_dict"])
	return w.m


class Model(nn.Module):
	def __init__(self, synth, variant):
		super().__init__()
		self.view = LegacyClassificationView(synth, (57, 97, 20), use_pretrained_encoder=(variant == "pretrained"))
		self.clf = nn.Linear(64 * 64 * 30, 1)
		nn.init.xavier_uniform_(self.clf.weight); nn.init.zeros_(self.clf.bias)


def train_fold(synth, variant, Xtr, ytr, epochs, lr, l1, seed, device):
	torch.manual_seed(seed); rng = np.random.default_rng(seed)
	m = Model(synth, variant).to(device)
	params = [p for p in m.parameters() if p.requires_grad]
	opt = torch.optim.Adam(params, lr=lr)
	bce = nn.BCEWithLogitsLoss()
	X = np.concatenate(Xtr); y = np.concatenate([np.full(len(x), l) for x, l in zip(Xtr, ytr)]).astype(np.float32)
	n_pairs = len(X) // 2
	for ep in range(epochs):
		m.train()
		i1 = rng.integers(0, len(X), n_pairs); i2 = rng.integers(0, len(X), n_pairs)
		for b in range(0, n_pairs, 16):
			a, c = i1[b:b + 16], i2[b:b + 16]
			x1, x2 = torch.from_numpy(X[a]).to(device), torch.from_numpy(X[c]).to(device)
			y1, y2 = torch.from_numpy(y[a]).to(device), torch.from_numpy(y[c]).to(device)
			v1, s1 = m.view(x1); v2, s2 = m.view(x2)
			loss_c = legacy_contrastive_loss(s1, s2, (y1 == y2)).mean()
			l1_1, l1_2 = m.clf(v1.detach().flatten(1)).squeeze(1), m.clf(v2.detach().flatten(1)).squeeze(1)  # detached view, as 2023
			loss = loss_c + bce(l1_1, y1) + bce(l1_2, y2) + l1 * m.clf.weight.abs().sum()
			opt.zero_grad(); loss.backward(); opt.step()
	return m


@torch.no_grad()
def subject_scores(m, Xte, device):
	m.eval()
	out = []
	for X in Xte:
		p = torch.cat([torch.sigmoid(m.clf(m.view(torch.from_numpy(X[i:i + 64]).to(device))[0].flatten(1))).squeeze(1)
					   for i in range(0, len(X), 64)])
		out.append(float(p.mean()))
	return np.array(out)


def cv(synth, variant, Xs, y, repeats, args, device, seed=0):
	P = np.zeros((repeats, len(y)))
	for i, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=10, n_repeats=repeats, random_state=seed).split(np.zeros(len(y)), y)):
		m = train_fold(synth, variant, [Xs[j] for j in tr], y[tr], args.epochs, args.lr, args.l1, seed * 1000 + i, device)
		P[i // 10, te] = subject_scores(m, [Xs[j] for j in te], device)
		del m; torch.cuda.empty_cache()
	return float(np.mean([roc_auc_score(y, p) for p in P])), P.mean(0)


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--cohorts", nargs="+", default=["pd", "fep1", "fep2", "semmelweis", "fribourg"])
	ap.add_argument("--variants", nargs="+", default=["pretrained", "legacy2023"])
	ap.add_argument("--repeats", type=int, default=2)
	ap.add_argument("--perm", type=int, default=20)
	ap.add_argument("--epochs", type=int, default=10)
	ap.add_argument("--lr", type=float, default=1e-3)
	ap.add_argument("--l1", type=float, default=1e-4)
	ap.add_argument("--out_dir", default=OUT)
	a = ap.parse_args()
	OUT = a.out_dir
	assert torch.cuda.is_available(), "GPU-only rule"
	dev = "cuda"
	os.makedirs(OUT, exist_ok=True)
	st = np.load(os.path.join(FINAL, "aligned_featstats.npz"))
	synth = load_synth()
	for c in a.cohorts:
		Xs, y, subs = cohort_windows(c, st["mu"], st["sd"])
		for v in a.variants:
			out = os.path.join(OUT, f"{c}_{v}.json")
			if os.path.exists(out):
				continue
			t0 = time.time()
			auc, p = cv(synth, v, Xs, y, a.repeats, a, dev)
			rng = np.random.default_rng(0)
			null = [cv(synth, v, Xs, rng.permutation(y), 1, a, dev, seed=b + 1)[0] for b in range(a.perm)]
			res = {"cohort": c, "variant": v, "auc": auc, "auc_ci95": list(stats.bootstrap_ci(p, y, n_boot=2000)),
				   "perm_p_greater": float((1 + sum(n >= auc for n in null)) / (1 + len(null))), "perm_null": null,
				   "subject_prob": p.tolist(), "y": y.tolist(), "subjects": subs, "args": vars(a), "minutes": (time.time() - t0) / 60}
			json.dump(res, open(out, "w"), indent=1)
			print(c, v, {k: res[k] for k in ("auc", "auc_ci95", "perm_p_greater", "minutes")}, flush=True)
