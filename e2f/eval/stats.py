"""Subject-level diagnostic metrics and paired comparisons.

All functions take window-level scores plus a subject id per window and aggregate to one
score per subject before computing anything: windows of the same subject are not
independent, so window-pooled AUCs overstate the effective sample size.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss


def aggregate(scores, labels, subjects, how="mean"):
	"""Collapse window-level scores to one score per subject (in sorted subject order)."""
	scores, labels, subjects = map(np.asarray, (scores, labels, subjects))
	subs = np.unique(subjects)
	agg = {"mean": np.mean, "median": np.median}[how]
	s = np.array([agg(scores[subjects == u]) for u in subs])
	y = np.array([labels[subjects == u][0] for u in subs])
	assert all((labels[subjects == u] == labels[subjects == u][0]).all() for u in subs), "label varies within subject"
	return s, y, subs


def classification_metrics(p, y, threshold=0.5):
	"""p: probabilities per subject, y: 0/1 labels."""
	p, y = np.asarray(p, float), np.asarray(y, int)
	pred = (p >= threshold).astype(int)
	tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
	fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
	sens = tp / (tp + fn) if tp + fn else np.nan
	spec = tn / (tn + fp) if tn + fp else np.nan
	return {"auc": roc_auc_score(y, p), "acc": (tp + tn) / len(y), "balanced_acc": np.nanmean([sens, spec]),
			"sensitivity": sens, "specificity": spec, "brier": brier_score_loss(y, np.clip(p, 0, 1)),
			"n": len(y), "n_pos": int(y.sum())}


def bootstrap_ci(p, y, metric=roc_auc_score, n_boot=10000, alpha=0.05, seed=0):
	"""Stratified percentile bootstrap over subjects."""
	p, y = np.asarray(p, float), np.asarray(y, int)
	rng = np.random.default_rng(seed)
	pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
	vals = np.empty(n_boot)
	for b in range(n_boot):
		idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
		vals[b] = metric(y[idx], p[idx])
	return np.quantile(vals, [alpha / 2, 1 - alpha / 2])


def _midrank(x):
	return stats.rankdata(x, method="average")


def delong(p1, p2, y):
	"""DeLong test for two correlated AUCs on the same subjects. Returns (auc1, auc2, z, p)."""
	p1, p2, y = np.asarray(p1, float), np.asarray(p2, float), np.asarray(y, int)
	pos, neg = y == 1, y == 0
	m, n = pos.sum(), neg.sum()
	aucs, v10, v01 = [], [], []
	for p in (p1, p2):
		tx, ty, tz = _midrank(p[pos]), _midrank(p[neg]), _midrank(np.concatenate([p[pos], p[neg]]))
		aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
		v10.append((tz[:m] - tx) / n)
		v01.append(1 - (tz[m:] - ty) / m)
	s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
	var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
	z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else np.nan
	return aucs[0], aucs[1], z, 2 * stats.norm.sf(abs(z))


def paired_bootstrap_delta(p1, p2, y, n_boot=10000, alpha=0.05, seed=0):
	"""CI for AUC(p1) - AUC(p2), resampling subjects jointly."""
	p1, p2, y = np.asarray(p1, float), np.asarray(p2, float), np.asarray(y, int)
	rng = np.random.default_rng(seed)
	pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
	d = np.empty(n_boot)
	for b in range(n_boot):
		idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
		d[b] = roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx])
	return roc_auc_score(y, p1) - roc_auc_score(y, p2), np.quantile(d, [alpha / 2, 1 - alpha / 2])


def calibration_bins(p, y, n_bins=5):
	"""Reliability curve with quantile bins (few subjects -> few bins)."""
	p, y = np.asarray(p, float), np.asarray(y, int)
	edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
	idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
	return [(p[idx == b].mean(), y[idx == b].mean(), int((idx == b).sum())) for b in range(n_bins) if (idx == b).any()]
