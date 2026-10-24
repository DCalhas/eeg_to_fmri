"""IAAFT surrogates (Schreiber & Schmitz 1996) for the BOLD-target null of Mehta et al. (arXiv 2504.10752):
preserve each time series' amplitude distribution and (approximately) its power spectrum, destroy its
relation to the EEG. Applied per column; the model is refit (and re-tuned) on (real EEG, surrogate BOLD)."""
import numpy as np


def iaaft(x, n_iter=100, rng=None):
	"""x: (T,) or (T, K). Returns one surrogate with the same shape."""
	rng = np.random.default_rng(rng)
	x = np.asarray(x, np.float64)
	one_d = x.ndim == 1
	X = x[:, None] if one_d else x
	T, K = X.shape
	amp = np.abs(np.fft.rfft(X, axis=0))
	sorted_x = np.sort(X, axis=0)
	S = np.take_along_axis(X, np.argsort(rng.random((T, K)), axis=0), axis=0)  # random shuffle per column
	for _ in range(n_iter):
		ph = np.angle(np.fft.rfft(S, axis=0))
		S = np.fft.irfft(amp * np.exp(1j * ph), n=T, axis=0)  # impose the spectrum
		ranks = np.argsort(np.argsort(S, axis=0), axis=0)
		S = np.take_along_axis(sorted_x, ranks, axis=0)  # impose the amplitude distribution
	return S[:, 0] if one_d else S


def perm_p(real, null):
	"""One-sided permutation p with the +1 correction: (1 + #{null >= real}) / (1 + N)."""
	null = np.asarray(null)
	return float((1 + np.sum(null >= real)) / (1 + len(null)))


def binomial_group(pvals, alpha=0.05):
	"""Group test: number of subjects with p < alpha vs Binomial(n, alpha)."""
	from scipy.stats import binomtest
	k = int(np.sum(np.asarray(pvals) < alpha))
	return {"k": k, "n": len(pvals), "p": float(binomtest(k, len(pvals), alpha, alternative="greater").pvalue)}
