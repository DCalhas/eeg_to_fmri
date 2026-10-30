"""GPU checks (seconds) for ridge_roi.py: SVD ridge vs closed form, IAAFT properties, perm_p. Run in the smoke job."""
import sys
import types
import numpy as np

sys.argv = ["ridge_roi.py", "-out", "/dev/null", "-gpu_mem", "2000"]
src = open("ridge_roi.py").read()
ns = {"__name__": "ridge_roi_test"}
exec(src.split("#---------------------------------------------------------------- main")[0], ns)  #definitions only
tf = ns["tf"]; Ridge, iaaft, perm_p = ns["Ridge"], ns["iaaft"], ns["perm_p"]
rng = np.random.RandomState(0); ok = True

#ridge vs closed form on standardized X and centred Y
X = rng.normal(size=(80, 30)).astype(np.float32); Xt = rng.normal(size=(20, 30)).astype(np.float32)
Y = (X[:, :3]@rng.normal(size=(3, 2))+0.5*rng.normal(size=(80, 2))).astype(np.float32)
mx, sx = X.mean(0), X.std(0)+1e-6; Xs, Xts = (X-mx)/sx, (Xt-mx)/sx
for a in [0.1, 10., 1000.]:
	B = np.linalg.solve(Xs.T@Xs+a*np.eye(30), Xs.T@(Y-Y.mean(0)))
	ref = Xts@B+Y.mean(0)
	got = Ridge(tf.constant(X), tf.constant(Xt)).predict(tf.constant(Y), [a])[0].numpy()
	if(not np.allclose(got, ref, atol=2e-3, rtol=2e-3)):
		ok = False; print("MISMATCH ridge alpha", a, np.abs(got-ref).max())
print("ridge checked")

#IAAFT: exact amplitude distribution, close power spectrum, relation to a signal destroyed
T = 300; t = np.arange(T); sig = np.sin(2*np.pi*t/40)
x = np.stack([sig+0.3*rng.normal(size=T), np.cumsum(rng.normal(size=T))/5], 1).astype(np.float32)
S = iaaft(tf.constant(x), 50, 1).numpy()
if(not np.allclose(np.sort(S, 1), np.sort(x, 0)[None], atol=1e-5)):
	ok = False; print("MISMATCH iaaft amplitude distribution")
psd = lambda v: np.abs(np.fft.rfft(v, axis=0))**2
rel = np.abs(psd(S[0])-psd(x)).sum(0)/psd(x).sum(0)
if(not np.all(rel < 0.1)):
	ok = False; print("MISMATCH iaaft spectrum relative error", rel)
corr = [np.corrcoef(S[i][:, 0], sig)[0, 1] for i in range(50)]
if(not abs(np.mean(corr)) < 0.15):
	ok = False; print("MISMATCH iaaft keeps the phase of the signal, mean corr", np.mean(corr))
print("iaaft checked, spectrum rel err", np.round(rel, 4), "mean |corr| with signal", round(float(np.mean(np.abs(corr))), 3))

if(not (perm_p(1., [0.]*99) == 0.01 and perm_p(0., [1.]*9) == 1.0)):
	ok = False; print("MISMATCH perm_p")
print("RIDGE_TESTS_OK" if ok else "RIDGE_TESTS_FAILED")
sys.exit(0 if ok else 1)
