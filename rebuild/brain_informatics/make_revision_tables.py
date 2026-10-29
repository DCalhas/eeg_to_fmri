"""Results tables for the Brain Informatics revision, from runs/revision/*.json (copied to data/revision).
Writes LaTeX table bodies to submission/tab_*.tex and the numbers quoted in the text to data/revision_summary.json.
Holm adjustment within families: trained-synthesizer added value over band power (20 comparisons) and the 40
comparisons including untrained synthesizers; the same for added value over the information-matched EEG view."""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "revision")
S = os.path.join(HERE, "submission")
C = ["fep1", "fep2", "semmelweis", "pd", "fribourg"]
CN = {"fep1": "FEP1", "fep2": "FEP2", "semmelweis": "Semmelweis", "pd": "PD", "fribourg": "Fribourg"}
M = ["cnn_tag", "e2fnet", "spec2vol", "ntvit"]
MN = {"cnn_tag": "CNN-TAG", "e2fnet": "E2fNet", "spec2vol": "Spec2VolCAMU-Net", "ntvit": "NT-ViT"}
load = lambda f: json.load(open(os.path.join(D, f))) if os.path.exists(os.path.join(D, f)) else None
V = {c: load(f"{c}.json")["views"] for c in C}
X = {c: (load(f"{c}_extraA.json") or {"views": {}})["views"] for c in C}
ICA = {c: (load(f"{c}_ica.json") or {"views": {}})["views"] for c in C}
PDR = {k: (load(f"pd_{k}.json") or {"views": {}})["views"] for k in ("pdrest", "pdrest_ica")}


def holm(p):
	p = np.asarray(p, float); o = np.argsort(p); m = len(p)
	adj = np.empty(m); run = 0.0
	for i, j in enumerate(o):
		run = max(run, min(1.0, (m - i) * p[j])); adj[j] = run
	return adj


def fmt_auc(r, p=True):
	s = f"${r['auc']:.2f}$"
	if p and r.get("perm_p_greater") is not None:
		pv = r["perm_p_greater"]
		s += f" (${pv:.3f}$)" if pv < 0.01 else f" (${pv:.2f}$)"
	return s


summary = {}
# ---------------------------------------------------------------- Table: single views
rows = []
for c in C:
	v = V[c]
	cells = [fmt_auc(v["bp_rel"]), fmt_auc(v["bp_abs"]), fmt_auc(v["stft_ms"]), fmt_auc(v["stft_rp"], False)]
	rows.append(f"        {CN[c]} & " + " & ".join(cells) + " \\\\")
open(os.path.join(S, "tab_eeg_rows.tex"), "w").write("\n".join(rows) + "\n")
rows = []
for c in C:
	v = V[c]
	for u, uname in (("", "trained"), ("_untrained", "untrained")):
		cells = [fmt_auc(v["synth_" + m + u]) for m in M]
		rows.append(f"        {CN[c] if not u else ''} & {uname} & " + " & ".join(cells) + " \\\\")
open(os.path.join(S, "tab_synth_rows.tex"), "w").write("\n".join(rows) + "\n")

# ---------------------------------------------------------------- Table: added value with Holm
fam = {}
for base in ("bp_rel", "stft_ms"):
	for kind, src_tag in (("trained", ""), ("untrained", "_untrained"), ("xp1", "xp1")):
		for c in C:
			for m in M:
				tag = f"xp1_{m}" if kind == "xp1" else f"{m}{src_tag}"
				src = X[c] if kind == "xp1" else V[c]
				r = src.get(f"{base}+synth_{tag}")
				if r and "delta" in r:
					fam[(base, kind, c, m)] = r["delta"]
for base in ("bp_rel", "stft_ms"):
	keys20 = [k for k in fam if k[0] == base and k[1] == "trained"]
	keys40 = [k for k in fam if k[0] == base and k[1] in ("trained", "untrained")]
	keysx = [k for k in fam if k[0] == base and k[1] == "xp1"]
	for keys, name in ((keys20, "holm20"), (keys40, "holm40"), (keysx, "holm_xp1")):
		adj = holm([fam[k]["p_boot"] for k in keys])
		for k, a in zip(keys, adj):
			fam[k][name] = float(a)
rows = []
for c in C:
	for i, m in enumerate(M):
		cells = []
		for base in ("bp_rel", "stft_ms"):
			for kind in ("trained", "untrained"):
				d = fam.get((base, kind, c, m))
				if d is None:
					cells.append("--"); continue
				mark = "$^{\\dagger}$" if d["holm40"] < 0.05 else ("$^{*}$" if (d["ci95"][0] > 0 or d["ci95"][1] < 0) else "")
				cells.append(f"${d['delta']:+.3f}${mark}")
		rows.append(f"        {CN[c] if i == 0 else ''} & {MN[m]} & " + " & ".join(cells) + " \\\\")
	rows.append("        \\hline")
open(os.path.join(S, "tab_delta_rows.tex"), "w").write("\n".join(rows[:-1]) + "\n")
summary["delta"] = {"|".join(k): v for k, v in fam.items()}
rows = []
KN = {"trained": "trained (NODDI)", "untrained": "untrained", "xp1": "trained (XP1)"}
for base, bname in (("bp_rel", "band power"), ("stft_ms", "information-matched EEG")):
	for c in C:
		for m in M:
			for kind in ("trained", "untrained", "xp1"):
				d = fam.get((base, kind, c, m))
				if d is None:
					continue
				h = d.get("holm20") if kind == "trained" else (d.get("holm_xp1") if kind == "xp1" else None)
				rows.append(f"        {bname} & {CN[c]} & {MN[m]} & {KN[kind]} & ${d['delta']:+.3f}$ & [${d['ci95'][0]:+.3f}$, ${d['ci95'][1]:+.3f}$] & "
							f"[${d['ci_bonf40'][0]:+.3f}$, ${d['ci_bonf40'][1]:+.3f}$] & ${d['p_boot']:.4f}$ & {'--' if h is None else f'${h:.3f}$'} & "
							f"{'--' if kind == 'xp1' else '$%.3f$' % d['holm40']} \\\\")
open(os.path.join(S, "tabS_delta_rows.tex"), "w").write("\n".join(rows) + "\n")
summary["n_holm40_sig"] = {base: [("|".join(k), fam[k]["delta"]) for k in fam if k[0] == base and k[1] in ("trained", "untrained") and fam[k]["holm40"] < 0.05] for base in ("bp_rel", "stft_ms")}
summary["n_holm20_sig"] = {base: [("|".join(k), fam[k]["delta"]) for k in fam if k[0] == base and k[1] == "trained" and fam[k]["holm20"] < 0.05] for base in ("bp_rel", "stft_ms")}

# ---------------------------------------------------------------- Table: sensitivity (AUC of the best EEG view and of each synthesized view)
def g(src, k):
	r = src.get(k); return None if r is None else r["auc"]
sens = {}
for c in C:
	s = {"bp_rel": g(V[c], "bp_rel"), "stft_ms": g(V[c], "stft_ms"), "bp_nogamma": g(V[c], "bp_nogamma"),
		 "bp_rel_cov": g(V[c], "bp_rel_cov"), "stft_ms_cov": g(V[c], "stft_ms_cov"),
		 "bp_rel_ica": g(ICA[c], "bp_rel"), "stft_ms_ica": g(ICA[c], "stft_ms"), "cnn_tag_ep50": g(X[c], "synth_cnn_tag_ep50")}
	for m in M:
		for u in ("", "_untrained"):
			t = m + u
			s[t] = g(V[c], f"synth_{t}"); s[t + "_cov"] = g(V[c], f"synth_{t}_cov"); s[t + "_ica"] = g(ICA[c], f"synth_{t}_ica")
			for k in (5, 10, 40):
				s[f"{t}_k{k}"] = g(V[c], f"synth_{t}_k{k}")
			s[t + "_block"] = g(V[c], f"synth_{t}_block")
		s["xp1_" + m] = g(X[c], f"synth_xp1_{m}")
	sens[c] = s
summary["sensitivity"] = sens
for k, src in PDR.items():
	summary[f"pd_{k}"] = {name: r["auc"] for name, r in src.items()}

# ---------------------------------------------------------------- metadata, artefacts, PD timing
meta = {}
for c in C:
	d = load(f"{c}.json")
	meta[c] = {"metadata_tests": d.get("metadata_tests"), "artefact_tests": d.get("artefact_tests"), "covariates": d.get("covariates")}
summary["metadata"] = meta
logs = {}
for c in C:
	lg = load(f"logs_ica/{c}_log.json")
	if lg:
		subs = lg["subjects"]
		y = np.array([s["label"] for s in subs]); n = np.array([s.get("n_ic_eog", np.nan) for s in subs], float)
		from scipy import stats
		logs[c] = {"eog_channels": lg["eog_channels"], "ic_eog_pt": float(np.nanmean(n[y == 1])), "ic_eog_hc": float(np.nanmean(n[y == 0])),
				   "p": float(stats.mannwhitneyu(n[y == 1], n[y == 0]).pvalue)}
summary["ica_components"] = logs
json.dump(summary, open(os.path.join(HERE, "data", "revision_summary.json"), "w"), indent=1, default=float)
print(json.dumps({"holm40_sig": summary["n_holm40_sig"], "holm20_sig": summary["n_holm20_sig"], "ica": logs}, indent=1, default=float))

# ---------------------------------------------------------------- compact main-text table: all views, AUC (* permutation p < 0.05)
def a(r):
	return f"${r['auc']:.2f}" + ("^{*}" if (r.get("perm_p_greater") is not None and r["perm_p_greater"] < 0.05) else "") + "$"
rows = []
for c in C:
	v = V[c]
	cells = [a(v["bp_rel"]), a(v["stft_ms"])] + [f"{a(v['synth_' + m])} / {a(v['synth_' + m + '_untrained'])}" for m in M]
	rows.append(f"        {CN[c]} & " + " & ".join(cells) + " \\\\")
open(os.path.join(S, "tab_compact_rows.tex"), "w").write("\n".join(rows) + "\n")
