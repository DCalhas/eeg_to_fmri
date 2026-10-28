"""Additional file 2 (supplementary tables) for the Brain Informatics revision, from data/revision_summary.json,
data/revision/*.json, data/fidelity_summary.json, data/final_models_val_fidelity.json and data/synth_xp1/*_val.json.
Writes submission/additional_file_2.tex."""
import os, json, glob

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(HERE, "submission")
summ = json.load(open(os.path.join(HERE, "data", "revision_summary.json")))
sens = summ["sensitivity"]
C = ["fep1", "fep2", "semmelweis", "pd", "fribourg"]; CN = {"fep1": "FEP1", "fep2": "FEP2", "semmelweis": "Semmelweis", "pd": "PD", "fribourg": "Fribourg"}
M = ["cnn_tag", "e2fnet", "spec2vol", "ntvit"]; MN = {"cnn_tag": "CNN-TAG", "e2fnet": "E2fNet", "spec2vol": "Spec2VolCAMU-Net", "ntvit": "NT-ViT"}
f2 = lambda x: "--" if x is None else f"{x:.2f}"
tex = [r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2cm,landscape]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,xcolor}
\renewcommand{\thetable}{S\arabic{table}}
\newcommand{\verify}[1]{\textcolor{red}{[VERIFY: #1]}}
\title{Additional file 2\\[4pt]\large What Does EEG-to-fMRI Synthesis Add to EEG-Based Diagnosis? A Controlled Benchmark in Psychosis, Schizophrenia and Parkinson's Disease}
\author{\normalsize Supplementary tables}
\date{}
\begin{document}
\maketitle
"""]
# S1 delta, full precision
rows = open(os.path.join(S, "tabS_delta_rows.tex")).read()
tex.append(r"""\footnotesize
\begin{longtable}{l l l l r c c r r r}
\caption{Added value of synthesized fMRI, full precision. $\Delta$AUC of the combined view over the baseline; 95\% and Bonferroni-level (40 comparisons) percentile intervals of a paired stratified bootstrap over subjects (20,000 resamples); two-sided bootstrap $p$; Holm-adjusted $p$ within the 20 trained-synthesizer comparisons (XP1: within the 20 XP1 comparisons) and within the 40 comparisons including untrained synthesizers.}\label{tab:s_delta}\\
\toprule
Baseline & Cohort & Synthesizer & Weights & $\Delta$AUC & 95\% CI & Bonferroni-40 CI & $p$ & Holm (20) & Holm (40) \\
\midrule
\endfirsthead
\toprule
Baseline & Cohort & Synthesizer & Weights & $\Delta$AUC & 95\% CI & Bonferroni-40 CI & $p$ & Holm (20) & Holm (40) \\
\midrule
\endhead
""" + rows + r"""\bottomrule
\end{longtable}
""")
# S2 sensitivity per synthesizer
lines = []
for c in C:
	t = sens[c]
	for m in M:
		for u, un in (("", "trained"), ("_untrained", "untrained")):
			k = m + u
			lines.append(f"{CN[c] if (m == 'cnn_tag' and not u) else ''} & {MN[m] if not u else ''} & {un} & " + " & ".join(f2(t.get(k + x)) for x in ("", "_k5", "_k10", "_k40", "_block", "_cov", "_ica")) +
						 f" & {f2(t.get('xp1_' + m)) if not u else ''} \\\\")
	lines.append(f" & \\multicolumn{{2}}{{l}}{{Band power / information-matched}} & {f2(t['bp_rel'])} / {f2(t['stft_ms'])} & & & & & {f2(t['bp_rel_cov'])} / {f2(t['stft_ms_cov'])} & {f2(t['bp_rel_ica'])} / {f2(t['stft_ms_ica'])} & \\\\")
	lines.append(f" & \\multicolumn{{2}}{{l}}{{Band power without 30--45\\,Hz; CNN-TAG 50 epochs}} & {f2(t['bp_nogamma'])} & & & & & & & {f2(t['cnn_tag_ep50'])} \\\\ \\midrule")
tex.append(r"""\begin{longtable}{l l l c c c c c c c c}
\caption{Sensitivity analyses: AUC of each synthesized view with 20 (primary), 5, 10 or 40 principal components, average pooling in $3\times4\times4$-voxel blocks, covariates regressed out within training folds, ICA-cleaned inputs (ocular components removed), and synthesizers trained on XP1 (last column; for the last row of each cohort, CNN-TAG trained for 50 epochs). Covariates: age and sex (all cohorts with these data), chlorpromazine-equivalent dose (FEP1, FEP2), recording hour (Semmelweis, Fribourg), recording year (Fribourg).}\label{tab:s_sens}\\
\toprule
Cohort & Synthesizer & Weights & $k=20$ & $k=5$ & $k=10$ & $k=40$ & Blocks & Covariates & ICA & XP1 \\
\midrule
\endfirsthead
\toprule
Cohort & Synthesizer & Weights & $k=20$ & $k=5$ & $k=10$ & $k=40$ & Blocks & Covariates & ICA & XP1 \\
\midrule
\endhead
""" + "\n".join(lines) + r"""
\bottomrule
\end{longtable}
""")
# S3 metadata and artefacts
lines = []
for c in C:
	md = summ["metadata"][c]
	for k, v in sorted((md["metadata_tests"] or {}).items()):
		if "auc" in v:
			lines.append(f"{CN[c]} & {k.replace('_', ' ')} & {v['mean_pt']:.2f} & {v['mean_hc']:.2f} & {v['auc']:.2f} & {v['p']:.4f} \\\\")
		else:
			tab = "; ".join(f"{kk}: {vv[1]}/{vv[0]}" for kk, vv in v["table"].items())
			lines.append(f"{CN[c]} & {k.replace('_', ' ')} & \\multicolumn{{2}}{{l}}{{{tab} (pt/hc)}} & & {v['p']:.4f} \\\\")
	for k, v in sorted((md["artefact_tests"] or {}).items()):
		lines.append(f"{CN[c]} & {k.replace('_', ' ')} (artefact) & {v['mean_pt']:.3f} & {v['mean_hc']:.3f} & {v['auc']:.2f} & {v['p']:.4f} \\\\")
	ic = summ["ica_components"].get(c)
	if ic:
		lines.append(f"{CN[c]} & ocular ICA components removed & {ic['ic_eog_pt']:.2f} & {ic['ic_eog_hc']:.2f} & & {ic['p']:.4f} \\\\")
	lines.append("\\midrule")
tex.append(r"""\begin{longtable}{l l r r r r}
\caption{Recording metadata and artefact measures by group: patient and control means, AUC of the measure for the label (Mann--Whitney) and two-sided $p$; categorical fields with Fisher or $\chi^2$ tests. Artefact measures: fraction of pseudo-TRs exceeding 150\,$\mu$V peak-to-peak or an EMG-like 30--45\,Hz power, median peak-to-peak amplitude ($\mu$V), median log 30--45\,Hz power, log RMS of the EOG channels and (PD) of the accelerometer, over the analysed crop.}\label{tab:s_meta}\\
\toprule
Cohort & Measure & Patients & Controls & AUC & $p$ \\
\midrule
\endfirsthead
\toprule
Cohort & Measure & Patients & Controls & AUC & $p$ \\
\midrule
\endhead
""" + "\n".join(lines[:-1]) + r"""
\bottomrule
\end{longtable}
""")
# S4 fidelity of the final models and of XP1
val = json.load(open(os.path.join(HERE, "data", "final_models_val_fidelity.json")))
lines = []
for sub, r in val.items():
	for m in ["template"] + M:
		x = r[m]
		lines.append(f"NODDI & {sub} & {'EEG-free template' if m == 'template' else MN[m]} & {x['ssim_2d_e2fnet']:.3f} & {x['rmse']:.4f} & {x['voxel_r_mean']:+.4f} \\\\")
for f in sorted(glob.glob(os.path.join(HERE, "data", "synth_xp1", "*_val.json"))):
	d = json.load(open(f)); m = os.path.basename(f)[4:-9]
	if m == "cnn_tag":
		t = d["val"]["template"]; lines.append(f"XP1 & sub-xp109, sub-xp110 & EEG-free template & {t['ssim_2d_e2fnet']:.3f} & {t['rmse']:.4f} & -- \\\\")
	x = d["val"]["model"]
	lines.append(f"XP1 & sub-xp109, sub-xp110 & {MN[m]} (epoch {d.get('best_epoch')}) & {x['ssim_2d_e2fnet']:.3f} & {x['rmse']:.4f} & {x['voxel_r_mean']:+.4f} \\\\")
fid = json.load(open(os.path.join(HERE, "data", "fidelity_summary.json")))
lines.append("\\midrule")
for k, v in fid.items():
	arm, m = k.split("/")
	lines.append(f"NODDI LOSO ({arm.replace('_', ' ')}) & 16 subjects & {MN.get(m, 'EEG-free template' if m == 'template' else 'Ridge (spectra)')} & {v['ssim'][0]:.3f} & {v['rmse'][0]:.4f}$\\pm${v['rmse'][1]:.4f} & {('%+.4f' % v['r'][0]) if 'r' in v else '--'} \\\\")
tex.append(r"""\begin{longtable}{l l l r r r}
\caption{Fidelity of the final synthesizers used for the clinical cohorts on the held-out validation subjects (NODDI 49 and 50; XP1 sub-xp109 and sub-xp110, pooled), and RMSE of the leave-one-subject-out NODDI evaluation (fidelity table of the main text). $r$: mean voxelwise temporal correlation.}\label{tab:s_fid}\\
\toprule
Data & Held-out subjects & Model & SSIM & RMSE & $r$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{longtable}
\end{document}
""")
open(os.path.join(S, "additional_file_2.tex"), "w").write("\n".join(tex))
print("written")
