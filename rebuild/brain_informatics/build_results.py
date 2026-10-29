"""Assemble the revised Results section of submission/manuscript.tex from the generated table rows and the
revision summary (make_revision_tables.py, make_fidelity_table.py). Replaces everything from \\section{Results}
up to \\section{Discussion}; the contrastive table and its paragraph are carried over verbatim."""
import os, re, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(HERE, "submission")
p = os.path.join(S, "manuscript.tex")
tex = open(p).read()
rows = lambda f: open(os.path.join(S, f)).read().rstrip("\n")
summ = json.load(open(os.path.join(HERE, "data", "revision_summary.json")))
sens = summ["sensitivity"]
C = ["fep1", "fep2", "semmelweis", "pd", "fribourg"]; CN = {"fep1": "FEP1", "fep2": "FEP2", "semmelweis": "Semmelweis", "pd": "PD", "fribourg": "Fribourg"}
M = ["cnn_tag", "e2fnet", "spec2vol", "ntvit"]

i = tex.index(r"\section{Results}"); j = tex.index(r"\section{Discussion}")
old = tex[i:j]
KEEP = os.path.join(HERE, "data", "results_carryover.json")  # blocks carried over from the pre-revision Results (first run)
if os.path.exists(KEEP):
	kept = json.load(open(KEEP))
else:
	kept = {"par": re.search(r"\\noindent\\textbf\{Contrastive sinusoid separation\.\}.*?\n\n", old, flags=re.S).group(0),
			"tab": re.search(r"\\begin\{table\}\[!htbp\]\s*\\centering\s*\\caption\{Contrastive sinusoid separation.*?\\end\{table\}", old, flags=re.S).group(0),
			"fig": re.search(r"\\begin\{figure\}\[!htbp\].*?\\label\{fig:results\}\s*\\end\{figure\}", old, flags=re.S).group(0)}
	json.dump(kept, open(KEEP, "w"), indent=1)
contrastive_par, contrastive_tab, fig_results = kept["par"], kept["tab"], kept["fig"]
fig_results = fig_results.replace("Mean AUC over 20 repeats of stratified 10-fold cross-validation with 95\\% bootstrap intervals, for EEG band power (squares) and",
								  "Mean AUC over 20 repeats of stratified 10-fold cross-validation with 95\\% bootstrap intervals, for EEG band power (squares), the information-matched EEG view (open diamonds) and")

def rng(vals):
	v = [x for x in vals if x is not None]
	return f"{min(v):.2f}--{max(v):.2f}"
mx = lambda c, keys: max(sens[c][k] for k in keys if sens[c].get(k) is not None)
def cell(a, b):
	return (f"${a:.2f}$" if a is not None else "--") + " & " + (f"${b:.2f}$" if b is not None else "--")
sens_rows = []
for name, fn in [
	("Primary (NODDI)", lambda c: (mx(c, M), sens[c]["stft_ms"])),
	("Untrained", lambda c: (mx(c, [m + "_untrained" for m in M]), sens[c]["stft_ms"])),
	("Trained on XP1", lambda c: (mx(c, ["xp1_" + m for m in M]), sens[c]["stft_ms"])),
	("CNN-TAG 50 epochs", lambda c: (sens[c]["cnn_tag_ep50"], sens[c]["stft_ms"])),
	("5--40 comp., pooling", lambda c: (mx(c, [m + x for m in M for x in ("_k5", "_k10", "_k40", "_block")]), sens[c]["stft_ms"])),
	("Covariates removed", lambda c: (mx(c, [m + "_cov" for m in M]), sens[c]["stft_ms_cov"])),
	("ICA (ocular)", lambda c: ((mx(c, [m + "_ica" for m in M]) if any(sens[c].get(m + "_ica") for m in M) else None), sens[c]["stft_ms_ica"])),
]:
	sens_rows.append(f"        {name} & " + " & ".join(cell(*fn(c)) for c in C) + " \\\\")
sens_rows = "\n".join(sens_rows)

d = lambda base, kind, c, m: summ["delta"][f"{base}|{kind}|{c}|{m}"]
fmtd = lambda x: f"${x['delta']:+.3f}$ [${x['ci95'][0]:+.3f}$, ${x['ci95'][1]:+.3f}$]"
ica = summ["ica_components"]
meta = summ["metadata"]

RES = r"""\section{Results}\label{section:results}

\subsection{EEG-only views}\label{section:res_eeg}

Relative band power separated Parkinson's disease from controls (AUC $0.786$, 95\% CI $0.63$--$0.92$, $p=0.002$) but not the psychosis or schizophrenia cohorts (Table~\ref{tab:eeg}): FEP1 $0.620$ ($p=0.07$), FEP2 $0.646$ ($p=0.050$), Semmelweis $0.571$ ($p=0.19$), Fribourg $0.539$ ($p=0.36$). Pooling the two first-episode samples ($n=138$) lowered the AUC to $0.568$ ($p=0.14$): the spectral differences that separate patients within one sample do not carry over to the other, consistent with sample heterogeneity (the two samples were also recorded mostly at different sampling rates, 1 and 3\,kHz). The information-matched view, the temporal mean and standard deviation of the synthesizers' own input spectra, was the best single view in Parkinson's disease ($0.825$, $p=0.001$), and an untrained sparse random projection of the full synthesizer input did equally well ($0.823$). In the other cohorts neither exceeded band power (FEP1 $0.52$, FEP2 $0.57$, Semmelweis $0.54$, Fribourg $0.56$).

\begin{table}[!htbp]
    \centering
    \caption{AUC of the EEG-only views (mean over 20 repeats of stratified 10-fold cross-validation; one-sided permutation $p$ in parentheses, 1,000 permutations; the random projection was not permutation-tested).}
    \label{tab:eeg}
    \footnotesize\setlength{\tabcolsep}{4pt}
    \begin{tabular}{@{}l c c c c@{}}
        \hline
        & \multicolumn{2}{c}{Band power} & Information- & Random \\
        Cohort & relative & absolute & matched & projection \\
        \hline
""" + rows("tab_eeg_rows.tex") + r"""
        \hline
    \end{tabular}
\end{table}

\subsection{Synthesized views}\label{section:res_clf}

In Parkinson's disease three of the four trained synthesizers produced views that discriminated patients from controls (E2fNet $0.744$, $p=0.010$; Spec2VolCAMU-Net $0.694$, $p=0.040$; NT-ViT $0.672$, $p=0.024$; CNN-TAG $0.483$, $p=0.46$; Table~\ref{tab:clf}, Fig.~\ref{fig:results}), but none reached band power or the information-matched view, and the same architectures with random weights did as well ($0.69$--$0.74$, all $p\leq0.026$). In the psychosis and schizophrenia cohorts no trained synthesizer was significant; among the untrained ones, Spec2VolCAMU-Net in Semmelweis ($0.740$, $p=0.006$) and NT-ViT in FEP2 ($0.626$, $p=0.048$) were. CNN-TAG in FEP2 fell well below chance ($0.329$; Section~\ref{section:res_checks}).

""" + fig_results + r"""

\begin{table}[!htbp]
    \centering
    \caption{AUC of the synthesized-fMRI views of trained and untrained (random-weight) synthesizers (one-sided permutation $p$ in parentheses, 500 permutations).}
    \label{tab:clf}
    \footnotesize\setlength{\tabcolsep}{3pt}
    \begin{tabular}{@{}l l c c c c@{}}
        \hline
        Cohort & Weights & CNN-TAG & E2fNet & Spec2VolCAMU-Net & NT-ViT \\
        \hline
""" + rows("tab_synth_rows.tex") + r"""
        \hline
    \end{tabular}
\end{table}

\subsection{Added value of synthesized fMRI}\label{section:res_delta}

Added to band power, no trained synthesizer gave a gain that survived Holm correction across the 20 comparisons (Table~\ref{tab:delta}). Nominal gains, with 95\% intervals excluding zero, occurred for Spec2VolCAMU-Net ($+0.035$) and NT-ViT ($+0.038$) in Parkinson's disease, E2fNet ($+0.067$) and Spec2VolCAMU-Net ($+0.112$) in Semmelweis, and E2fNet in Fribourg ($+0.075$), but adding the information-matched EEG view to band power gave a gain of the same size where the EEG carried signal (Parkinson's disease, $+0.030$, 95\% CI $+0.008$ to $+0.061$). Added to the information-matched view, no trained synthesizer improved classification in any cohort; the only difference that survived correction among trained synthesizers was a loss, for CNN-TAG in FEP2 ($-0.226$). Including the untrained synthesizers (40 comparisons per baseline), one gain survived Holm correction against both baselines: the untrained Spec2VolCAMU-Net in Semmelweis, """ + fmtd(d("bp_rel", "untrained", "semmelweis", "spec2vol")) + r""" over band power and """ + fmtd(d("stft_ms", "untrained", "semmelweis", "spec2vol")) + r""" over the information-matched view; the same network trained on NODDI added """ + f"${d('stft_ms', 'trained', 'semmelweis', 'spec2vol')['delta']:+.3f}$" + r""". Full-precision intervals, bootstrap $p$-values and Bonferroni-level intervals for all comparisons are given in Additional file~2.

\begin{table}[!htbp]
    \centering
    \caption{Added value of synthesized fMRI: $\Delta$AUC of the combined view over band power and over the information-matched EEG view, $^{*}$95\% paired bootstrap interval excludes zero; $^{\dagger}$Holm-adjusted $p<0.05$ across the 40 comparisons of trained and untrained synthesizers for that baseline. Intervals, bootstrap $p$-values and Bonferroni-level intervals are in Additional file~2.}
    \label{tab:delta}
    \scriptsize\setlength{\tabcolsep}{3pt}
    \begin{tabular}{@{}l l c c c c@{}}
        \hline
        & & \multicolumn{2}{c}{Over band power} & \multicolumn{2}{c}{Over information-matched EEG} \\
        Cohort & Synthesizer & Trained & Untrained & Trained & Untrained \\
        \hline
""" + rows("tab_delta_rows.tex") + r"""
        \hline
    \end{tabular}
\end{table}

\subsection{Sensitivity analyses}\label{section:res_sens}

Table~\ref{tab:sens} compares, for each analysis, the best synthesized view (the maximum over the four synthesizers, which favours synthesis) with the information-matched EEG view. \textbf{Second training dataset.} Synthesizers retrained on XP1 reached AUC """ + rng([sens[c]["xp1_" + m] for c in C for m in M]) + r""" across cohorts and synthesizers; none improved on the information-matched view (Additional file~2), and only Spec2VolCAMU-Net in Semmelweis ($""" + f"{sens['semmelweis']['xp1_spec2vol']:.2f}" + r"""$) and in Parkinson's disease ($""" + f"{sens['pd']['xp1_spec2vol']:.2f}" + r"""$) were nominally significant. \textbf{Training length.} CNN-TAG trained for 50 epochs without epoch selection (validation SSIM $0.636$ against $0.686$ after the first epoch) reached $""" + f"{sens['pd']['cnn_tag_ep50']:.2f}" + r"""$ in Parkinson's disease and $""" + f"{sens['semmelweis']['cnn_tag_ep50']:.2f}" + r"""$ in Semmelweis. \textbf{Dimensionality.} With 5, 10 or 40 components or block pooling, single synthesized views varied by up to $0.4$ AUC within a cohort (e.g.\ untrained E2fNet in Parkinson's disease, $0.34$ with 5 components and $0.77$ with 40), which shows how unstable they are at these sample sizes; no setting brought a trained synthesizer above the information-matched view. \textbf{Covariates.} Regressing out age and sex (and medication dose, recording hour or year where available) within training folds left Parkinson's disease unchanged (band power $""" + f"{sens['pd']['bp_rel_cov']:.2f}" + r"""$, information-matched $""" + f"{sens['pd']['stft_ms_cov']:.2f}" + r"""$), lowered FEP1 band power to $""" + f"{sens['fep1']['bp_rel_cov']:.2f}" + r"""$, and lowered the untrained Spec2VolCAMU-Net in Semmelweis from $0.74$ to $""" + f"{sens['semmelweis']['spec2vol_untrained_cov']:.2f}" + r"""$, so part of that effect follows age, sex or recording hour. \textbf{Artefacts.} Removing ocular components by ICA left Parkinson's disease unchanged (band power $0.81$, information-matched $0.80$) but lowered band power in FEP1 (from $0.62$ to $0.54$) and FEP2 (from $0.65$ to $0.58$), so part of the weak EEG signal of the first-episode cohorts was ocular. It also abolished the untrained Spec2VolCAMU-Net effect in Semmelweis ($0.57$, $p=0.13$); after ICA, no synthesized view gave a gain that survived Holm correction in any cohort (smallest adjusted $p=0.42$ over band power and $0.08$ over the information-matched view, 40 comparisons each); the only difference that survived among the trained synthesizers was a loss (Spec2VolCAMU-Net in FEP2, $-0.156$ relative to the information-matched view). Excluding the 30--45\,Hz band raised FEP1 band power to $""" + f"{sens['fep1']['bp_nogamma']:.2f}" + r"""$ and left the other cohorts within $0.03$. In Parkinson's disease, restricting the analysis to the marked eyes-closed and eyes-open spans kept band power at $0.81$ ($0.82$ after ICA), so its discrimination is not carried by the set-up period, the oddball task or ocular artefacts; the information-matched view, whose temporal standard deviation rests on about 45 pseudo-TRs in this shorter crop, fell to $0.68$ ($0.75$ after ICA). In this crop the one-epoch CNN-TAG reached $0.81$ ($p=0.010$), above its untrained version ($0.57$), the only analysis in which a trained synthesizer clearly outperformed its random-weight control; it added nothing to band power ($+0.013$, 95\% CI $-0.018$ to $+0.045$), fell to $0.73$ after ICA, and its gain over the information-matched view ($+0.130$, 95\% CI $+0.003$ to $+0.270$, uncorrected $p=0.047$) did not survive Holm correction across the 16 comparisons of this analysis (adjusted $p\geq0.19$).

\begin{table}[!htbp]
    \centering
    \caption{Sensitivity analyses: AUC of the best synthesized view (syn., maximum over the four synthesizers, which favours synthesis) and of the information-matched EEG view (EEG) under the same analysis.}
    \label{tab:sens}
    \scriptsize\setlength{\tabcolsep}{2.5pt}
    \begin{tabular}{@{}l cc cc cc cc cc@{}}
        \hline
        & \multicolumn{2}{c}{FEP1} & \multicolumn{2}{c}{FEP2} & \multicolumn{2}{c}{Semmelweis} & \multicolumn{2}{c}{PD} & \multicolumn{2}{c}{Fribourg} \\
        Analysis & syn. & EEG & syn. & EEG & syn. & EEG & syn. & EEG & syn. & EEG \\
        \hline
""" + sens_rows + r"""
        \hline
    \end{tabular}
\end{table}

\subsection{Recording conditions, artefacts and a below-chance estimate}\label{section:res_checks}

No recording metadata differed between groups in FEP1 or Parkinson's disease; in FEP2 the raw recordings of patients were longer (removed by the fixed crop). Semmelweis patients were recorded earlier in the day than controls (mean hour $""" + f"{meta['semmelweis']['metadata_tests']['hour']['mean_pt']:.1f}" + r"""$ against $""" + f"{meta['semmelweis']['metadata_tests']['hour']['mean_hc']:.1f}" + r"""$, $p<0.001$), and in Fribourg the nine participants recorded in 2016 were all patients and the auxiliary channel layout differed by group (Additional file~2). Artefact burden differed by group in two cohorts: in Parkinson's disease controls had larger EEG amplitudes and more ocular activity than patients (EOG power, AUC $""" + f"{meta['pd']['artefact_tests']['eog_log_rms']['auc']:.2f}" + r"""$, $p=""" + f"{meta['pd']['artefact_tests']['eog_log_rms']['p']:.3f}" + r"""$), consistent with the reduced blink rate of Parkinson's disease, whereas accelerometer power did not differ ($p=""" + f"{meta['pd']['artefact_tests']['acc_log_rms']['p']:.2f}" + r"""$); in Semmelweis patients had more high-amplitude segments ($p=""" + f"{meta['semmelweis']['artefact_tests']['frac_rejected']['p']:.3f}" + r"""$). The number of ocular components removed by ICA did not differ between groups in any cohort (all $p\geq""" + f"{min(v['p'] for v in ica.values()):.2f}" + r"""$). In Parkinson's disease every recording followed the same order (one minute of eyes closed, then one minute of eyes open, then the oddball task), and the onsets of the rest spans and of the task did not differ between groups ($p=0.54$ and $p=0.31$); the fixed 0--400\,s crop of the primary analysis therefore included a variable set-up period before rest and the start of the oddball task.

The below-chance AUC of CNN-TAG in FEP2 is not explained by a nuisance factor: none of its first 20 principal components was associated with sampling rate, recording duration, age, sex or medication dose ($|\rho|\leq0.20$), and no subject carried more than $26\%$ of any of the first five. In every outer fold the inner cross-validation AUC was below $0.5$ ($0.35$--$0.44$) while the training AUC was $0.71$--$0.83$, the signature of anti-learning in features without signal: when the full-sample difference between groups is close to zero, the difference in a training fold is opposite to that in its held-out fold, so a fitted direction reverses out of sample \cite{varoquaux2018}.

""" + contrastive_par.replace(r"\noindent\textbf{Contrastive sinusoid separation.}", r"\subsection{Contrastive sinusoid separation}\label{section:res_contrastive}" + "\n\n") + contrastive_tab + r"""

\subsection{What the synthesizers learn}\label{section:res_fidelity}

On NODDI, leave-one-subject-out over the 16 subjects (Table~\ref{tab:fidelity}), the EEG-free template (the mean training volume) reached SSIM $0.675$ and PSNR $21.2$; under E2fNet's own data pipeline, which pairs EEG and fMRI from the start of the file, the same template reached $0.676$ and $21.2$. Both are above the published SSIM of E2fNet ($0.605$) and NT-ViT ($0.581$) and slightly below that of Spec2VolCAMU-Net ($0.693$); the published values come from each study's own data split and preprocessing and are therefore indicative only. Retrained with trigger-aligned EEG, ridge regression, CNN-TAG and E2fNet were below the template on SSIM ($0.631$--$0.655$, Wilcoxon $p\leq0.003$ over subjects) and all four synthesizers were worse on RMSE ($p\leq0.0002$); Spec2VolCAMU-Net tied it on SSIM ($0.679$, higher in 11 of 16 subjects, $p=0.30$). Pairing EEG from the start of the file or replacing EEG by the ECG left these values almost unchanged, and no model showed a voxelwise temporal correlation above that obtained with circularly shifted EEG. The final synthesizers used for the clinical cohorts behaved the same on the two held-out NODDI subjects (SSIM within $0.03$ of the template, voxelwise $r\approx0$; Additional file~2), as did the synthesizers retrained on XP1 (SSIM $0.662$--$0.699$ against a template of $0.684$ on two held-out XP1 subjects). Figure~\ref{fig:examples} shows why: Spec2VolCAMU-Net and NT-ViT produce an almost constant, template-like volume (voxelwise temporal SD close to zero), E2fNet varies mainly at the brain edge, and the one-epoch CNN-TAG fluctuates more than the real data without following it. Between subjects, a synthesizer's output therefore differs mainly through the subject's overall EEG spectrum. With a target that removes anatomy and motion (Section~\ref{setting:fidelity}), EEG predicted a statistically detectable but negligible part of BOLD in held-out subjects: ridge regression from full spectra $r=0.026$, UnEBOLT $r=0.035$ (above shift and swap nulls, $p=0.02$ and $p=0.03$), with $R^2\approx0$ for both; within subjects, $r\approx0$.

\begin{table*}[!htbp]
    \centering
    \caption{Fidelity on NODDI, leave-one-subject-out (16 test subjects; mean $\pm$ SD over subjects, seeds averaged within a subject). PSNR from the RMSE of $[0,1]$-scaled volumes (RMSE in Additional file~2). $r$: mean voxelwise temporal correlation with the real volumes; shift null: the same with circularly shifted EEG. Epoch: median selected epoch.}
    \label{tab:fidelity}
    \scriptsize\setlength{\tabcolsep}{2.5pt}
    \begin{tabular}{@{}l l c c c c c@{}}
        \hline
        Input & Model & SSIM & PSNR & $r$ & $r$ (shift null) & Epoch \\
        \hline
""" + rows("table_fidelity_rows.tex") + r"""
        \hline
    \end{tabular}
\end{table*}

\begin{figure*}[!htbp]
    \centering
    \includegraphics[width=\textwidth]{fig5.pdf}
    \caption{\textbf{Synthesized volumes of a held-out subject.} NODDI subject 49 (held out from training), final synthesizers used for the clinical cohorts. \textbf{A} One volume (axial slice 15 of 30) of the real data, the EEG-free template and each synthesizer. \textbf{B} Voxelwise temporal standard deviation over the 287 volumes (same scale for all). \textbf{C} Brain-mean intensity over time, demeaned, on the same intensity scale.}
    \label{fig:examples}
\end{figure*}

"""
open(os.path.join(HERE, "data", "results_draft.tex"), "w").write(RES)
tex2 = tex[:i] + RES + tex[j:]
open(p, "w").write(tex2)
print("draft written:", len(RES), "chars; old Results", len(old))
