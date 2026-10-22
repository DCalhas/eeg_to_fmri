# Rebuild plan: synthesized fMRI for EEG-only diagnosis (Q1 target)

Branch `rebuild-2026`. Frida: `~/eeg_to_fmri/{repo,venv,envs/tf29,data,runs,logs}`; `uncertainty/` belongs to the parallel uncertainty-paper session.

## Claim we can defend

The prior-work scan from 2026-10-22 shows we are not the first to use synthesized fMRI for diagnosis:
- Our medRxiv 2023 preprint (10.1101/2023.08.07.23293748) did it first for schizophrenia.
- CATD (Yao et al., IEEE TMI 2025) then did it for Parkinson's: EEG + synthetic BOLD on 8 subjects of ds003490, 5-fold CV, no code.

New claim: **the first benchmark that runs several open state-of-the-art EEG-to-fMRI synthesizers through one diagnostic pipeline, validated at subject level on adequately sized EEG-only clinical cohorts, with controls that test whether the synthesis (rather than the EEG encoder) carries the diagnostic signal.** Our sinusoid-separation fine-tuning is one method within it.

Research questions:
- RQ1 Fidelity: how well does each synthesizer reproduce held-out fMRI? NODDI, Oddball and CN-EPFL with LOSO; SSIM, PSNR, RMSE, voxel/ROI r against mean-volume and subject-mean baselines.
- RQ2 Transfer: how much does fidelity drop rest→task and across montage/sampling rate? Train on NODDI (rest), test on Oddball, XP1 and XP2 (task).
- RQ3 Diagnosis: does the synthesized view carry diagnostic information, and does it add anything beyond EEG? Fribourg SZ (ds004000, n=43); PD (ds003490, n≈50, the same cohort CATD used, but the full n).
- RQ4 Spatial validity: are the label-associated spatial patterns explained by the decoder alone? Tested with permutation nulls, a random decoder, a frozen synthesizer and AAL/Harvard-Oxford mapping.
- RQ5 Harmonization (new hypothesis): synthesized fMRI is a montage-independent common space. Does it transfer better across SZ cohorts with different caps, rates and paradigms than STFT or band power does? Train on cohort A, test on cohort B.

## Clinical cohorts (Frida `data/`)

| Cohort | ID | n (pt/HC) | Paradigm | EEG | Notes |
|---|---|---|---|---|---|
| Fribourg | ds004000 | 19/24 | task (Ultimatum) | 128ch 2048Hz | original paper; group only |
| FEP rest 1 | ds003944 | 50/32 | rest 5 min | 61ch 1000Hz | demographics, BPRS/SANS/SAPS, CPZ-equivalent dose |
| FEP rest 2 | ds003947 | 31/30 | rest 5 min | 61ch 1000Hz | check subject overlap with ds003944 |
| Semmelweis | Zenodo 14808296 | 38/39 | rest | 64ch 1000Hz | age/sex matched; clinical sheet |
| ASEEG | Zenodo 18029536 | 51/50 | rest EO/EC | 19ch 500Hz | optional |
| RepOD | 10.18150/repod.0107441 | 14/14 | rest EC | 19ch 250Hz | optional, med washout |
| PD (CATD's cohort) | ds003490 | ~25/25 | rest + oddball | 64ch | head-to-head with CATD (they used n=8) |

Pretraining: NODDI (OSF 94c5t; raw 5 kHz + BVA export). Transfer/fidelity: ds000116 Oddball (EEG only in legacy OpenfMRI tarballs), XP1 ds002336, XP2 ds002338, CN-EPFL ds002158; optionally NatView (rest, preprocessed, 22 subj).

Shared with the uncertainty session: `data/` (owner writes `.DONE`), `envs/tf29` (read-only for them).

## Synthesizers

| Model | Output | Code | Plan |
|---|---|---|---|
| Ours (MLHC 2023, CNN-TAG + RFF/style) | 64×64×30 vol | own (TF) | port to PyTorch; TF reproduction once |
| E2fNet (EMBC 2025) | 30×64×64 vol | MIT, no weights | train |
| Spec2VolCAMU-Net (JNE 2025) | vol | code, no license, weights removed | train (research use; ask authors re license) |
| NT-ViT (ECCVW 2024) | vol | code, no license, no weights | train |
| NeuroBOLT (NeurIPS 2024) / UnEBOLT (MIDL 2026) | ROI (DiFuMo-64) | GPL-3, weights | retrain on NODDI ROIs; ROI-level arm |
| CATD (TMI 2025) | surface (fsaverage4) | none | simplified re-implementation, or report as not reproducible |
| CNN-TC (Liu 2019) | vol | our re-implementation `src/benchmarks/liu.py` | port |
| Linear ridge (STFT→voxels) | vol | trivial | lower bound |

## Diagnostic evaluation

- **Views:**
  - raw EEG with the resampling bug fixed;
  - STFT;
  - band power;
  - EEGNet / ShallowConvNet;
  - synthesized fMRI from each synthesizer, both frozen and contrastive-fine-tuned (ours);
  - fusions of STFT with the synthesized fMRI.
- **Classifiers:** L1/L2 logistic regression (main), plus one nonlinear model as a check.
- **Validation:** nested subject-level CV (LOSO outer). Metrics are computed on subject-level aggregated probabilities. Bootstrap CIs; DeLong / paired permutation tests for ΔAUC; Brier score and calibration; multiple seeds.
- **Controls:**
  - label permutation;
  - random-init decoder;
  - synthesizer without fMRI pretraining;
  - contrastive encoder without decoder;
  - demonstration of the LOOCV anti-learning artifact.
- **Confounds:** age and sex are in participants.tsv for ds004000 (check); medication; recording-level artifacts.

## Milestones

- M0 Data and environments: ds004000, NODDI (OSF 94c5t), ds000116, ds002158, ds002336/8, ds003490. TF 2.9 env for reproduction.
- M1 Reproduce 2023 with the old TF code: STFT ≈0.93, fMRI ≈0.77 (window-pooled), recomputed at subject level.
- M2 PyTorch port of our model, with parity against TF on NODDI fidelity.
- M3 Fidelity benchmark (RQ1, RQ2) across all synthesizers.
- M4 Diagnostic benchmark (RQ3) plus controls (RQ4).
- M5 Paper (new title; target venue chosen after M4 results).
