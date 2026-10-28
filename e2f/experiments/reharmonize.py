"""Artefact-handling and crop variants of the clinical harmonisation (review point 6). CPU only (Slurm -p amd).

Same steps as bandpower_baseline.harmonize_one (EEG channels by type -> 10-05 positions -> notch, 0.5-45 Hz,
average reference, 250 Hz -> crop -> spherical-spline interpolation onto the 57 NODDI channels), plus:
  ica          ICA (FastICA, 20 components, fitted on a 1 Hz high-passed copy) before interpolation; components
               removed if correlated with the EOG (mne find_bads_eog, z > 3, using only EOG channels present in every
               recording of the cohort, so the cleaning cannot differ by group through the channel set); muscle-like
               components (mne find_bads_muscle, threshold 0.5) are counted but not removed, because with the 45 Hz
               low-pass the criterion flags most components
  pdrest       Parkinson's disease only: crop = the marked eyes-closed/eyes-open spans (first to last `Eyes ...`
               event + 1 s) instead of 0-400 s, which contained a variable unmarked set-up period and the start of an
               auditory oddball task
  pdrest_ica   both
Writes runs/harmonized_<variant>/<cohort>/<sub>.npz (format of harmonized_v2) and runs/harmonized_<variant>/<cohort>_log.json
(per subject: crop, excluded components by type, PD event timing).
"""
import os, json, argparse
import numpy as np
import pandas as pd
import mne

from e2f.data import cohorts, preprocess as P
from e2f.experiments.bandpower_baseline import LINE, CROP

RUNS = os.path.expanduser("~/eeg_to_fmri/runs")


def pd_rest_span(rec):
	ev = pd.read_csv(rec["path"].replace("_eeg.set", "_events.tsv"), sep="\t")
	eyes = ev[ev.trial_type.astype(str).str.startswith("Eyes")]
	tones = ev[ev.trial_type.astype(str).str.contains("Tone")]
	ec = eyes[eyes.trial_type.str.startswith("Eyes Closed")]; eo = eyes[eyes.trial_type.str.startswith("Eyes Open")]
	return {"t0": float(eyes.onset.min()), "t1": float(eyes.onset.max() + 1.0), "first_tone": float(tones.onset.min()) if len(tones) else None,
			"ec_s": float(len(ec)), "eo_s": float(len(eo)), "ec_first": bool(len(ec) and (not len(eo) or ec.onset.min() < eo.onset.min()))}


def common_eog(cohort, recs):
	common = None
	for r in recs:
		raw = cohorts.load_raw(r, preload=False)
		names = {c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eog"}
		common = names if common is None else common & names
	return sorted(common or [])


def one(cohort, rec, variant, eog_names):
	out_dir = os.path.join(RUNS, f"harmonized_{variant}", cohort)
	f = os.path.join(out_dir, f"{rec['sub']}.npz")
	mne.set_log_level("ERROR")
	raw = cohorts.load_raw(rec, preload=True)
	log = {"sub": rec["sub"], "label": rec["label"]}
	crop = CROP[cohort]
	if variant.startswith("pdrest"):
		span = pd_rest_span(rec); log.update(span); crop = (span["t0"], span["t1"])
	eog = raw.copy().pick(eog_names) if (variant.endswith("ica") and eog_names) else None
	el = rec["path"].replace("_task-responder_run-1_eeg.vhdr", "_electrodes.tsv") if cohort == "fribourg" else None
	h = P.set_positions(raw, el)
	h = P.clean(h, line_freq=LINE[cohort])
	h.crop(crop[0], min(crop[1], h.times[-1]))
	if variant.endswith("ica"):
		hp = h.copy().filter(1.0, None, verbose=0)
		ica = mne.preprocessing.ICA(n_components=20, method="fastica", random_state=0, max_iter="auto").fit(hp)
		bad_eog, bad_mus = [], []
		if eog is not None:
			eog.load_data()  # same filters and rate as the EEG, no re-referencing
			eog.notch_filter(np.arange(LINE[cohort], min(eog.info["sfreq"] / 2, 3 * LINE[cohort] + 1), LINE[cohort]), picks="all", verbose=0)
			eog.filter(0.5, 45.0, picks="all", verbose=0).resample(P.SFREQ, verbose=0)
			eog = eog.crop(crop[0], min(crop[1], eog.times[-1]))
			n = min(eog.n_times, hp.n_times)
			comb = hp.copy().crop(0, (n - 1) / hp.info["sfreq"])
			e2 = eog.copy().crop(0, (n - 1) / eog.info["sfreq"]).set_channel_types({c: "eog" for c in eog.ch_names}, on_unit_change="ignore")
			e2.set_meas_date(comb.info["meas_date"])
			comb.add_channels([e2], force_update_info=True)
			bad_eog, _ = ica.find_bads_eog(comb, ch_name=eog.ch_names, threshold=3.0)
		try:
			bad_mus, _ = ica.find_bads_muscle(hp, threshold=0.5)
		except Exception as e:
			log["muscle_error"] = repr(e)[:120]
		ica.exclude = sorted(set(bad_eog))  # ocular components only: with the 45 Hz low-pass, find_bads_muscle flags most components (12-16 of 20 in the smoke test); muscle is handled by the no-gamma sensitivity view
		h = ica.apply(h)
		log.update(n_ic_eog=len(set(bad_eog)), n_ic_muscle=len(set(bad_mus)), n_ic_excluded=len(ica.exclude))
	h = P.to_noddi_montage(h)
	os.makedirs(out_dir, exist_ok=True)
	tmp = f + ".tmp.npz"
	np.savez(tmp, eeg=h.get_data().astype(np.float32), sfreq=h.info["sfreq"], ch=np.array(h.ch_names), label=rec["label"],
			 duration=h.n_times / h.info["sfreq"])
	os.replace(tmp, f)
	log["duration"] = h.n_times / h.info["sfreq"]
	return log


if __name__ == "__main__":
	from joblib import Parallel, delayed
	ap = argparse.ArgumentParser()
	ap.add_argument("--variant", required=True, choices=["ica", "pdrest", "pdrest_ica"])
	ap.add_argument("--cohorts", nargs="+", default=["fep1", "fep2", "semmelweis", "pd", "fribourg"])
	ap.add_argument("--n_jobs", type=int, default=32)
	ap.add_argument("--max_subjects", type=int, default=None, help="smoke test")
	a = ap.parse_args()
	for c in (["pd"] if a.variant.startswith("pdrest") else a.cohorts):
		recs = cohorts.COHORTS[c]()[:a.max_subjects] if a.max_subjects else cohorts.COHORTS[c]()
		eog_names = common_eog(c, recs) if a.variant.endswith("ica") else []
		logs = Parallel(n_jobs=a.n_jobs)(delayed(one)(c, r, a.variant, eog_names) for r in recs)
		if not a.max_subjects:
			json.dump({"eog_channels": eog_names, "subjects": logs}, open(os.path.join(RUNS, f"harmonized_{a.variant}", f"{c}_log.json"), "w"), indent=1, default=float)
		else:
			print(json.dumps(logs, default=float)[:3000], flush=True)
		print(c, a.variant, "eog", eog_names, "done", len(logs), flush=True)
