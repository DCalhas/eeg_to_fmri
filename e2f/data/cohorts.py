"""Unified access to the clinical EEG cohorts on Frida (~/eeg_to_fmri/data).

Each cohort returns a list of records: dict(cohort, sub, label (1=patient), path, fmt, task, meta).
`load_raw(rec)` returns an mne Raw with real 10-10 channel names (from channels.tsv where the file
uses generic names) and non-EEG channels typed as such, so downstream code can drop them by type
rather than by position (the 2023 Fribourg bug).
"""
import os
import numpy as np
import pandas as pd
import mne

DATA = os.path.expanduser("~/eeg_to_fmri/data")


def _bids_participants(ds):
	return pd.read_csv(os.path.join(DATA, ds, "participants.tsv"), sep="\t")


def fribourg(task="responder"):
	p = _bids_participants("ds004000")
	return [dict(cohort="fribourg", sub=r.participant_id, label=int(r.group == "P"), fmt="brainvision", task=task,
				 path=os.path.join(DATA, "ds004000", r.participant_id, "eeg", f"{r.participant_id}_task-{task}_run-1_eeg.vhdr"),
				 meta={}) for r in p.itertuples()]


def _fep(ds, task):
	p = _bids_participants(ds)
	recs = []
	for r in p.itertuples():
		d = os.path.join(DATA, ds, r.participant_id, "eeg")
		vhdr = [f for f in os.listdir(d) if f.endswith(".vhdr")]
		assert len(vhdr) == 1, (r.participant_id, vhdr)
		recs.append(dict(cohort=ds, sub=r.participant_id, label=int(r.type == "Psychosis"), fmt="brainvision", task=task,
						 path=os.path.join(d, vhdr[0]), meta=dict(age=r.age, sex=r.gender, race=r.race)))
	return recs


def fep1():
	return _fep("ds003944", "Rest")


def fep2():
	return _fep("ds003947", "rest")


def semmelweis():
	d = os.path.join(DATA, "semmelweis")
	demo = pd.read_excel(os.path.join(d, "schizophrenia_EI_dynamics_table.xlsx"), sheet_name="Demographic")
	files = os.path.join(d, "extracted", "sch_resting_64ch_ec")
	recs = []
	for r in demo.itertuples():
		f = os.path.join(files, f"{r.code}_ec.dat")
		if not os.path.exists(f):
			continue
		bno = str(r.BNO)
		recs.append(dict(cohort="semmelweis", sub=r.code, label=int(bno.startswith("F2")), fmt="curry", task="rest_ec",
						 path=f, meta=dict(age=r.age, sex=r.sex, bno=bno, edu_years=r.edu_years)))
	return recs


def pd_cavanagh(med="OFF"):
	"""ds003490 resting session. Patients: the session recorded with `med` status; controls: ses-01."""
	p = _bids_participants("ds003490")
	recs = []
	for r in p.itertuples():
		if r.Group == "PD":
			ses = "01" if r.sess1_Med == med else "02"
		else:
			ses = "01"
		f = os.path.join(DATA, "ds003490", r.participant_id, f"ses-{ses}", "eeg", f"{r.participant_id}_ses-{ses}_task-Rest_eeg.set")
		recs.append(dict(cohort="ds003490", sub=r.participant_id, label=int(r.Group == "PD"), fmt="eeglab", task="Rest",
						 path=f, meta=dict(age=r.age, sex=r.sex, session=ses, med=med if r.Group == "PD" else "n/a")))
	return recs


COHORTS = {"fribourg": fribourg, "fep1": fep1, "fep2": fep2, "semmelweis": semmelweis, "pd": pd_cavanagh}

_NONEEG = {"HEO": "eog", "VEO": "eog", "VEOG": "eog", "HEOG": "eog", "EKG": "ecg", "ECG": "ecg", "EMG": "emg",
		   "Trigger": "stim", "X": "misc", "Y": "misc", "Z": "misc"}


def load_raw(rec, preload=False):
	fmt, path = rec["fmt"], rec["path"]
	if fmt == "brainvision":
		raw = mne.io.read_raw_brainvision(path, preload=preload, verbose=0)
	elif fmt == "curry":
		raw = mne.io.read_raw_curry(path, preload=preload, verbose=0)
	elif fmt == "eeglab":
		raw = mne.io.read_raw_eeglab(path, preload=preload, verbose=0)
	else:
		raise ValueError(fmt)
	ch_tsv = path.replace("_eeg.vhdr", "_channels.tsv").replace("_eeg.set", "_channels.tsv")
	if os.path.exists(ch_tsv) and ch_tsv != path:
		ch = pd.read_csv(ch_tsv, sep="\t")
		if len(ch) == len(raw.ch_names):
			raw.rename_channels(dict(zip(raw.ch_names, ch["name"])))
			types = {n: str(t).lower() for n, t in zip(ch["name"], ch["type"]) if str(t).lower() in ("eeg", "eog", "ecg", "emg", "misc", "stim")}
			raw.set_channel_types(types, on_unit_change="ignore")
	types = {n: t for n, t in _NONEEG.items() if n in raw.ch_names}
	types.update({n: "eog" for n in raw.ch_names if n.startswith("EXG")})
	if types:
		raw.set_channel_types(types, on_unit_change="ignore")
	return raw


def header_summary(rec):
	"""Metadata that could leak the label: channel set, rate, duration, date, filters, file size."""
	raw = load_raw(rec)
	eeg = mne.pick_types(raw.info, eeg=True)
	names = [raw.ch_names[i] for i in eeg]
	md = raw.info["meas_date"]
	return dict(n_channels=len(raw.ch_names), n_eeg=len(eeg), ch_set=hash(tuple(sorted(raw.ch_names))) % 10**8,
				eeg_set=hash(tuple(sorted(names))) % 10**8, sfreq=raw.info["sfreq"], duration=raw.n_times / raw.info["sfreq"],
				highpass=raw.info["highpass"], lowpass=raw.info["lowpass"], n_annot=len(raw.annotations),
				year=md.year if md else np.nan, month=md.month if md else np.nan, hour=md.hour if md else np.nan,
				file_mb=_data_file_mb(rec["path"]))


def _data_file_mb(path):
	stem = os.path.splitext(path)[0]
	sizes = [os.path.getsize(stem + e) for e in (".eeg", ".dat", ".fdt", ".set") if os.path.exists(stem + e)]
	return max(sizes) / 1e6 if sizes else np.nan
