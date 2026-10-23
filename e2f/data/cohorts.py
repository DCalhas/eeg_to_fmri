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


EXPECTED = {"fribourg": (19, 24), "fep1": (47, 30), "fep2": (31, 30), "semmelweis": (38, 39), "pd": (25, 25)}


def _check(name, recs):
	"""Fail loudly if label logic or file discovery changed the group sizes (audit B6)."""
	n_pt = sum(r["label"] == 1 for r in recs); n_hc = sum(r["label"] == 0 for r in recs)
	assert (n_pt, n_hc) == EXPECTED[name], (name, n_pt, n_hc, EXPECTED[name])
	assert all(r["label"] in (0, 1) for r in recs)
	return recs


def fribourg(task="responder"):
	p = _bids_participants("ds004000")
	assert set(p.group) == {"HC", "P"}, set(p.group)
	return _check("fribourg", [dict(cohort="fribourg", sub=r.participant_id, label=int(r.group == "P"), fmt="brainvision", task=task,
				 path=os.path.join(DATA, "ds004000", r.participant_id, "eeg", f"{r.participant_id}_task-{task}_run-1_eeg.vhdr"),
				 meta={}) for r in p.itertuples()])


def _fep(ds, task, dedupe=True):
	"""dedupe: ds003944 holds 5 people twice (sub-X and sub-XA, same age/sex/race, days apart); keep sub-X."""
	p = _bids_participants(ds)
	assert set(p.type) == {"Control", "Psychosis"}, set(p.type)
	ids = set(p.participant_id)
	recs = []
	for r in p.itertuples():
		if dedupe and r.participant_id.endswith("A") and r.participant_id[:-1] in ids:
			continue
		d = os.path.join(DATA, ds, r.participant_id, "eeg")
		vhdr = [f for f in os.listdir(d) if f.endswith(".vhdr")]
		assert len(vhdr) == 1, (r.participant_id, vhdr)
		recs.append(dict(cohort=ds, sub=r.participant_id, label=int(r.type == "Psychosis"), fmt="brainvision", task=task,
						 path=os.path.join(d, vhdr[0]), meta=dict(age=r.age, sex=r.gender, race=r.race)))
	return recs


def fep1():
	return _check("fep1", _fep("ds003944", "Rest"))


def fep2():
	return _check("fep2", _fep("ds003947", "rest"))


def semmelweis():
	d = os.path.join(DATA, "semmelweis")
	x = pd.read_excel(os.path.join(d, "schizophrenia_EI_dynamics_table.xlsx"), sheet_name=None)
	demo, patients = x["Demographic"], set(x["PANSS"].code)  # patient = has PANSS (38/39 as in Racz et al. 2025);
	# 3 PANSS-rated patients lack an F2x BNO code (sch_023 'NMDA', sch_045, sch_047), so BNO alone gives 35/42
	files = os.path.join(d, "extracted", "sch_resting_64ch_ec")
	recs = []
	for r in demo.itertuples():
		f = os.path.join(files, f"{r.code}_ec.dat")
		if not os.path.exists(f):
			continue
		bno = str(r.BNO)
		recs.append(dict(cohort="semmelweis", sub=r.code, label=int(r.code in patients), fmt="curry", task="rest_ec",
						 path=f, meta=dict(age=r.age, sex=r.sex, bno=bno, edu_years=r.edu_years)))
	return _check("semmelweis", recs)


def pd_cavanagh(med="OFF"):
	"""ds003490 resting session. Patients: the session recorded with `med` status; controls: ses-01."""
	p = _bids_participants("ds003490")
	assert set(p.Group) == {"PD", "CTL"}, set(p.Group)
	recs = []
	for r in p.itertuples():
		if r.Group == "PD":
			s1, s2 = str(r.sess1_Med).upper(), str(r.sess2_Med).upper()
			assert {s1, s2} == {"ON", "OFF"}, (r.participant_id, s1, s2)
			ses = "01" if s1 == med.upper() else "02"
		else:
			ses = "01"
		f = os.path.join(DATA, "ds003490", r.participant_id, f"ses-{ses}", "eeg", f"{r.participant_id}_ses-{ses}_task-Rest_eeg.set")
		recs.append(dict(cohort="ds003490", sub=r.participant_id, label=int(r.Group == "PD"), fmt="eeglab", task="Rest",
						 path=f, meta=dict(age=r.age, sex=r.sex, session=ses, med=med if r.Group == "PD" else "n/a")))
	return _check("pd", recs)


COHORTS = {"fribourg": fribourg, "fep1": fep1, "fep2": fep2, "semmelweis": semmelweis, "pd": pd_cavanagh}

_NONEEG = {"HEO": "eog", "VEO": "eog", "VEOG": "eog", "HEOG": "eog", "EOG": "eog", "EKG": "ecg", "ECG": "ecg", "EMG": "emg",
		   "Trigger": "stim", "TRIG": "stim", "Status": "stim", "X": "misc", "Y": "misc", "Z": "misc", "RESP": "resp"}
_BIDS_TYPES = {"eeg": "eeg", "eog": "eog", "veog": "eog", "heog": "eog", "ecg": "ecg", "emg": "emg", "misc": "misc",
			   "stim": "stim", "trig": "stim", "ref": "misc", "other": "misc", "resp": "resp"}


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
		assert len(ch) == len(raw.ch_names), (rec["sub"], len(ch), len(raw.ch_names))
		if list(ch["name"]) != raw.ch_names:
			# generic file names (EEG001..) -> BIDS names, positional; only allowed when the file names are generic
			assert all(n.upper().startswith(("EEG", "EOG", "ECG", "MISC", "STI")) for n in raw.ch_names), raw.ch_names[:5]
			raw.rename_channels(dict(zip(raw.ch_names, ch["name"])))
		types = {n: _BIDS_TYPES[str(t).lower()] for n, t in zip(ch["name"], ch["type"]) if str(t).lower() in _BIDS_TYPES}
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
