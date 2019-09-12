library(eeguana)
eeg_01 = read_vhdr("/home/david/eeg_informed_fmri/datasets/01/EEG/32/raw/20130410320002.vhdr")
plot(eeg_01[['F3']])
