python clf_cv.py $1 fmri -dataset_synth $2 -epochs 10 -feature_selection -segmentation_mask -variational -variational_dependent_h 20 -aleatoric_uncertainty -variational_dist VonMises -variational_coefs 32,32,15 -resolution_decoder 2 -gpu_mem 4000 -path_labels /home/ist_davidcalhas/eeg_to_fmri/metrics/ -save_explainability -seed 2
#python clf_cv.py $1 fmri -dataset_synth $2 -epochs 0 -feature_selection -segmentation_mask -aleatoric_uncertainty -seed 2
