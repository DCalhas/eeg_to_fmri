#first run -save_metrics
#python -u main.py metrics 01 -learning_rate 0.0001 -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 4 -save_metrics
#python -u main.py metrics 01 -fourier_features -learning_rate 0.0001 -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 4 -save_metrics
#python -u main.py metrics 01 -topographical_attention -learning_rate 0.0001 -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 4 -save_metrics
#python -u main.py metrics 01 -fourier_features -topographical_attention -learning_rate 0.0001 -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 4 -save_metrics

#recover pvalues -save_metrics
#python -u main.py metrics 01 -learning_rate 0.0001 -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 4 -save_metrics
#python -u main.py metrics 01 -fourier_features -learning_rate 0.0001 -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 4 -save_metrics
#python -u main.py metrics 01 -topographical_attention -learning_rate 0.0001 -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 4 -save_metrics
#python -u main.py metrics 01 -fourier_features -topographical_attention -learning_rate 0.0001 -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 4 -save_metrics

#quality checkup
#python -u main.py residues 01 -learning_rate 0.0001 -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 4 -save_metrics
#python -u main.py residues 01 -fourier_features -learning_rate 0.0001 -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 4 -save_metrics
#python -u main.py residues 01 -topographical_attention -learning_rate 0.0001 -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 4 -save_metrics
#python -u main.py residues 01 -fourier_features -topographical_attention -learning_rate 0.0001 -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 4 -save_metrics

#mean residues
python -u main.py mean_residues 01 -learning_rate 0.0001 -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 4
python -u main.py mean_residues 01 -fourier_features -learning_rate 0.0001 -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 4
python -u main.py mean_residues 01 -topographical_attention -learning_rate 0.0001 -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 4
python -u main.py mean_residues 01 -fourier_features -topographical_attention -learning_rate 0.0001 -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 4