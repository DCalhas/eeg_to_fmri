#first run
python main.py metrics 01 -topographical_attention -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 1
python main.py metrics 01 -fourier_features -topographical_attention -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 1

#recover pvalues
python main.py metrics 01 -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 1
python main.py metrics 01 -fourier_features -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 1
python main.py metrics 01 -topographical_attention -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 1
python main.py metrics 01 -fourier_features -topographical_attention -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 1

#quality checkup
python main.py residues 01 -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 1
python main.py residues 01 -fourier_features -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 1
python main.py residues 01 -topographical_attention -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 1
python main.py residues 01 -fourier_features -topographical_attention -na_path /home/ist_davidcalhas/eeg_to_fmri/na_models/na_specification_1 -verbose -gpu_mem 2000 -batch_size 1