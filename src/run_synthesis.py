#!/usr/bin

#i
python -u main.py metrics $1 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 2
python -u main.py metrics $1 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 3
python -u main.py metrics $1 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 5
python -u main.py metrics $1 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 7
python -u main.py metrics $1 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 11


#ii
python -u main.py metrics $1 -topographical_attention -conditional_attention_style -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 2
python -u main.py metrics $1 -topographical_attention -conditional_attention_style -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 3
python -u main.py metrics $1 -topographical_attention -conditional_attention_style -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 5
python -u main.py metrics $1 -topographical_attention -conditional_attention_style -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 7
python -u main.py metrics $1 -topographical_attention -conditional_attention_style -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 11


#iii
python -u main.py metrics $1 -fourier_features -random_fourier -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 2
python -u main.py metrics $1 -fourier_features -random_fourier -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 3
python -u main.py metrics $1 -fourier_features -random_fourier -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 5
python -u main.py metrics $1 -fourier_features -random_fourier -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 7
python -u main.py metrics $1 -fourier_features -random_fourier -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 11


#iv
python -u main.py metrics $1 -fourier_features -random_fourier -topographical_attention -conditional_attention_style -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 2
python -u main.py metrics $1 -fourier_features -random_fourier -topographical_attention -conditional_attention_style -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 3
python -u main.py metrics $1 -fourier_features -random_fourier -topographical_attention -conditional_attention_style -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 5
python -u main.py metrics $1 -fourier_features -random_fourier -topographical_attention -conditional_attention_style -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 7
python -u main.py metrics $1 -fourier_features -random_fourier -topographical_attention -conditional_attention_style -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 11

#w/o
python -u main.py metrics $1 -fourier_features -random_fourier -topographical_attention -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 2
python -u main.py metrics $1 -fourier_features -random_fourier -topographical_attention -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 3
python -u main.py metrics $1 -fourier_features -random_fourier -topographical_attention -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 5
python -u main.py metrics $1 -fourier_features -random_fourier -topographical_attention -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 7
python -u main.py metrics $1 -fourier_features -random_fourier -topographical_attention -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 11

#w prior

#iv
python -u main.py metrics $1 -fourier_features -random_fourier -topographical_attention -conditional_attention_style -conditional_attention_style_prior -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 2
python -u main.py metrics $1 -fourier_features -random_fourier -topographical_attention -conditional_attention_style -conditional_attention_style_prior -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 3
python -u main.py metrics $1 -fourier_features -random_fourier -topographical_attention -conditional_attention_style -conditional_attention_style_prior -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 5
python -u main.py metrics $1 -fourier_features -random_fourier -topographical_attention -conditional_attention_style -conditional_attention_style_prior -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 7
python -u main.py metrics $1 -fourier_features -random_fourier -topographical_attention -conditional_attention_style -conditional_attention_style_prior -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 11
