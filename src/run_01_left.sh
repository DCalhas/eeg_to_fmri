#liu
python benchmarks/liu.py metrics 01 -interval_eeg 1 -save -memory_limit 3000 -seed 2
python benchmarks/liu.py metrics 01 -interval_eeg 1 -save -memory_limit 3000 -seed 3
python benchmarks/liu.py metrics 01 -interval_eeg 1 -save -memory_limit 3000 -seed 5
python benchmarks/liu.py metrics 01 -interval_eeg 1 -save -memory_limit 3000 -seed 7
python benchmarks/liu.py metrics 01 -interval_eeg 1 -save -memory_limit 3000 -seed 11

#ii w/o
python -u main.py metrics 01 -topographical_attention -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 2
python -u main.py metrics 01 -topographical_attention -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 3
python -u main.py metrics 01 -topographical_attention -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 5
python -u main.py metrics 01 -topographical_attention -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 7
python -u main.py metrics 01 -topographical_attention -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 11

#ii w prior
python -u main.py metrics 01 -topographical_attention -conditional_attention_style -conditional_attention_style_prior -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 2
python -u main.py metrics 01 -topographical_attention -conditional_attention_style -conditional_attention_style_prior -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 3
python -u main.py metrics 01 -topographical_attention -conditional_attention_style -conditional_attention_style_prior -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 5
python -u main.py metrics 01 -topographical_attention -conditional_attention_style -conditional_attention_style_prior -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 7
python -u main.py metrics 01 -topographical_attention -conditional_attention_style -conditional_attention_style_prior -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 3000 -batch_size 4 -save_metrics -seed 11
