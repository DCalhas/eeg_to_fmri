
printf "import bicpy;bicpy.run(bicpy.DEFAULT_PARAMS, \"/home/ist_davidcalhas/eeg_to_fmri/src/notebooks/view_$1.arff\")" > ./eeg_fmri.py; python ./eeg_fmri.py; rm ./eeg_fmri.py

mv output/result.txt ../$2/demo.txt
