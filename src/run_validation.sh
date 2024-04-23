python validate.py encoder_conv encoder_nonlocal -splits 20 > local_vs_nonlocal.txt
python validate.py encoder_conv encoder_attention -splits 20 > local_vs_attention.txt
python validate.py encoder_nonlocal encoder_attention -splits 20 > nonlocal_vs_attention.txt