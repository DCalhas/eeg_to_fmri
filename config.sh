#install nvidia-driver-450
#sudo apt-get purge nvidia-*
#sudo apt-get install nvidia-driver-450
#reboot


source $HOME/anaconda3/bin/activate
conda create -n eeg_fmri python=3.8
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate eeg_fmri
conda install -c conda-forge cudatoolkit==11.2

#download cudnn
if [ ! -f cudnn-11.2-linux-x64-v8.1.1.33.tgz ]
then
    wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.1.33/11.2_20210301/cudnn-11.2-linux-x64-v8.1.1.33.tgz
fi
tar -xzvf cudnn-11.2-linux-x64-v8.1.1.33.tgz
mv cuda/include/cudnn*.h $CONDA_PREFIX/include/
mv cuda/lib64/libcudnn* $CONDA_PREFIX/lib/
rm -r cuda
chmod a+r $CONDA_PREFIX/include/cudnn*.h $CONDA_PREFIX/lib/libcudnn*
#set up variables when activating envorimonent
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

pip install tensorflow-gpu==2.9.0
pip install -r requirements.txt

echo "I: Setting up datasets directory"
echo "Now you have to set the path for the datasets."
echo -n "Please specify a path for the datasets (the default is "$PWD"/datasets):"
read EEG_FMRI_DATASETS

echo 'export EEG_FMRI='$PWD >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export EEG_FMRI_DATASETS='$EEG_FMRI_DATASETS >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh