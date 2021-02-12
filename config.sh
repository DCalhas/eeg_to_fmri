#install nvidia-driver-450
#sudo apt-get purge nvidia-*
#sudo apt-get install nvidia-driver-450
#reboot


source $HOME/anaconda3/bin/activate
conda create -n eeg_fmri python=3.8
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate eeg_fmri
conda install -c anaconda cudatoolkit==11.0.221

#download cudnn
tar -xzvf cudnn-11.0-linux-x64-v8.0.1.13.tgz
mv cuda/include/cudnn*.h ../anaconda3/envs/eeg_fmri/include/
mv cuda/lib64/libcudnn* ../anaconda3/envs/eeg_fmri/lib/
rm -r cuda
chmod a+r ../anaconda3/envs/eeg_fmri/include/cudnn*.h ../anaconda3/envs/eeg_fmri/lib/libcudnn*

pip install tensorflow-gpu==2.4.0
pip install -r requirements.txt

echo "I: Setting up datasets directory"


FILE=datasets/zipped_datasets/01.zip
if [ -f "$FILE" ]; then
    echo "I: Datasets already downloaded."
    unzip $FILE -d datasets/.
else 
    echo "I: Please download datasets indicated in the Datasets Description of the paper"
    echo "I: Organize them in datasets/01 and datasets/02"
    echo "I: If the zipped datasets file is available please unzip them in the directories specified"
    printf "I: If not please download them from:\nFirst dataset:\nEEG1.zip: https://osf.io/mx8ze/download \nEEG2.zip: https://osf.io/2zmup/download\nfMRI.zip: https://osf.io/vd5yz/download\nSecond dataset: http://openfmri.s3.amazonaws.com/tarballs/ds000116_R2.0.0_raw.tgz\n"
    mkdir datasets
    mkdir datasets/01
    mkdir datasets/02
    mkdir datasets/zipped_datasets
fi


journalctl --disk-usage
journalctl --vacuum-time=3d