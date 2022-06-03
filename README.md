# From EEG to fMRI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


[![Code: Documentation](https://img.shields.io/badge/code-documentation-green)](https://github.com/DCalhas/eeg_to_fmri/blob/master/DOCUMENTATION.md)



## Setup

TODO


## How do I test this research on my dataset?

Testing a new dataset on this framework should not be too difficult. Do the following (in the order you feel most comfortable):
- define the number of individuals, **n_individuals_NEW**, that your dataset contains, this can be done in the [data_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/data_utils.py#L32) file;
- additionally you may define new variables in the [data_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/data_utils.py) file, corresponding to **n_individuals_train_NEW** and **n_individuals_test_NEW**, which refer to the number of individuals used for the training and testing set, respectively;
- define **dataset_NEW** variable in the [fmri_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/eeg_utils.py#L47) and [eeg_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/fmri_utils.py#L43) files. At this point you might be thinking: "Why is this guy defining the same variable in two different places?", well he ain't too smart tbh and he lazy af;
- define the frequency, **fs_NEW**, at which the EEG recording was sampled, this can be done in the [eeg_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/eeg_utils.py#L38) file;
- define the Time Response, **TR_NEW**, at which each fMRI volume was sampled, this can be done in the [fmri_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/fmri_utils.py#L27);
- additionally, you might want to define the list of channels (if your EEG electrode setup follows the [10-20 system](https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG))), to retrieve more advanced analysis, such as EEG electrode relevance. This should be done in the beginning of the [eeg_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/eeg_utils.py) file;
- last, but no least, comes the time to implement the two functions that read the EEG and fMRI recordings, corresponding to **get_eeg_instance_NEW**, at [eeg_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/eeg_utils.py#L171), and **get_indviduals_path_NEW**, at [fmri_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/fmri_utils.py#L299);

#### Implementing the get_eeg_instance_NEW function

Ideally you want this function to return an [mne.io.Raw](https://mne.tools/stable/generated/mne.io.Raw.html) object, that contains the EEG data. In this "tutorial" only this is the only supported option, however do it as you like most.

The inputs of this function are:
- *individual* - int, the individual one wants to retrieve. This function is being executed inside a for loop, ```for individual in range(getattr(data_utils, "n_individuals_"+dataset)```, that goes through the range of individuals, **n_individuals_NEW**, you set in the [data_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/data_utils.py#L32) file;
- *path_eeg* - str, the path where your dataset is located, e.g. ```path_eeg="/tmp/NEW/"```, this may be an optional argument set as ```path_eeg="/tmp/"+dataset_NEW+"/"```;
- *task* - str, can be set to None if it does not apply to your dataset;

So given these inputs one can start by listing the directories of your dataset (now this can depend on how you organized the data, we assume that each individual has a folder dedicated to itself and the sorted names of the folders have the individual's folders first and after the auxiliary description ones, e.g. "info" for information about the dataset):


```python
def get_eeg_instance_NEW(individual, path_eeg="/tmp/"+dataset_NEW+"/", task=None,):
	individuals = sorted([f for f in listdir(path_eeg) if isdir(join(path_eeg, f))])

	individual = individuals[individual]

	path=path_eeg+individual+"/"
	print(path)#for debug purposes only, please remove this line after function is implemented
```

```bash
/tmp/NEW/sub-001/
```

Inside the path described above should be a set of files needed to load a eeg brainvision object. If you sort these files, likely 
the ```.vhdr``` is the second option:

```python
	brainvision_files = sorted([f for f in listdir(path) if isfile(join(path, f))])
	vhdr_file = brainvision_files[1]
```

Now one only needs to return the Brainvision object:

```python
	complete_path = path + vhdr_file
	return mne.io.read_raw_brainvision(complete_path, preload=True, verbose=0)
```

#### Implementing the get_individuals_path_NEW function


```
fmri = :o
```

## Acknowledgements

We would like to thank everyone at [INESC-ID](https://www.inesc-id.pt/) that accompanied the journey throughout my PhD. This work was supported by national funds through Fundação para a Ciência e Tecnologia ([FCT](https://www.fct.pt/index.phtml.pt)), under the Ph.D. Grant SFRH/BD/5762/2020 to David Calhas, ILU project DSAIPA/DS/0111/2018 and INESC-ID pluriannual UIDB/50021/2020.

## Cite this repository

If you use this software in your work, please cite it using the following metadata:

```
@article{calhas2022eeg,
  title={EEG to fMRI Synthesis Benefits from Attentional Graphs of Electrode Relationships},
  author={Calhas, David and Henriques, Rui},
  journal={arXiv preprint arXiv:2203.03481},
  year={2022}
}
```


## License

[MIT License](https://choosealicense.com/licenses/mit/)
