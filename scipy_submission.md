
# Abstract
Recently, EEG to fMRI synthesis has received focus by the neuroscience research community. It provides neuroimaging cross modal synthesis from EEG to a synthesized fMRI modality. EEG-to-fMRI is a Python package that gives functionality for this task, being the first open source software enabling neuroimaging synthesis. This package was developed for neuroscience, machine learning and health care communities. We give a description of the methodology implemented in the package, along with results reported in a previous study.


# Motivation
Neuronal activity, measured through electroencephalography (EEG), is related to haemodynamical acitivity, measured through functional magnetic resonance imaging (fMRI). EEG captures the electrical field dynamics, whose source is located from the firing neurons' action potentials. In its turn, fMRI measures the blood supply. These two, while being studied simultaneously [\[1,2,3\]](#references), differ in: temporal and spatial resolution, brain functions captured, recording and hardware cost. Recently, studies that map EEG to fMRI have emerged. Such a mapping would reduce health care costs, MRI hardware is very scarce [\[4\]](#references), and enable discoveries of new neuroscience insights. As Python becomes a hub for scientific development we find the need, to provide open source software that provides solutions for EEG to fMRI synthesis, urgent, in order for third party scientific contributions coming from other laboratories to coexist and allow integration in an health care setting. To that end, we provide a description of the open source software [EEG-to-fMRI](https://pypi.org/project/eeg-to-fmri/) along with its public [repository](https://github.com/eeg-to-fmri/eeg-to-fmri).
# Methods
The mapping function provided in this software is the one proposed by [\[5\]](#references). It consists on transforming EEG from a channel by time representation to a channel by time frequency one, achieved using the short time Fourier transform [\[6\]](#references) by means of the fast Fourier transform ([```scipy.fft.fft```](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html)) available in the SciPy. This representation is then forward through a deep neural network (implemented as a [```tf.keras.Model```](https://www.tensorflow.org/api_docs/python/tf/keras/Model)), with contributions ranging from Resnet blocks [\[7\]](#references), an automated machine learning framework [\[8\]](#references) and Fourier features [\[9\]](#references). Ultimately, this enables the prediction of an fMRI volume associated with a $26$ second segment of an EEG recording. On top of this, the software package provides: explainability analysis tools, uncertainty quantification algorithms, classification using the fMRI predicted signal and several visualization functions. 
# Results
The package provides metrics for synthesis evaluation in ```eeg_to_fmri.metrics.quantitative_metrics```. We report results from the [\[5\]](#references) study on the NODDI dataset [\[10\]](#references). An example with a reduced dataset is available in [synthesis notebook](https://github.com/eeg-to-fmri/eeg-to-fmri/blob/main/examples/synthesis.ipynb). The best model, which used the configuration of the ```eeg_to_fmri.models.synthesizers.EEG_to_fMRI``` achieved 0.3972  RMSE and 0.4613 SSIM. This constitutes the state-of-the-art for this task and provides a view that can be applied in EEG only datasets for classification task (as shown in the [classification notebook](https://github.com/eeg-to-fmri/eeg-to-fmri/blob/main/examples/classification_contrastive.ipynb) example).
# Conclusion
This is the first package, to the best of our knowledge, that provides a machine learning oriented synthesis between neuroimaging modalities (EEG and fMRI). It is targeted to help the neuroscience community, in tasks such as modality augmentation, resolution enhancement, neuroimaging explainability techniques, among others.
# References
\[1\]: Rojas, Gonzalo M et al. 2018 Frontiers in neuroscience.
\[2\]: Abreu, Rodolfo et al. 2021 Brain topography.
\[3\]: Cury, Claire et al. 2020 Frontiers in neuroscience.
\[4\]: Ogbole, Godwin Inalegwu et al. 2018 Pan African Medical Journal.
\[5\]: Calhas, David et al. 2022 arXiv:2203.03481.
\[6\]: Allen, Jonathan 1977 IEEE Transactions ASSP.
\[7\]: He, Kaiming et al. 2016 CVPR.
\[8\]: Calhas, David et al. 2022 CLR AAAI Workshop.
\[9\]: Tancik, Matthew et al. 2020 arXiv:2006.10739.
\[10\]: Deligianni, Fani et al. 2014 Frontiers in Neuroscience.




# Examples
We encourage the use of the notebooks referenced in the description of this proposal. 
# Public speaking ability
I have previously participated in the following workshops:
- AAAI Combining Learning and Reasoning: Programming Languages, Formalisms, and Representations Workshop;
- Machine Learning in Science 2022, FCUL;
- Neurips The Symbiosis of Deep Learning and Differential Equations Workshop;
- INESC-ID Talks.
In addition, I am a teaching assistant since 2019 and have been doing theatre for one year. 