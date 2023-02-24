
# Introduction

Neuronal activity, usually measured through electroencephalography (EEG), is related to haemodynamical acitivity, measured through functional magnetic resonance imaging (fMRI). The first captures the dynamics of the electrical field, whose source is located from the firing neurons' action potentials. In its turn, the second measures the blood supply dynamics. These two while being studied simultaneously \cite{rojas2018study,abreu2021eeg,cury2020sparse} differ in many aspects such as: temporal and spatial resolution, brain functions captured, recording and hardware cost. Recently, we have seen the contribution of multiple studies that target a mapping function able to provide a synthesized fMRI view from an EEG signal. Such a mapping could allow health care cost reductions and discoveries of new neuroscience insights on the relationship between these two modalities. Indeed, pathologies that require MRI scans diagnostics benefit from a lower cost EEG assessment, since availability of MRI hardware is very scarce \cite{ogbole2018survey}. As Python \cite{van1995python} becomes a hub for scientific development \cite{harris2020array,virtanen2020scipy,abadi2016tensorflow} we find, the need to provide open source software that provides solutions for \textit{EEG to fMRI synthesis}, urgent, in order for third party scientific contributions coming other laboratories to coexist and health care software integration to develop for diagnostic settings. To that end, we provide a description of the open source software [EEG-to-fMRI](https://pypi.org/project/eeg-to-fmri/), which originated from an academic scientific project funded by FCT, and make publicly available a github [repository](https://github.com/eeg-to-fmri/eeg-to-fmri).

# Methods

The mapping function provided in this software is the one proposed by \citet{calhas2022eeg}. It consists on transforming EEG from a channel by time representation to a channel by time frequency one, achieved using the short time Fourier transform \cite{allen1977short} by means of the fast Fourier transform ([scipy.fft.fft](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html)) available in the SciPy. This representation is then forward through a deep neural network (implemented as a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)), with contributions ranging from Resnet blocks \cite{he2016deep}, an automated machine learning framework \cite{calhas2022automatic} and Fourier features \cite{tancik2020fourier}. Ultimately, this enables the prediction of an fMRI volume associated with a $26$ second segment of an EEG recording. Figure \ref{fig:eeg_fmri_synthesis} illustrates this methodology. On top of this, the software package provides: explainability analysis tools, uncertainty quantification algorithms, classification using the fMRI predicted signal and several visualization functions. 

# Results

The package provides metrics for synthesis evaluation in eeg\_to\_fmri.metrics.quantitative\_metrics. We report results from the \citet{calhas2022eeg} study on the NODDI dataset \cite{dataset2014noddi}. An example with a reduced dataset is available in [synthesis notebook](https://github.com/eeg-to-fmri/eeg-to-fmri/blob/main/examples/synthesis.ipynb). The best model, which used the configuration of the eeg\_to\_fmri.models.synthesizers.EEG\_to\_fMRI achieved $0.3972$  RMSE and $0.4613$ SSIM. This constitutes the state-of-the-art for this task and provides a view that can be applied in EEG only datasets for classification task (as shown in the [classification notebook](https://github.com/eeg-to-fmri/eeg-to-fmri/blob/main/examples/classification_contrastive.ipynb) example).

# Conclusion

This is the first package, to the best of our knowledge, that provides a machine learning oriented synthesis between neuroimaging modalities (EEG and fMRI). It is targeted to help the neuroscience community, in tasks such as modality augmentation, resolution enhancement, neuroimaging explainability techniques, among others.
 [\[1\]](#references)

# References

\[1\]: [Bromley, J., Guyon, I., LeCun, Y., Säckinger, E., & Shah, R. (1993). Signature verification using a" siamese" time delay neural network. Advances in neural information processing systems, 6.](https://proceedings.neurips.cc/paper/1993/file/288cc0ff022877bd3df94bc9360b9c5d-Paper.pdf)

