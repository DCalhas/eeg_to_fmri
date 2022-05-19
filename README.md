# From EEG to fMRI







## Documentation

The code of this PhD thesis is extensive. Please read this section carefully if you have any issue with the code before you publish any issue or contact the authors. If any issue still persists after reading the documentation, please publish an issue on the Github repository.

### Main.py

The main.py file located at src/main.py is the script used for retrieve the results present in the [EEG to fMRI Synthesis](https://arxiv.org/abs/2203.03481) paper.

> The arguments given to the file of the form:
> $ python main.py <ARG1> <ARG2> ... <ARGN>
> are as follows:
> 	**Required:**
>	- *mode*
>		- metrics: the model is trained on a selected dataset and metrics, such as RMSE, SSIM and [Sharpness](https://arxiv.org/abs/1609.04836) (for uncertainty) are computed;
>		- residues: plots of the residues, on the test set data, in a white to black (bad to good, respectively) scale are retrieved. Example: TODO
>		- quality: retrieves the plots of each synthesized fMRI view.
>		- lrp_eeg_channels: Propagate the relevances from $\hat{y}$ to the channels [graph representation](https://github.com/DCalhas/eeg_to_fmri/blob/master/src/layers/topographical_attention.py). This is done using the [Layer-wise Relevance Propagation](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140). 
>		- lrp_eeg_fmri: Propagate the relevances from the latent representation $z_y$ to $y$.



## Acknowledgements

