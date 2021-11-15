import tensorflow.compat.v1 as tf

import numpy as np

import utils.losses_utils as losses

from utils import bnn_utils

import matplotlib.pyplot as plt

from matplotlib import cm

from matplotlib.ticker import LinearLocator

from matplotlib.gridspec import GridSpec

from scipy import spatial

from numpy.linalg import norm

from nilearn import plotting, image

from scipy.ndimage import rotate

from pathlib import Path

import pathlib

from scipy.stats import gamma

uncertainty_plots_directory = str(Path.home())+"/eeg_to_fmri/src/results_plots/uncertainty/"
uncertainty_losses_plots_directory = str(Path.home())+"/eeg_to_fmri/src/results_plots/uncertainty_losses/"
gamma_plots_directory = str(Path.home())+"/eeg_to_fmri/src/results_plots/gamma/"

def get_models_and_shapes(eeg_file='../../optimized_nets/eeg/eeg_30_partitions.json', 
                        bold_file='../../optimized_nets/bold/bold_30_partitions.json',
                        decoder_file='../../optimized_nets/decoder/decoder_30_partitions.json'):

    json_file = open(eeg_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    eeg_network = tf.keras.models.model_from_json(loaded_model_json)

    json_file = open(bold_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    bold_network = tf.keras.models.model_from_json(loaded_model_json)

    json_file = open(decoder_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    decoder_network = tf.keras.models.model_from_json(loaded_model_json)

    return eeg_network, bold_network, decoder_network


def _plot_mean_std(reconstruction_loss, distance, tset="train", n_partitions=30, model="M", ax=None):

    inds_ids = []
    inds_mean = np.zeros(len(reconstruction_loss)//n_partitions)
    inds_std = np.zeros(len(reconstruction_loss)//n_partitions)

    #compute mean 
    for ind in range(inds_mean.shape[0]):
        inds_ids += ['Ind_' + str(ind+1)]
        inds_mean[ind] = np.mean(reconstruction_loss[ind:ind+n_partitions])
        inds_std[ind] = np.std(reconstruction_loss[ind:ind+n_partitions])

    print(tset + " set", "mean: ", np.mean(reconstruction_loss))
    print(tset + " set", "std: ", np.std(reconstruction_loss))


    ax.errorbar(inds_ids, inds_mean, inds_std, linestyle='None', elinewidth=0.5, ecolor='r', capsize=10.0, markersize=10.0, marker='o')
    ax.set_title(distance + " on " + tset + " set " + " (" + model + ")")
    ax.set_xlabel("Individuals")
    if("Cosine" in distance):
        ax.set_ylabel("Correlation")
    else:
        ax.set_ylabel("Distance")

def _plot_mean_std_loss(synthesized_bold, bold, distance_function, distance_name, set_name, model_name, n_partitions=30, ax=None):
    reconstruction_loss = np.zeros((synthesized_bold.shape[0], 1))

    for instance in range(len(reconstruction_loss)):
        instance_synth = synthesized_bold[instance]
        instance_bold = bold[instance]

        instance_synth = instance_synth.reshape((1, instance_synth.shape[0], instance_synth.shape[1], instance_synth.shape[2]))
        instance_bold = instance_bold.reshape((1, instance_bold.shape[0], instance_bold.shape[1], instance_bold.shape[2]))

        reconstruction_loss[instance] = distance_function(instance_synth, instance_bold).numpy()

    _plot_mean_std(reconstruction_loss, distance=distance_name, tset=set_name, model=model_name, n_partitions=n_partitions, ax=ax)



def plot_mean_std_loss(eeg_train, bold_train, 
                        eeg_val, bold_val, 
                        eeg_test, bold_test, 
                        encoder_network, decoder_network, 
                        distance_name, distance_function,
                        model_name, n_partitions=30):

    n_plotted = 1

    n_plots = int(type(eeg_train) is np.ndarray and type(bold_train) is np.ndarray) + \
        int(type(eeg_val) is np.ndarray and type(bold_val) is np.ndarray) + \
        int(type(eeg_test) is np.ndarray and type(bold_test) is np.ndarray)

    plt.figure(figsize=(20,5))

    if(type(eeg_train) is np.ndarray and type(bold_train) is np.ndarray):
        ax1 = plt.subplot(1,n_plots,n_plotted)
        n_plotted += 1

        shared_eeg_train = encoder_network.predict(eeg_train)
        synthesized_bold_train = decoder_network.predict(shared_eeg_train)
        _plot_mean_std_loss(synthesized_bold_train, bold_train, distance_function, distance_name, "train", model_name, n_partitions=n_partitions, ax=ax1)

    if(type(eeg_val) is np.ndarray and type(bold_val) is np.ndarray):
        ax2 = plt.subplot(1,n_plots,n_plotted)
        n_plotted += 1

        shared_eeg_val = encoder_network.predict(eeg_val)
        synthesized_bold_val = decoder_network.predict(shared_eeg_val)
        _plot_mean_std_loss(synthesized_bold_val, bold_val, distance_function, distance_name, "validation", model_name, n_partitions=n_partitions, ax=ax2)

    

    if(type(eeg_test) is np.ndarray and type(bold_test) is np.ndarray):
        ax3 = plt.subplot(1,n_plots,n_plotted)
        n_plotted += 1

        shared_eeg_test = encoder_network.predict(eeg_test)
        synthesized_bold_test = decoder_network.predict(shared_eeg_test)
        _plot_mean_std_loss(synthesized_bold_test, bold_test, distance_function, distance_name, "test", model_name, n_partitions=n_partitions, ax=ax3)

    plt.show()

def plot_loss_results(eeg_train, bold_train, eeg_val, bold_val, eeg_test, bold_test, eeg_network, decoder_network, model_name, n_partitions=30):

    plot_mean_std_loss(eeg_train, bold_train, 
    eeg_val, bold_val, 
    eeg_test, bold_test, 
    eeg_network, decoder_network, 
    "Log Cosine", losses.get_reconstruction_log_cosine_loss,
    model_name, n_partitions=n_partitions)

    plot_mean_std_loss(eeg_train, bold_train, 
    eeg_val, bold_val, 
    eeg_test, bold_test, 
    eeg_network, decoder_network, 
    "Log Cosine Voxels Mean", losses.get_reconstruction_log_cosine_voxel_loss,
    model_name, n_partitions=n_partitions)

    plot_mean_std_loss(eeg_train, bold_train, 
    eeg_val, bold_val, 
    eeg_test, bold_test, 
    eeg_network, decoder_network, 
    "Cosine", losses.get_reconstruction_cosine_loss,
    model_name, n_partitions=n_partitions)

    plot_mean_std_loss(eeg_train, bold_train, 
    eeg_val, bold_val, 
    eeg_test, bold_test, 
    eeg_network, decoder_network, 
    "Cosine Voxels Mean", losses.get_reconstruction_cosine_voxel_loss,
    model_name, n_partitions=n_partitions)

    plot_mean_std_loss(eeg_train, bold_train, 
    eeg_val, bold_val, 
    eeg_test, bold_test, 
    eeg_network, decoder_network, 
    "Euclidean", losses.get_reconstruction_euclidean_loss,
    model_name, n_partitions=n_partitions)

    plot_mean_std_loss(eeg_train, bold_train, 
    eeg_val, bold_val, 
    eeg_test, bold_test, 
    eeg_network, decoder_network, 
    "Euclidean Per Volume", losses.get_reconstruction_euclidean_volume_loss,
    model_name, n_partitions=n_partitions)

    plot_mean_std_loss(eeg_train, bold_train, 
    eeg_val, bold_val, 
    eeg_test, bold_test, 
    eeg_network, decoder_network, 
    "Mean Absolute Error Per Volume", losses.get_reconstruction_absolute_volume_loss,
    model_name, n_partitions=n_partitions)

    plot_mean_std_loss(eeg_train, bold_train, 
    eeg_val, bold_val, 
    eeg_test, bold_test, 
    eeg_network, decoder_network, 
    "KL Loss", losses.get_reconstruction_kl_loss,
    model_name, n_partitions=n_partitions)


######################################################################################################################################################
#
#                                                            PLOT VOXELS REAL AND SYNTHESIZED
#
######################################################################################################################################################


def plot_view_mask(img, timestep=4, vmin=None, vmax=None, resampling_factor=4, symmetric_cmap=False, save_file="/tmp/plot.html"):
    img = image.index_img(img, timestep)

    if(vmin is None):
        vmin=np.amin(img.get_data())
    if(vmax is None):
        vmax=np.amax(img.get_data())

    view = plotting.view_img(img, 
                            threshold=None,
                            colorbar=True,
                            annotate=False,
                            draw_cross=False,
                            cut_coords=[0, 0,  0],
                            black_bg=True,
                            bg_img=False,
                            cmap="inferno",
                            symmetric_cmap=symmetric_cmap,
                            vmax=vmax,
                            vmin=vmin,
                            dim=-2,
                            resampling_interpolation="nearest")

    view.save_as_html(save_file)


def _plot_voxel(real_signal, synth_signal, rows=1, columns=2, index=1, y_bottom=None, y_top=None):
    ax = plt.subplot(rows, columns, index)
    ax.plot(list(range(0, len(real_signal)*2, 2)), real_signal, color='b')
    ax.set_xlabel("Seconds")
    ax.set_ylabel("BOLD intensity")
    
    if(y_bottom==None and y_top==None):
        y_bottom_real = np.amin(real_signal)
        y_top_real = np.amax(real_signal)
        y_bottom_synth = np.amin(synth_signal)
        y_top_synth = np.amax(synth_signal)
        
    ax.set_ylim(y_bottom_real, y_top_real)
    
    if(index == 1):
        ax.set_title("Real BOLD Signal", y=0.99999)

        
    
    ax = plt.subplot(rows, columns, index+1)
    ax.plot(list(range(0, len(synth_signal)*2, 2)), synth_signal, color='r')
    ax.set_xlabel("Seconds")
    ax.set_ylabel("BOLD intensity")
    
    ax.set_ylim(y_bottom_synth, y_top_synth)
    
    if(index == 1):
        ax.set_title("Synthesized BOLD Signal")

def _plot_voxels(real_set, synth_set, individual=0, voxels=None, y_bottom=None, y_top=None, title_pos=0.999, pad=0.1, normalized=False):
    n_voxels=len(voxels)
    fig = plt.figure(figsize=(20,n_voxels*2))

    fig.suptitle('Top-' + str(len(voxels)) + ' correlated voxels', fontsize=16, y=title_pos)

    if(individual != None):
        real_set = real_set[individual] 
        synth_set = synth_set[individual]

    index=1
    if(voxels):
        for voxel in voxels:
            real_voxel = real_set[voxel]
            synth_voxel = synth_set[voxel]

            if(normalized):
                real_voxel = real_voxel/norm(real_voxel)
                synth_voxel = synth_voxel/norm(synth_voxel)


            _plot_voxel(real_voxel, synth_voxel, 
                        rows=n_voxels, index=index, 
                        y_bottom=y_bottom, y_top=y_top)
            index += 2

    fig.tight_layout(pad=pad)

    plt.show()

def rank_best_synthesized_voxels(real_signal, synth_signal, top_k=10, ignore_static=True, verbose=0):
    sort_voxels = {}
    n_voxels = real_signal.shape[0]
    
    for voxel in range(n_voxels):
        #ignore voxels that are constant over time
        if(ignore_static and all(x==real_signal[voxel][0] for x in real_signal[voxel])):
            continue
        voxel_a = real_signal[voxel].reshape((real_signal[voxel].shape[0]))
        voxel_b = synth_signal[voxel].reshape((synth_signal[voxel].shape[0]))
        distance_cosine = spatial.distance.cosine(voxel_a/norm(voxel_a), voxel_b/norm(voxel_b))
        if(verbose>1):
            print("Distance:", distance_cosine)

        sort_voxels[voxel] = distance_cosine

    sort_voxels = dict(sorted(sort_voxels.items(), key=lambda kv: kv[1]))
    
    if(verbose>0):
        print(list(sort_voxels.values())[0:top_k])

    return list(sort_voxels.keys())[0:top_k]


##########################################################################################################
#
#                                                HEATMAP
#
##########################################################################################################

def heat_map(real_bold_set, synth_bold_set, individual=8, timestep=0, normalize=False):
    real_mapping = np.copy(real_bold_set[individual][:, timestep, 0])
    synth_mapping = np.copy(synth_bold_set[individual][:, timestep, 0])
    
    if(normalize):
        for voxel in range(len(real_bold_set[individual])):
            real_bold_set[individual][voxel] = real_bold_set[individual][voxel]/norm(real_bold_set[individual][voxel])            
            synth_bold_set[individual][voxel] = synth_bold_set[individual][voxel]/norm(synth_bold_set[individual][voxel])
                
    real_mapping.resize((50, 52))
    synth_mapping.resize((50, 52))
    
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,8))
    
    ax1.imshow(real_mapping, cmap='gnuplot2', interpolation='nearest')#, cmap="YlGnBu")
    
    ax1.set_xticks([], [])
    ax1.set_yticks([],[])
    ax1.set_title('Real Bold Signal')
    
    ax2.imshow(synth_mapping, cmap='gnuplot2', interpolation='nearest')#, cmap="YlGnBu")
    
    ax2.set_xticks([], [])
    ax2.set_yticks([],[])
    ax2.set_title('Synthesized Bold Signal')
    
    plt.show()

def plot_epistemic_aleatoric_uncertainty(setting, model, array_set, volume, xslice, yslice, zslice, T=10):
    save_file=uncertainty_plots_directory+setting+"/"+"v_"+ str(volume)+"_x_"+str(xslice)+"_y_"+str(yslice)+"_z_"+str(zslice)+".pdf"

    pathlib.Path(uncertainty_plots_directory+setting).mkdir(parents=True, exist_ok=True) 

    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 4, figsize=(15,6))
    axes[0][0].imshow(rotate(array_set[volume,xslice,:,:,:], 90),cmap=plt.cm.nipy_spectral)
    axes[0][0].set_xticks([])
    axes[0][0].set_yticks([])
    axes[0][1].imshow(rotate(model(array_set[volume:volume+1])[0].numpy()[0,xslice,:,:,:], 90, axes=(0,1)),cmap=plt.cm.nipy_spectral)
    axes[0][1].set_xticks([])
    axes[0][1].set_yticks([])
    axes[0][2].imshow(rotate(bnn_utils.aleatoric_uncertainty(model, array_set[volume:volume+1], T=T).numpy()[0,xslice,:,:,:], 90, axes=(0,1)))
    axes[0][2].set_xticks([])
    axes[0][2].set_yticks([])
    axes[0][3].imshow(rotate(bnn_utils.epistemic_uncertainty(model, array_set[volume:volume+1], T=T).numpy()[0,xslice,:,:,:], 90, axes=(0,1)))
    axes[0][3].set_xticks([])
    axes[0][3].set_yticks([])

    axes[1][0].imshow(rotate(array_set[volume,:,yslice,:,:], 90),cmap=plt.cm.nipy_spectral)
    axes[1][0].set_xticks([])
    axes[1][0].set_yticks([])
    axes[1][1].imshow(rotate(model(array_set[volume:volume+1])[0].numpy()[0,:,yslice,:,:], 90, axes=(0,1)),cmap=plt.cm.nipy_spectral)
    axes[1][1].set_xticks([])
    axes[1][1].set_yticks([])
    axes[1][2].imshow(rotate(bnn_utils.aleatoric_uncertainty(model, array_set[volume:volume+1], T=T).numpy()[0,:,yslice,:,:], 90, axes=(0,1)))
    axes[1][2].set_xticks([])
    axes[1][2].set_yticks([])
    axes[1][3].imshow(rotate(bnn_utils.epistemic_uncertainty(model, array_set[volume:volume+1], T=T).numpy()[0,:,yslice,:,:], 90, axes=(0,1)))
    axes[1][3].set_xticks([])
    axes[1][3].set_yticks([])

    axes[2][0].imshow(array_set[volume,:,:,zslice,:],cmap=plt.cm.nipy_spectral, aspect="auto")
    axes[2][0].set_xticks([])
    axes[2][0].set_yticks([])
    axes[2][1].imshow(model(array_set[volume:volume+1])[0].numpy()[0,:,:,zslice,:],cmap=plt.cm.nipy_spectral, aspect="auto")
    axes[2][1].set_xticks([])
    axes[2][1].set_yticks([])
    axes[2][2].imshow(bnn_utils.aleatoric_uncertainty(model, array_set[volume:volume+1], T=T).numpy()[0,:,:,zslice,:], aspect="auto")
    axes[2][2].set_xticks([])
    axes[2][2].set_yticks([])
    axes[2][3].imshow(bnn_utils.epistemic_uncertainty(model, array_set[volume:volume+1], T=T).numpy()[0,:,:,zslice,:], aspect="auto")
    axes[2][3].set_xticks([])
    axes[2][3].set_yticks([])

    plt.tight_layout()
    plt.savefig(save_file, format="pdf")


def gamma_epoch_plot(setting, parameters_history, epochs=10):
    save_file=gamma_plots_directory+setting+"/"
    pathlib.Path(save_file).mkdir(parents=True, exist_ok=True) 

    plt.style.use("default")

    for epoch in range(epochs):
        plt.figure()
        distribution = gamma.pdf(np.linspace (0, 100, 200), 
                                  a=np.abs(parameters_history[epoch][0])+1e-9, 
                                  scale=1/(np.abs(parameters_history[epoch][1])+1e-9))
        plt.plot(np.linspace (0, 1, 200), distribution)
        plt.title("$\\lambda = \\frac{1}{\\sigma^2} \\sim $Ga($\\alpha$,$\\beta$)", size=12)
        plt.ylabel("$f_{\\lambda} = f_{\\frac{1}{\\sigma^2}}$", size=15)
        plt.rc('font', family='serif')
        plt.tight_layout()
        
        plt.savefig(save_file+"epoch_"+str(epoch)+".pdf", format="pdf")


def uncertainty_losses_plot(setting, losses_history, loss_i, epochs=10):
    save_file=uncertainty_losses_plots_directory+setting+"/"
    pathlib.Path(save_file).mkdir(parents=True, exist_ok=True) 

    plt.style.use("default")

    plt.figure()
    fig, axes = plt.subplots(1,4,figsize=(30,6))

    axes[0].plot(losses_history[:,0], color="black")
    axes[0].grid(True)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("$\mathcal{L}_"+str(loss_i)+"$", size=18)
    axes[0].set_xticks(range(0,epochs))
    axes[0].set_xticklabels(range(1,epochs+1))
    axes[1].plot(losses_history[:,1], color="blue")
    axes[1].grid(True)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("$\mathcal{L}_"+str(2)+"$", size=18)
    axes[1].set_xticks(range(0,epochs))
    axes[1].set_xticklabels(range(1,epochs+1))
    axes[2].plot(losses_history[:,2], color="orange")
    axes[2].grid(True)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("$\mathcal{L}_"+str(1)+"$", size=18)
    axes[2].set_xticks(range(0,epochs))
    axes[2].set_xticklabels(range(1,epochs+1))
    axes[3].plot(losses_history[:,3], color="green")
    axes[3].grid(True)
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("$||y_i-f^{W}(x_i)||_2^2$", size=18)
    axes[3].set_xticks(range(0,epochs))
    axes[3].set_xticklabels(range(1,epochs+1))

    plt.rc('font', family='serif', size=15)
    plt.tight_layout()
    plt.savefig(save_file+"losses_convergence_val.pdf", format="pdf")



##########################################################################################################
#
#                                                3 Dimensinoal plot with 2d slices
#
##########################################################################################################
"""
    
Inputs:
    instance: Numpy.ndarray - of shape (X,Y,Z,1)
    factor: float - that resamples the Z axis slices
    h_resolution: int - resolution in the horizontal dimension
    v_resolution: int - resolution in the vertical dimension
    save: bool - whether to save the figure
    save_path: str - path to save the figure, save has to be True
Returns:
    matplotlib.Figure - The figure to plot, no saving option implemented
"""
def plot_3D_representation_projected_slices(instance, factor=3, h_resolution=1, v_resolution=1, cmap=plt.cm.nipy_spectral, res_img=None, legend_colorbar="redidues", slice_label=True, save=False, save_path=None, save_format="pdf"):
    label = "$Z_{"

    #this is a placeholder for the residues plot
    if(res_img is None):
        res_img=instance
        res_img_none=False
    else:
        res_img_none=True
    
    fig = plt.figure(figsize=(25,7))
    gs = GridSpec(2, 7, figure=fig, wspace=0.01, hspace=0.05)#, wspace=-0.4)

    axes = fig.add_subplot(gs[:,0:2], projection='3d', proj_type='ortho')
    
    cmap = plt.get_cmap(cmap)
    cmap.set_over("w")
    
    x, y = np.mgrid[0:instance[:,:,0].shape[0], 0:instance[:,:,0].shape[1]]
    
    #normalization
    instance = (instance[:,:,:,:]-np.amin(instance[:,:,:,:]))/(np.amax(instance[:,:,:,:])-np.amin(instance[:,:,:,:]))
    res_img = (res_img[:,:,:,:]-np.amin(res_img[:,:,:,:]))/(np.amax(res_img[:,:,:,:])-np.amin(res_img[:,:,:,:]))
    instance[np.where(res_img < 0.37)]= 1.001
    #renormalization
    instance[np.where(instance <=1.0)] = (instance[np.where(instance <=1.0)]-np.amin(instance[np.where(instance <=1.0)]))/(np.amax(instance[np.where(instance <=1.0)])-np.amin(instance[np.where(instance <=1.0)]))

    for axis in range((instance[:,:,:].shape[2])//factor):
        img = rotate(instance[:,:,axis*factor,0], 90)

        ax = axes.plot_surface(x,y,np.ones(x.shape)+5*(axis),
                                facecolors=cmap(img), cmap=cmap, 
                                shade=False, antialiased=True, zorder=0,
                                cstride=v_resolution, rstride=h_resolution)
        if(slice_label):
            axes.text(60, 60, 3+5*(axis), label+str(axis*factor)+"}$", size=13, zorder=100)

    #axes.text(60, 60, 20+5*(instance[:,:,:].shape[2])//factor, "3-Dimensional sliced representation", size=13, zorder=100)
    if(slice_label):
        pos=[0.13, 0.15, 0.01, 0.7]
    elif(not slice_label and res_img_none):
        pos=[0.12, 0.15, 0.01, 0.7]
    else:
        pos=[0.09, 0.15, 0.01, 0.7]
    cbaxes = fig.add_axes(pos)  # This is the position for the colorbar
    cb = plt.colorbar(ax, cax = cbaxes)
    if(not slice_label):
        text_legend = fig.add_axes([pos[0]-0.007]+pos[1:])
        text_legend.axis("off")
        text_legend.text(-0.25,0.45, legend_colorbar, size=17, rotation=90)  # This is the position for the colorbar
        cb.ax.get_yaxis().set_ticks([0,1])
        cb.ax.get_yaxis().set_ticklabels(["Good","Bad"], size=17)


    axes.axis("off")
    axes.view_init(elev=5.5, azim=7)#(100,150)
    axes.dist = 5
    
    #plot each slice in 2 Dimensional plot
    row = 1
    col = 6
    for axis in range((instance[:,:,:].shape[2])//factor):
        axes = fig.add_subplot(gs[row,col])
        
        img = rotate(instance[:,:,axis*factor,0], 90)
        
        axes.imshow(cmap(img))
        if(slice_label):
            axes.text(28, 1, label+str(axis*factor)+"}$", size=13,
                 va="baseline", ha="left", multialignment="left",)

        axes.axis("off")

        col -= 1
        if(col == 1):
            col=6
            row-=1

    plt.rcParams["font.family"] = "Times New Roman"

    if(save):
        fig.savefig(save_path, format=save_format)

    return fig