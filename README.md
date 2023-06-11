# <p align="center">  MindDiffuser  </p> 
This is the official code for the paper "MindDiffuser: Controlled Image Reconstruction from Human Brain Activity with Semantic and Structural Diffusion"<br>

![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/overview.png)<br>

    Schematic diagram of MindDiffuser. 
    (a) Decoders are trained to fit fMRI with averaged CLIP text embeddings 𝑐, CLIP image feature 𝑍𝑖𝐶𝐿𝐼𝑃, and VQ-VAE latent feature 𝑧.
    (b) The two-stage image reconstruction process. In stage 1, an initial reconstructed image is generated using the decoded CLIP text feature 𝑐 and VQ-VAE latent feature 𝑧. In stage 2, the decoded CLIP image feature is used as a constraint to iteratively adjust 𝑐 and 𝑧 until the final reconstruction result matches the original image in terms of both semantic and structure.
![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/plane_00.png)<br>

    A brief comparison of image reconstruction results.
![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/four_sub_00.png)<br>

    Reconstruction results of MindDiffuser on multiple subjects
![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/cortex_sub2_00.png)<br>

    During the feature decoding process, we use L2-regularized linear regression model to automatically select voxels to fit three types
    of feature: semantic feature 𝑐, detail feature 𝑧, and structural feature 𝑍𝐶𝐿𝐼𝑃. We ultilize pycortex to project the weights of each 
    voxel in the fitted model onto the corresponding 3D coordinates in the visual cortex.

# <p align="center">  Preliminaries  </p> 
This code was developed and tested with:

*  Python version 3.8.5
*  PyTorch version 1.11.0
*  A100 40G
*  The conda environment defined in environment.yml

# <p align="center">  Dataset  </p> 
    `NSD dataset` http://naturalscenesdataset.org/
    `Data preparation` https://github.com/styvesg/nsd

# <p align="center">  Experiments  </p> 
![](https://github.com/ReedOnePeck/MindDiffuser/blob/main/Images/1686488621334.png)

## <p> MindDiffuser </p>
### <p> Model preparation  </p>
First, set up the conda enviroment as follows:<br>

    conda env create -f environment.yml  # create conda env
    conda activate MindDiffuser          # activate conda env

### <p> Feature extraction </p>
### <p> Feature decoding </p>
### <p> Image reconstruction </p>

## <p> Reproduce the results of "High-resolution image reconstruction with latent diffusion models from human brain activity"(CVPR2023)  </p>

## <p> Reproduce the results of "Reconstruction of Perceived Images from fMRI Patterns and Semantic Brain Exploration using Instance-Conditioned GANs" </p>






