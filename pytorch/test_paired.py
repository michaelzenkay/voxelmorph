"""
*Preliminary* pytorch implementation.

VoxelMorph testing
"""


# python imports
import os
from os.path import join, basename, dirname
import glob
import random
import sys
from argparse import ArgumentParser
import nibabel as nib

import numpy as np
import torch
from model import cvpr2018_net, SpatialTransformer
import datagenerators
from datagenerators import load_volfile
import scipy.io as sio
import losses

def export2nii(vol,fn):
    try:
        vol=vol.cpu().detach().numpy()
    except:
        None
    vol = np.squeeze(vol)
    if len(vol.shape)!=5:
        print('Can not handle this input image')
        return
    nii = nib.Nifti1Image(vol,np.eye(4))
    nib.save(nii,fn)
    print('saved to ' +fn)

def test(gpu,
         moving_fns,
         model, 
         modelfn,
         outdir,
         vol_size=(128,256,256),
         batch_size = 1):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param init_model_file: the model directory to load from
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == "vm2":
        nf_dec = [32, 32, 32, 32, 32, 16, 16]

    # Set up model
    model = cvpr2018_net(vol_size, nf_enc, nf_dec)
    model.to(device)
    model.load_state_dict(torch.load(modelfn, map_location=lambda storage, loc: storage))

    train_example_gen = datagenerators.paired_gen(moving_fns, batch_size)

    # Use this to warp segments
    trf = SpatialTransformer(vol_size, mode='nearest')
    trf.to(device)

    for i in range(0,len(moving_fns)):
        # Get Images
        fixed_image, moving_image , fixed_fn, moving_fn= next(train_example_gen)

        # To GPU
        fixed_gpu = torch.from_numpy(fixed_image[0]).to(device).float()
        moving_gpu = torch.from_numpy(moving_image[0]).to(device).float()

        # Shapeshifting
        fixed_gpu = fixed_gpu.permute(0, 4, 1, 2, 3)
        moving_gpu = moving_gpu.permute(0, 4, 1, 2, 3)

        # Forward Pass
        warped, flow = model(moving_gpu, fixed_gpu)

        # Compare
        og_loss = losses.mmi(warped,fixed_gpu).cpu().detach().numpy()
        warp_loss = losses.mmi(moving_gpu,fixed_gpu).cpu().detach().numpy()
        diff = og_loss-warp_loss

        warped_fn = join(outdir,basename(moving_fn))
        fixed_fn1 = join(outdir, basename(fixed_fn))
        if not os.path.exists(fixed_fn1):
            export2nii(fixed_gpu, fixed_fn1)
        export2nii(warped, warped_fn)

        print('OG Loss '+ str(og_loss) + ' : Warp Loss ' + str(warp_loss) + ' : diff ' + str(diff))
 

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--gpu",
                        type=str,
                        help="gpu id")

    parser.add_argument("--model",
                        type=str,
                        dest="model",
                        choices=['vm1', 'vm2'],
                        default='vm2',
                        help="voxelmorph 1 or 2")

    parser.add_argument("--modelfn",
                        type=str,
                        dest="modelfn",
                        help="model weight file")

    test(**vars(parser.parse_args()))

