import os
from os.path import join, basename, dirname
from argparse import ArgumentParser
import nibabel as nib
import numpy as np
import torch
from model import cvpr2018_net, SpatialTransformer
import datagenerators
from datagenerators import load_volfile
import scipy.io as sio
import losses

def export2nii(vol,fn, verbose=False):
    try:
        vol=vol.cpu().detach().numpy()
    except:
        None
    vol = np.squeeze(vol)
    if len(vol.shape)>3:
        print('Can not handle this input image')
        return
    nii = nib.Nifti1Image(vol,np.eye(4))
    nib.save(nii,fn)
    if verbose==True:
        print('saved to ' +fn)

def register(moving_fns, modelfn, outdir, vol_size=(128,256,256), batch_size = 1, gpu = '0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    # Set up model
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 32, 16, 16]
    model = cvpr2018_net(vol_size, nf_enc, nf_dec)
    model.to(device)
    model.load_state_dict(torch.load(modelfn, map_location=lambda storage, loc: storage))

    # Data Generator
    datagen = datagenerators.test_gen(moving_fns)

    # Use this to warp segments
    trf = SpatialTransformer(vol_size, mode='nearest')
    trf.to(device)

    for i in range(0,len(moving_fns)):
        # Get Images
        fix_img, mov_img, ant_img, fix_fn, mov_fn, ant_fn = next(datagen)

        # To GPU
        fix_gpu = torch.from_numpy(fix_img[0]).to(device).float()
        mov_gpu = torch.from_numpy(mov_img[0]).to(device).float()

        # Shapeshifting
        fix_gpu = fix_gpu.permute(0, 4, 1, 2, 3)
        mov_gpu = mov_gpu.permute(0, 4, 1, 2, 3)

        # Forward Pass
        warped, flow = model(mov_gpu, fix_gpu)

        # Save
        warped_fn = join(outdir, basename(mov_fn))
        fixed_fn1 = join(outdir, basename(fix_fn))
        if not os.path.exists(fixed_fn1):
           export2nii(fix_gpu, fixed_fn1)
        export2nii(warped, warped_fn)