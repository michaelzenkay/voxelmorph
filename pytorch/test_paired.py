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

class html_output():
    def __init__(self,fn):
        self.html = '<html><head></head><body>'
        self.fn = fn
    def img(self,img,align='left', height="256", width="256"):
        self.html = self.html+ "<img src=" + img + " align= " + align + ' width="' + width + '" height="' + height + '">'
    def txt(self,intxt):
        self.html= self.html+ intxt+ '<br>'
        print(intxt)
    def br(self):
        self.html = self.html+ '<br>'
    def finish(self):
        self.html= self.html + '</body></html>'
        with open(self.fn, 'w') as fd:
            fd.write(self.html)

def test(gpu,
         moving_fns,
         model, 
         modelfn,
         outdir,
         vol_size=(128,256,256),
         batch_size = 1,
         ants_dir="/data/mike/breast/all/regorient/"):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param init_model_file: the model directory to load from
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    id=basename(modelfn)[:-5]
    html = html_output(join(outdir,id + 'results.html'))
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

    train_example_gen = datagenerators.test_gen(moving_fns)

    # Use this to warp segments
    trf = SpatialTransformer(vol_size, mode='nearest')
    trf.to(device)

    loss_vm = np.zeros(len(moving_fns))
    loss_og = np.zeros(len(moving_fns))
    loss_an = np.zeros(len(moving_fns))

    for i in range(0,len(moving_fns)):
        # Get Images
        fix_img, mov_img, ant_img, fix_fn, mov_fn, ant_fn = next(train_example_gen)

        # To GPU
        fix_gpu = torch.from_numpy(fix_img[0]).to(device).float()
        mov_gpu = torch.from_numpy(mov_img[0]).to(device).float()
        ant_gpu = torch.from_numpy(ant_img[0]).to(device).float()

        # Shapeshifting
        fix_gpu = fix_gpu.permute(0, 4, 1, 2, 3)
        mov_gpu = mov_gpu.permute(0, 4, 1, 2, 3)
        ant_gpu = ant_gpu.permute(0, 4, 1, 2, 3)

        # Forward Pass
        warped, flow = model(mov_gpu, fix_gpu)

        # Compare
        loss_vm[i] = losses.mmi(warped,fix_gpu).cpu().detach().numpy()
        loss_og[i] = losses.mmi(mov_gpu,fix_gpu).cpu().detach().numpy()
        if not np.mean(ant_img)==0:
            loss_an[i] = losses.mmi(fix_gpu,ant_gpu).cpu().detach().numpy()
        else:
            loss_an[i] = 0

        # Save
        warped_fn = join(outdir, basename(mov_fn))
        fixed_fn1 = join(outdir, basename(fix_fn))
        #if not os.path.exists(fixed_fn1):
        #    export2nii(fix_gpu, fixed_fn1)
        #export2nii(warped, warped_fn)

        diff = loss_vm[i] - loss_an[i]
        andiff = loss_og[i] - loss_an[i]

        if diff<0:
           winner = 'vm'
        else:
           winner = 'an'
        # Log and display
        html.txt('diff '+ str(diff).zfill(3) + winner)

    # Average Results
    html.txt('AVG [VM '+ str(np.mean(loss_vm)) + ' OG' + str(np.mean(loss_og)) + ' AN ' + str(np.mean(loss_an)) + ']')
    np.save(join(outdir, id+'_loss_vm.npy'), loss_vm)
    np.save(join(outdir, id+'_loss_og.npy'), loss_og)
    np.save(join(outdir, id+'_loss_an.npy'), loss_an)

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