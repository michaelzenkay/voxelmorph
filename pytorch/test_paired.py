"""
*Preliminary* pytorch implementation.

VoxelMorph testing
"""


# python imports
import os
import glob
import random
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from model import cvpr2018_net, SpatialTransformer
import datagenerators
from datagenerators import load_volfile
import scipy.io as sio
import losses


def test(gpu,
         moving_fns,
         model, 
         modelfn,
         vol_size=(128,256,256)):
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

    batch_size=1
    train_example_gen = datagenerators.paired_gen(moving_fns, batch_size)


    # Use this to warp segments
    trf = SpatialTransformer(vol_size, mode='nearest')
    trf.to(device)

    for i in range(0,len(moving_fns)):
        # Get Images
        fixed_image, moving_image = next(train_example_gen)

        # To GPU
        input_fixed = torch.from_numpy(fixed_image[0]).to(device).float()
        input_moving = torch.from_numpy(moving_image[0]).to(device).float()

        # Shapeshifting
        input_fixed = input_fixed.permute(0, 4, 1, 2, 3)
        input_moving = input_moving.permute(0, 4, 1, 2, 3)

        # Forward Pass
        warped, flow = model(input_moving, input_fixed)

        # Compare
        og_loss = losses.mmi(warped,input_fixed).cpu().detach().numpy()
        warp_loss = losses.mmi(input_moving,input_fixed).cpu().detach().numpy()
        diff = og_loss-warp_loss

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

