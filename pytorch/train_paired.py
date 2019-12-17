"""
*Preliminary* pytorch implementation.

VoxelMorph training.
"""


# python imports
import os
import glob
import random
import warnings
from argparse import ArgumentParser

# external imports
import numpy as np
import torch
from torch.optim import Adam
import nibabel as nib

# internal imports
from model import cvpr2018_net
import datagenerators
import losses

def train(gpu,
          data_dir,
          lr,
          n_iter,
          data_loss,
          model,
          reg_param, 
          batch_size,
          n_save_iter,
          model_dir,
          vol_size=(128, 256, 256)):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable, 'conditional' creates average
    :param lr: learning rate
    :param n_iter: number of training iterations
    :param data_loss: data_loss: 'mse' or 'ncc
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param n_save_iter: Optional, default of 500. Determines how many epochs before saving model version.
    :param model_dir: the model directory to save to
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    #### Data and Dataloaders

    # Training Data Names
    fixed_fns = glob.glob(os.path.join(data_dir, '1*1.nii.gz'))
    fixed_fns = fixed_fns + glob.glob(os.path.join(data_dir,'1*3.nii.gz'))

    # data generator
    train_example_gen = datagenerators.paired_gen(fixed_fns, batch_size)

    #### Model
    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == "vm2":
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    else:
        raise ValueError("Not yet implemented!")

    model = cvpr2018_net(vol_size, nf_enc, nf_dec)
    model.to(device)

    #### Optimizer
    opt = Adam(model.parameters(), lr=lr)

    #### Losses
    if data_loss == "ncc":
        sim_loss_fn = losses.ncc_loss
    elif data_loss == "mmi":
        sim_loss_fn = losses.mmi
    elif data_loss == "jhmi":
        sim_loss_fn = losses.jhmi
    elif data_loss == "demons":
        sim_loss_fn = losses.demons
    else:
        sim_loss_fn = losses.mse_loss

    # L1 Penalization for flow field
    grad_loss_fn = losses.gradient_loss

    # Training loop.
    for i in range(n_iter):

        # Save model checkpoint
        if i % n_save_iter == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % i)
            torch.save(model.state_dict(), save_file_name)

        # Generate the moving images and convert them to tensors.
        fixed_image, moving_image = next(train_example_gen)

        input_fixed = torch.from_numpy(fixed_image[0]).to(device).float()
        input_fixed = input_fixed.permute(0, 4, 1, 2, 3)

        input_moving = torch.from_numpy(moving_image[0]).to(device).float()
        input_moving = input_moving.permute(0, 4, 1, 2, 3)

        # Run the data through the model to produce warp and flow field
        input_warped, flow_field = model(input_moving, input_fixed)

        # Calculate loss
        recon_loss = sim_loss_fn(input_warped, input_fixed)
        grad_loss = grad_loss_fn(flow_field)
        loss = recon_loss + reg_param * grad_loss

        print('%d loss[tot: %6.2f - recon: %6.2f - L1: %6.6f]' % (i, loss.item(), recon_loss.item(), grad_loss.item()))

        # Backwards and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = ArgumentParser()

    parser.add_argument("--gpu",
                        type=str,
                        default='0',
                        help="gpu id")

    parser.add_argument("--data_dir",
                        type=str,
                        help="data folder with training vols")

    parser.add_argument("--atlas_file",
                        type=str,
                        dest="atlas_file",
                        default='../data/atlas_norm.npz',
                        help="gpu id number")

    parser.add_argument("--lr",
                        type=float,
                        dest="lr",
                        default=1e-4,
                        help="learning rate")

    parser.add_argument("--n_iter",
                        type=int,
                        dest="n_iter",
                        default=150000,
                        help="number of iterations")

    parser.add_argument("--data_loss",
                        type=str,
                        dest="data_loss",
                        default='ncc',
                        help="data_loss: mse of ncc")

    parser.add_argument("--model",
                        type=str,
                        dest="model",
                        choices=['vm1', 'vm2'],
                        default='vm2',
                        help="voxelmorph 1 or 2")

    parser.add_argument("--lambda", 
                        type=float,
                        dest="reg_param", 
                        default=0.01,  # recommend 1.0 for ncc, 0.01 for mse
                        help="regularization parameter")

    parser.add_argument("--batch_size", 
                        type=int,
                        dest="batch_size", 
                        default=1,
                        help="batch_size")

    parser.add_argument("--n_save_iter", 
                        type=int,
                        dest="n_save_iter", 
                        default=500,
                        help="frequency of model saves")

    parser.add_argument("--model_dir", 
                        type=str,
                        dest="model_dir", 
                        default='./models/',
                        help="models folder")


    train(**vars(parser.parse_args()))

