"""
*Preliminary* pytorch implementation.

VoxelMorph training.
"""


# python imports
from glob import glob
import os
import warnings
from argparse import ArgumentParser

# external imports
import torch
from torch.optim import Adam

# internal imports
from model import cvpr2018_net
import datagenerators
import losses

def train(gpu,
          data_fns,
          lr,
          n_iter,
          data_loss,
          model,
          reg_param, 
          batch_size,
          n_save_iter,
          model_dir,
          resume,
          vol_size=(128, 256, 256)):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param data_fns: image filenames
    :param lr: learning rate
    :param n_iter: number of training iterations
    :param data_loss: data_loss: 'mse', 'ncc', 'mmi','jhmi','demons'
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param n_save_iter: Optional, default of 500. Frequency of model save.
    :param model_dir: the model directory to save to
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    #### Data and Dataloaders
    # data generator
    train_example_gen = datagenerators.paired_gen(data_fns, batch_size)

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

    # Resume
    start=0
    if resume==True:
        modfn = sorted(glob(os.path.join(model_dir,'*.ckpt')))[-1]
        model.load_state_dict(torch.load(modfn))
        start=int(os.path.basename(modfn).split('.')[0])

    #### Optimizer
    opt = Adam(model.parameters(), lr=lr)

    #### Losses
    if data_loss == "ncc":
        sim_loss_fn = losses.ncc
    elif data_loss == "mmi":
        sim_loss_fn = losses.mmi
    elif data_loss == "jhmi":
        sim_loss_fn = losses.jhmi
    elif data_loss == "demons":
        sim_loss_fn = losses.demons
    else:
        sim_loss_fn = losses.mse_loss

    # L2 Penalization for flow field
    grad_loss_fn = losses.gradient_loss

    # Training loop.
    for i in range(start,n_iter):

        # Save model checkpoint
        if i % n_save_iter == 0:
            save_file_name = os.path.join(model_dir, str(i).zfill(7) + '.ckpt')
            torch.save(model.state_dict(), save_file_name)

        # Generate the moving images and convert them to tensors.
        fixed_image, moving_image, fix_fn, mov_fn = next(train_example_gen)

        # To GPU
        input_fixed = torch.from_numpy(fixed_image[0]).to(device).float()
        input_moving = torch.from_numpy(moving_image[0]).to(device).float()

        # Shapeshifting
        input_fixed = input_fixed.permute(0, 4, 1, 2, 3)
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

    parser.add_argument("--gpu", type=str,
                        default='0', help="gpu id")

    parser.add_argument("--data_fns", type=str,
                        help="filenames of training vols")

    parser.add_argument("--lr", type=float,
                        default=1e-4, help="learning rate")

    parser.add_argument("--n_iter", type=int,
                        default=150000, help="number of iterations")

    parser.add_argument("--data_loss", type=str,
                        default='ncc', help="data_loss: mse of ncc")

    parser.add_argument("--model", type=str,
                        default='vm2', help="voxelmorph 1(vm1) or 2(vm2)")

    parser.add_argument("--reg_param",  type=float,
                        default=0.01, help="regularization parameter e.g. 1.0 (ncc), 0.01 (mse)")

    parser.add_argument("--batch_size",  type=int,
                        default=1, help="batch_size")

    parser.add_argument("--n_save_iter",  type=int,
                        default=500, help="frequency of model saves")

    parser.add_argument("--model_dir",  type=str,
                        default='./models/',help="models folder")

    parser.add_argument("--resume", type=bool,
                        default=False, help="resume")

    train(**vars(parser.parse_args()))