import numpy as np
import train_paired as train
import os

gpu = '0'
data_dir = "/data1/mike/breast/all/orient/"
lr = 1e-4
n_iter = 150000
data_loss = 'mmi'
model = 'vm2'
reg_param = 0.01
batch_size = 1
n_save_iter = 500
model_dir = "/data/mike/vm/paired_mmi"
atlas_file=None
if not os.path.exists(model_dir):
      os.mkdir(model_dir,mode=0o775)

train.train(gpu,
      data_dir,
      lr,
      n_iter,
      data_loss,
      model,
      reg_param,
      batch_size,
      n_save_iter,
      model_dir)

