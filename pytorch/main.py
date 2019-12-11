import numpy as np
import train


gpu = '1'
data_dir = "D:\\"
lr = 1e-4
n_iter = 150000
data_loss = 'ncc'
model = 'vm2'
reg_param = 0.01
batch_size = 1
n_save_iter = 500
model_dir = "D:\\models\\"
atlas_file = None

train.train(gpu,
      data_dir,
      atlas_file,
      lr,
      n_iter,
      data_loss,
      model,
      reg_param,
      batch_size,
      n_save_iter,
      model_dir)