"""
*Preliminary* pytorch implementation.

data generators for voxelmorph
"""

import numpy as np
import sys
import cv2
import torch

def load_example_by_name(vol_name, seg_name=None):
    """
    load a specific volume and segmentation
    """
    X = np.load(vol_name)['vol_data']
    X = np.reshape(X, (1,) + X.shape + (1,))

    return_vals = [X]

    if(seg_name):
        X_seg = np.load(seg_name)['vol_data']
        X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
        return_vals.append(X_seg)

    return tuple(return_vals)

def load_volfile(datafile):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), assume variable names 'vol_data'
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % datafile

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        if 'nib' not in sys.modules:
            try:
                import nibabel as nib
            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')

        X = nib.load(datafile).get_data()

    else:  # npz
        X = np.load(datafile)['vol_data']

    return X

def example_gen(vol_names, batch_size=1, return_segs=False, seg_dir=None):
    """
    generate examples

    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)

        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
    """

    while True:
        idxes = np.random.randint(len(vol_names), size=batch_size)

        X_data = []
        for idx in idxes:
            X = load_volfile(vol_names[idx])
            X = X[np.newaxis, ..., np.newaxis]
            X_data.append(X)

        if batch_size > 1:
            return_vals = [np.concatenate(X_data, 0)]
        else:
            return_vals = [X_data[0]]

        # also return segmentations
        if return_segs:
            X_data = []
            for idx in idxes:
                v = vol_names[idx].replace('norm', 'aseg')
                v = v.replace('vols', 'asegs')
                X_seg = load_volfile(v)
                X_seg = X_seg[np.newaxis, ..., np.newaxis]
                X_data.append(X_seg)

            if batch_size > 1:
                return_vals.append(np.concatenate(X_data, 0))
            else:
                return_vals.append(X_data[0])

        yield tuple(return_vals)

def example_gen_mzl(vol_names, batch_size=1):
    """
    generate examples

    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)

        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
    """

    while True:
        idxes = np.random.randint(len(vol_names), size=batch_size)

        vol_shape = (128, 256, 256)

        X_data = []
        for idx in idxes:
            X = load_volfile(vol_names[idx])

            # Preprocess -> 104x256x256
            zfill = np.zeros(vol_shape)
            diff = vol_shape[0] - X.shape[0]
            start =  int(diff/2)
            end = start + X.shape[0]
            i = 0
            # Zero Fill
            for ii in range(start,end):
                # Reshape if necessary
                try:
                    if X.shape[1] != X.shape[2] or X.shape[2] != vol_shape[1]:
                        zfill[ii] = cv2.resize(X[i,:,:],(vol_shape[1],vol_shape[2]), interpolation=cv2.INTER_CUBIC)
                    else:
                        zfill[ii]= X[i,:,:]
                    i = i+1
                except:
                    print('error with zerofill')
            X = zfill

            # Get dimensions right
            X = X[np.newaxis, ..., np.newaxis]
            X_data.append(X)

        if batch_size > 1:
            return_vals = [np.concatenate(X_data, 0)]
        else:
            return_vals = [X_data[0]]

        yield tuple(return_vals)

def paired_gen(fixed_vols, batch_size=1):

    while True:
        idxes = np.random.randint(len(fixed_vols), size=batch_size)

        vol_shape = (128, 256, 256)

        X_data, Y_data = [],[]
        for idx in idxes:
            fn_x = fixed_vols[idx]
            X = load_volfile(fn_x)
            fn_y= fn_x[:-8] + '2.nii.gz'
            Y = load_volfile(fn_y)


            # Preprocess -> 104x256x256
            zfill_X = np.zeros(vol_shape)
            zfill_Y = np.zeros(vol_shape)

            diff = vol_shape[0] - X.shape[0]
            start =  int(diff/2)
            end = start + X.shape[0]
            i = 0
            # Zero Fill
            for ii in range(start,end):
                # Reshape if necessary
                try:
                    if X.shape[1] != X.shape[2] or X.shape[2] != vol_shape[1]:
                        zfill_X[ii] = cv2.resize(X[i, :, :], (vol_shape[1], vol_shape[2]), interpolation=cv2.INTER_CUBIC)
                        zfill_Y[ii] = cv2.resize(Y[i, :, :], (vol_shape[1], vol_shape[2]), interpolation=cv2.INTER_CUBIC)
                    else:
                        zfill_X[ii] = X[i, :, :]
                        zfill_Y[ii] = Y[i, :, :]
                    i = i+1
                except:
                    print('error with zerofill')
            X = zfill_X
            Y = zfill_Y

            # Get dimensions right
            X = X[np.newaxis, ..., np.newaxis]
            Y = Y[np.newaxis, ..., np.newaxis]

            X_data.append(X)
            Y_data.append(Y)

        if batch_size > 1:
            return_vals_X = [np.concatenate(X_data, 0)]
            return_vals_Y = [np.concatenate(Y_data, 0)]
        else:
            return_vals_X = [X_data[0]]
            return_vals_Y = [Y_data[0]]

        yield tuple(return_vals_X), tuple(return_vals_Y), fn_x, fn_y