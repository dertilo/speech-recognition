import torch

import numpy as np
import tensorflow as tf
from tensorflow_addons.image import sparse_image_warp

from data_related.data_augmentation.spec_augment import visualization_spectrogram

if __name__ == "__main__":
    x = np.array(np.arange(0, 200, 1) % 5 == 0, dtype=np.float)
    y = np.array(np.arange(0, 200, 1) % 5 == 0, dtype=np.float)
    xx, yy = np.meshgrid(x, y, sparse=True)
    original = xx + yy
    # spect=torch.from_numpy(X)
    # points = [[y,x] for x in [50] for y in range(0,201,50)]
    # new_points = [[y,x] for x in [60] for y in range(0,201,50)]
    x_coord = np.array([k for k in range(0,x.shape[0],50)])
    delta_x_coord = np.random.triangular(-10,0,10,size=x_coord.shape)
    points = [[y,x] for x in x_coord for y in [0,200]]
    new_points = [[y,x] for x in x_coord + delta_x_coord for y in [0,200]]

    src_points = np.array([points],dtype=np.float)
    dest_points = np.array([new_points],dtype=np.float)
    X = np.reshape(original, (original.shape[0]//2, 2, original.shape[1], 1))

    X_warped,_ = sparse_image_warp(X, src_points, dest_points,num_boundary_points=1,interpolation_order=1)
    o = torch.from_numpy(X_warped.numpy().reshape(original.shape)).squeeze()
    visualization_spectrogram(o, "warped")
