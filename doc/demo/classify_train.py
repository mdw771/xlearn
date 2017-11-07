#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script 
"""
# To run this example please download the test data from the classify_train folder at 
# http://tinyurl.com/APS-xlearn

from __future__ import print_function

import os
from six.moves import zip
import glob

import dxchange
import numpy as np
from xlearn.utils import *
from xlearn.classify import train

# ================================================================
np.random.seed(1337)

dim_img = 128
patch_size = (dim_img, dim_img)
batch_size = 50
file_batch_size = 10 # set this according to RAM
nb_classes = 2
nb_epoch = 12
window=((800, 800), (1600, 1600))

good_ind_range = 'all' # tuple or 'all'
bad_ind_range = 'all' # tuple or 'all'

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
# ================================================================

good_folder = os.path.join('training_set', 'good')
bad_folder = os.path.join('training_set', 'bad')

good_filelist = glob.glob(os.path.join(good_folder, '*.tiff'))
good_filelist.sort()
bad_filelist = glob.glob(os.path.join(bad_folder, '*.tiff'))
bad_filelist.sort()

good_chunks = divide_chunks(good_filelist, chunk_size=file_batch_size)
bad_chunks = divide_chunks(bad_filelist, chunk_size=file_batch_size)
if len(good_chunks) > len(bad_chunks):
    bad_chunks.append([None] * (len(good_chunks) - len(bad_chunks)))
else:
    good_chunks.append([None] * (len(bad_chunks) - len(good_chunks)))

for good_inds, bad_inds in zip(good_chunks, bad_chunks):
    uncenter1 = dxchange.read_tiff_stack(os.path.join(bad_folder, '00000.tiff'), ind=ind_uncenter1, digit=4)
    uncenter2 = dxchange.read_tiff_stack(fname, ind=ind_uncenter2, digit=4)
    uncenter = np.concatenate((uncenter1, uncenter2), axis=0)
    uncenter = nor_data(uncenter)
    print (uncenter.shape)
    uncenter = img_window(uncenter[:, 360:1460, 440:1440], 200)
    print (uncenter.shape)
    uncenter_patches = extract_3d(uncenter, patch_size, 10)
    print(uncenter_patches.shape)
    np.random.shuffle(uncenter_patches)
    print ('uncenter_patches', uncenter_patches.shape)
    # print uncenter_patches.shape
    center_img = dxchange.read_tiff('../../test/test_data/1048.tiff')
    center_img = nor_data(center_img)
    print (center_img.shape)
    center_img = img_window(center_img[360:1460, 440:1440], 400)
    center_patches = extract_3d(center_img, patch_size, 1)
    np.random.shuffle(center_patches)
    print ('center_patches', center_patches.shape)
    # plt.imshow(center_img, cmap='gray', interpolation= None)
    # plt.show()

    x_train = np.concatenate((uncenter_patches[0:500], center_patches[0:50000]), axis=0)
    x_test = np.concatenate((uncenter_patches[500:1000], center_patches[50000:60000]), axis=0)
    x_train = x_train.reshape(x_train.shape[0], 1, dim_img, dim_img)
    x_test = x_test.reshape(x_test.shape[0], 1, dim_img, dim_img)
    y_train = np.zeros(50500)
    y_train[500:] = 1
    y_test = np.zeros(10500)
    y_test[500:] = 1

    model = train(x_train, y_train, x_test, y_test, dim_img, nb_filters, nb_conv, batch_size, nb_epoch, nb_classes)
    model.save_weights('training_checkpoint.h5')
