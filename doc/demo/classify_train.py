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
from xlearn.classify import train, model
from keras.utils import np_utils
import keras.callbacks

# ================================================================
np.random.seed(1337)

dim_img = 128
patch_size = (dim_img, dim_img)
batch_size = 50
file_batch_size = 10 # set this according to RAM
nb_classes = 2
nb_epoch = 8
window=((700, 700), (1300, 1300))

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

try:
    os.mkdir('checkpoints')
except:
    pass

raw_shape = dxchange.read_tiff(glob.glob(os.path.join(good_folder, '*.tiff'))[0]).shape

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

mdl = model(dim_img, nb_filters, nb_conv, nb_classes)

i_batch = 0


for good_set, bad_set in zip(good_chunks, bad_chunks):

    print('Batch {}'.format(i_batch))
    checkpoint = keras.callbacks.ModelCheckpoint('checkpoints/batch_{}.hdf5'.format(i_batch))

    if good_set is not None:
        good_data = np.zeros([len(good_set), raw_shape[0], raw_shape[1]])
        for i, fname in enumerate(good_set):
            good_data[i] = dxchange.read_tiff(fname)
    else:
        good_data = None
    if bad_set is not None:
        bad_data = np.zeros([len(bad_set), raw_shape[0], raw_shape[1]])
        for i, fname in enumerate(bad_set):
            bad_data[i] = dxchange.read_tiff(fname)
    else:
        bad_data = None

    if bad_data is not None:
        # bad_data = nor_data(bad_data)
        print ('bad_data raw shape: {}; mean: {}'.format(bad_data.shape, bad_data.mean()))
        bad_data = img_window(bad_data[:, window[0][0]:window[1][0], window[0][1]:window[1][1]], 200, reject_bg=True, threshold=1e-4)
        print ('bad_data windows shape: {}; mean: {}: '.format(bad_data.shape, bad_data.mean()))
        uncenter_patches = extract_3d(bad_data, patch_size, 3)
        print('bad_data extracted patches shape: {}; mean: {}: '.format(uncenter_patches.shape, uncenter_patches.mean()))
        uncenter_patches = nor_data(uncenter_patches)
        np.random.shuffle(uncenter_patches)

    if good_data is not None:
        # good_data = nor_data(good_data)
        print ('good_data raw shape: {}; mean: {}'.format(good_data.shape, good_data.mean()))
        good_data = img_window(good_data[:, window[0][0]:window[1][0], window[0][1]:window[1][1]], 400, reject_bg=True, threshold=1e-4)
        print ('good_data windows shape: {}; mean: {}: '.format(good_data.shape, good_data.mean()))
        center_patches = extract_3d(good_data, patch_size, 4)
        print ('good_data extracted patches shape: {}; mean: {}: '.format(center_patches.shape, center_patches.mean()))
        center_patches = nor_data(center_patches)
        np.random.shuffle(center_patches)

    divider_bad = int(uncenter_patches.shape[0] * 0.7)
    divider_good = int(center_patches.shape[0] * 0.7)
    x_train = np.concatenate((uncenter_patches[0:divider_bad], center_patches[0:divider_good]), axis=0)
    x_test = np.concatenate((uncenter_patches[divider_bad:], center_patches[divider_bad:]), axis=0)
    x_train = x_train.reshape(x_train.shape[0], 1, dim_img, dim_img)
    x_test = x_test.reshape(x_test.shape[0], 1, dim_img, dim_img)
    y_train = np.zeros(len(x_train))
    y_train[divider_bad:] = 1
    y_test = np.zeros(len(x_test))
    y_test[uncenter_patches.shape[0]-divider_bad:] = 1

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    try:
        mdl.load_weights('weight_center.h5')
    except IOError:
        print('No weights to load. Starting from empty model.')
    mdl.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch,
            verbose=1, validation_data=(x_test, y_test), callbacks=[checkpoint])
    mdl.save_weights('weight_center.h5')
    score = mdl.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    i_batch += 1