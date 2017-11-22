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
file_batch_size_good = 1  # set this according to RAM
file_batch_size_bad = 36  # set this according to RAM
nb_classes = 3
nb_epoch = 4
window=((600, 600), (1300, 1300))
save_intermediate = True

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
bad_low_folder = os.path.join('training_set', 'bad_low')
bad_high_folder = os.path.join('training_set', 'bad_high')

try:
    os.mkdir('checkpoints')
except:
    pass

raw_shape = dxchange.read_tiff(glob.glob(os.path.join(good_folder, '*.tiff'))[0]).shape

good_filelist = glob.glob(os.path.join(good_folder, '*.tiff'))
good_filelist.sort()
bad_low_filelist = glob.glob(os.path.join(bad_low_folder, '*.tiff'))
bad_low_filelist.sort()
bad_high_filelist = glob.glob(os.path.join(bad_high_folder, '*.tiff'))
bad_high_filelist.sort()

good_chunks = divide_chunks(good_filelist, chunk_size=file_batch_size_good)
bad_low_chunks = divide_chunks(bad_low_filelist, chunk_size=file_batch_size_bad)
bad_high_chunks = divide_chunks(bad_high_filelist, chunk_size=file_batch_size_bad)
max_len = max([len(good_chunks), len(bad_low_chunks), len(bad_high_chunks)])
if len(bad_low_chunks) < max_len:
    bad_low_chunks.append([None] * (max_len - len(bad_low_chunks)))
if len(bad_high_chunks) < max_len:
    bad_high_chunks.append([None] * (max_len - len(bad_high_chunks)))
if len(good_chunks) < max_len:
    good_chunks.append([None] * (max_len - len(good_chunks)))

mdl = model(dim_img, nb_filters, nb_conv, nb_classes)

i_batch = 0


for good_set, bad_low_set, bad_high_set in zip(good_chunks, bad_low_chunks, bad_high_chunks):

    print('Batch {}'.format(i_batch))
    checkpoint = keras.callbacks.ModelCheckpoint('checkpoints/batch_{}.hdf5'.format(i_batch))

    if good_set is not None:
        good_data = np.zeros([len(good_set), raw_shape[0], raw_shape[1]])
        for i, fname in enumerate(good_set):
            good_data[i] = dxchange.read_tiff(fname)
    else:
        good_data = None

    if bad_low_set is not None:
        bad_low_data = np.zeros([len(bad_low_set), raw_shape[0], raw_shape[1]])
        for i, fname in enumerate(bad_low_set):
            bad_low_data[i] = dxchange.read_tiff(fname)
    else:
        bad_low_data = None

    if bad_high_set is not None:
        bad_high_data = np.zeros([len(bad_high_set), raw_shape[0], raw_shape[1]])
        for i, fname in enumerate(bad_high_set):
            bad_high_data[i] = dxchange.read_tiff(fname)
    else:
        bad_high_data = None

    if bad_low_data is not None:
        # bad_data = nor_data(bad_data)
        print ('bad_low_data raw shape: {}; mean: {}'.format(bad_low_data.shape, bad_low_data.mean()))
        bad_low_data = img_window(bad_low_data[:, window[0][0]:window[1][0], window[0][1]:window[1][1]], 200, reject_bg=True, threshold=1.2e-4)
        bad_low_data = convolve_stack(bad_low_data, get_gradient_kernel())
        print ('bad_low_data windows shape: {}; mean: {}: '.format(bad_low_data.shape, bad_low_data.mean()))
        bad_low_data = extract_3d(bad_low_data, patch_size, 6, reject_bj=False, threshold=1.2e-4)
        print('bad_low_data extracted patches shape: {}; mean: {}: '.format(bad_low_data.shape, bad_low_data.mean()))
        bad_low_data = nor_data(bad_low_data)
        if save_intermediate:
            dxchange.write_tiff(bad_low_data, 'debug/bad_low_patches_{}.tiff'.format(i_batch), dtype='float32', overwrite=True)
        np.random.shuffle(bad_low_data)

    if bad_high_data is not None:
        # bad_data = nor_data(bad_data)
        print ('bad_high_data raw shape: {}; mean: {}'.format(bad_high_data.shape, bad_high_data.mean()))
        bad_high_data = img_window(bad_high_data[:, window[0][0]:window[1][0], window[0][1]:window[1][1]], 200, reject_bg=True, threshold=1.2e-4)
        bad_high_data = convolve_stack(bad_high_data, get_gradient_kernel())
        print ('bad_high_data windows shape: {}; mean: {}: '.format(bad_high_data.shape, bad_high_data.mean()))
        bad_high_data = extract_3d(bad_high_data, patch_size, 6, reject_bj=False, threshold=1.2e-4)
        print('bad_high_data extracted patches shape: {}; mean: {}: '.format(bad_high_data.shape, bad_high_data.mean()))
        bad_high_data = nor_data(bad_high_data)
        if save_intermediate:
            dxchange.write_tiff(bad_high_data, 'debug/bad_high_patches_{}.tiff'.format(i_batch), dtype='float32', overwrite=True)
        np.random.shuffle(bad_high_data)

    if good_data is not None:
        # good_data = nor_data(good_data)
        print ('good_data raw shape: {}; mean: {}'.format(good_data.shape, good_data.mean()))
        good_data = img_window(good_data[:, window[0][0]:window[1][0], window[0][1]:window[1][1]], 400, reject_bg=True, threshold=1.2e-4)
        good_data = convolve_stack(good_data, get_gradient_kernel())
        print ('good_data windows shape: {}; mean: {}: '.format(good_data.shape, good_data.mean()))
        good_data = extract_3d(good_data, patch_size, 4, reject_bj=False, threshold=1.2e-4)
        print ('good_data extracted patches shape: {}; mean: {}: '.format(good_data.shape, good_data.mean()))
        good_data = nor_data(good_data)
        if save_intermediate:
            dxchange.write_tiff(good_data, 'debug/good_patches_{}.tiff'.format(i_batch), dtype='float32', overwrite=True)
        np.random.shuffle(good_data)

    divider_bad_low = int(bad_low_data.shape[0] * 0.7)
    divider_bad_high = int(bad_high_data.shape[0] * 0.7)
    divider_good = int(good_data.shape[0] * 0.7)
    x_train = np.concatenate((bad_low_data[0:divider_bad_low], bad_high_data[0:divider_bad_high], good_data[0:divider_good]), axis=0)
    x_test = np.concatenate((bad_low_data[divider_bad_low:], bad_high_data[divider_bad_high:], good_data[divider_good:]), axis=0)
    x_train = x_train.reshape(x_train.shape[0], 1, dim_img, dim_img)
    x_test = x_test.reshape(x_test.shape[0], 1, dim_img, dim_img)

    # label bad_low as 0, bad_high as 1, good as 2
    y_train = np.zeros(len(x_train))
    y_train[divider_bad_low:] += 1
    y_train[divider_bad_low+divider_bad_high:] += 1
    y_test = np.zeros(len(x_test))
    y_test[bad_low_data.shape[0]-divider_bad_low:] += 1
    y_test[bad_low_data.shape[0]-divider_bad_low+bad_high_data.shape[0]-divider_bad_high:] += 1

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