#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script 
"""
# To run this example please download the test data from the classify_evaluate folder at 
# http://tinyurl.com/APS-xlearn 

from __future__ import print_function
import dxchange
import numpy as np
from xlearn.utils import nor_data
from xlearn.utils import extract_3d
from xlearn.utils import img_window
from xlearn.classify import model
from keras import backend as K
import matplotlib.pyplot as plt
import time
import glob
import os

# ================================================================
np.random.seed(1337)

window=((700, 700), (1300, 1300))
dest_folder = '../../test/test_data'

dim_img = 128
patch_size = (dim_img, dim_img)
batch_size = 50
nb_classes = 2
nb_epoch = 12

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
nb_evl = 100
# ================================================================

start_time = time.time()
fnames = glob.glob(os.path.join(dest_folder, '*.tiff'))
fnames = np.sort(fnames)

mdl = model(dim_img, nb_filters, nb_conv, nb_classes)

mdl.load_weights('weight_center.h5')
print('The model loading time is %s seconds'%(time.time()-start_time))
start_time = time.time()
Y_score = np.zeros((len(fnames)))

for i in range(len(fnames)):
    print(fnames[i])
    img = dxchange.read_tiff(fnames[i])
    img = nor_data(img)
    X_evl = np.zeros((nb_evl, dim_img, dim_img))

    for j in range(nb_evl):
        X_evl[j] = img_window(img[window[0][0]:window[1][0], window[0][1]:window[1][1]], dim_img, reject_bg=True, threshold=1e-4)
    X_evl = X_evl.reshape(X_evl.shape[0], 1, dim_img, dim_img)

    # get_layer_output = K.function([mdl.layers[0].input],
    #                               [mdl.layers[0].output, mdl.layers[1].output, mdl.layers[2].output, mdl.layers[3].output])
    # print(len(get_layer_output([X_evl])),
    #       get_layer_output([X_evl])[0].shape,
    #       get_layer_output([X_evl])[1].shape,
    #       get_layer_output([X_evl])[2].shape,
    #       get_layer_output([X_evl])[3].shape)

    Y_evl = mdl.predict(X_evl, batch_size=batch_size)
    Y_score[i] = sum(np.dot(Y_evl, [0, 1]))
    #print('The evaluate score is:', Y_score[i])
    #Y_score = sum(np.round(Y_score))/len(Y_score)


ind_max = np.argmax(Y_score)
print('The well-centered reconstruction is:', fnames[ind_max])
print('The prediction runs for %s seconds'%(time.time()-start_time))
plt.plot(Y_score)
plt.show()


