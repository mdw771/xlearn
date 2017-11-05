import os
import glob
import shutil
import numpy as np

import dxchange

from xlearn.utils import *


def create_training_set(dir='.', dest_folder='training_set', window=((800, 800), (1600, 1600)), reject_bg=True, **kwargs):

    try:
        os.mkdir(dest_folder)
        os.mkdir(os.path.join(dest_folder, 'good'))
        os.mkdir(os.path.join(dest_folder, 'bad'))
    except:
        pass
    if os.path.join(dir, 'center_pos.txt') in os.listdir(dir):
        with open('center_pos.txt') as f:
            center = float(f.readlines()[0])
            good_fname = '{:.2f}.tiff'.format(center)
            bad_fname_ls = ['{:.2f}.tiff'.format(center - 5),
                            '{:.2f}.tiff'.format(center - 10),
                            '{:.2f}.tiff'.format(center - 20),
                            '{:.2f}.tiff'.format(center + 5),
                            '{:.2f}.tiff'.format(center + 10),
                            '{:.2f}.tiff'.format(center + 20)]
            if os.path.exists(os.path.join(dir, 'center', good_fname)):
                dest_fname = str(get_max_index(os.path.join(dest_folder, 'good')) + 1) + '.tiff'
                shutil.copy(os.path.join(dir, 'center', good_fname),
                            os.path.join(dest_folder, 'good', dest_fname))

                for bad_fname in bad_fname_ls:
                    if os.path.exists(os.path.join(dir, 'center', bad_fname)):
                        dest_fname = str(get_max_index(os.path.join(dest_folder, 'bad')) + 1) + '.tiff'
                        shutil.copy(os.path.join(dir, 'center', bad_fname),
                                    os.path.join(dest_folder, 'bad', dest_fname))
    else:
        folder_list = get_folder_list(dir)
        for folder in folder_list:
            create_training_set(dir=folder, window=window, **kwargs)



def get_folder_list(dir, folder_pattern='*'):

    return [o for o in glob.glob(os.path.join(dir, folder_pattern)) if os.path.isdir(o)]


def get_max_index(dir):

    flist = glob.glob(os.path.join(dir, '*.tiff'))
    flist.sort()
    return int(os.path.splitext(flist[-1])[0])
