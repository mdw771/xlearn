import os
import glob
import shutil
import gc
import numpy as np

import dxchange

from xlearn.utils import *


def create_training_set(dir='.', dest_folder='training_set', window=((800, 800), (1600, 1600)), reject_bg=True,
                        padding_in_case_err=1000, **kwargs):

    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    if not os.path.exists(os.path.join(dest_folder, 'good')):
        os.mkdir(os.path.join(dest_folder, 'good'))
    if not os.path.exists(os.path.join(dest_folder, 'bad')):
        os.mkdir(os.path.join(dest_folder, 'bad'))

    if 'center_pos.txt' in os.listdir(dir):
        print(dir)
        try:
            with open(os.path.join(dir, 'center_pos.txt')) as f:
                center = float(f.readlines()[0])
                if not (os.path.exists(os.path.join(dir, 'center', '{:.2f}.tiff'.format(center))) or
                        os.path.exists(os.path.join(get_folder_list(os.path.join(dir, 'center'))[0],
                                                    '{:.2f}.tiff'.format(center)))):
                    center += padding_in_case_err
                good_fname = '{:.2f}.tiff'.format(center)
                bad_fname_ls = ['{:.2f}.tiff'.format(center - 5),
                                '{:.2f}.tiff'.format(center - 10),
                                '{:.2f}.tiff'.format(center - 20),
                                '{:.2f}.tiff'.format(center + 5),
                                '{:.2f}.tiff'.format(center + 10),
                                '{:.2f}.tiff'.format(center + 20)]
                if os.path.exists(os.path.join(dir, 'center', good_fname)):
                    dest_fname = os.path.basename(dir) + '.tiff'
                    shutil.copy2(os.path.join(dir, 'center', good_fname),
                                 os.path.join(dest_folder, 'good', dest_fname))
                    for i, bad_fname in enumerate(bad_fname_ls):
                        if os.path.exists(os.path.join(dir, 'center', bad_fname)):
                            dest_fname = os.path.basename(dir) + '_{:02d}.tiff'.format(i)
                            shutil.copy2(os.path.join(dir, 'center', bad_fname),
                                        os.path.join(dest_folder, 'bad', dest_fname))
                else:
                    true_center_folder = get_folder_list(os.path.join(dir, 'center'))
                    true_center_folder.sort()
                    true_center_folder = true_center_folder[int(len(true_center_folder) / 2)]
                    dest_fname = os.path.basename(dir) + '.tiff'
                    shutil.copy2(os.path.join(true_center_folder, good_fname),
                                os.path.join(dest_folder, 'good', dest_fname))
                    for i, bad_fname in enumerate(bad_fname_ls):
                        if os.path.exists(os.path.join(true_center_folder, bad_fname)):
                            dest_fname = os.path.basename(dir) + '_{:02d}.tiff'.format(i)
                            shutil.copy2(os.path.join(true_center_folder, bad_fname),
                                         os.path.join(dest_folder, 'bad', dest_fname))
        except:
            print('WARNING: An error occurred in {}. Proceeding to next folder.'.format(dir))

    else:
        folder_list = get_folder_list(dir)
        for folder in folder_list:
            create_training_set(dir=folder, window=window, **kwargs)
    gc.collect()


if __name__ == '__main__':

    expected_size = 1920
    padding_in_case_err = 1000
    expected_volume = expected_size * expected_size * 4

    create_training_set(expected_size=expected_size,
                        padding_in_case_err=padding_in_case_err)

    print('Checking file sizes...')
    flist = glob.glob(os.path.join('training_set', 'good', '*.tiff'))
    for f in flist:
        if os.path.getsize(f) > expected_volume * 1.2:
            print('{} is oversized. Cropping.'.format(f))
            img = dxchange.read_tiff(f)
            img = np.squeeze(img)[padding_in_case_err:padding_in_case_err + expected_size,
                  padding_in_case_err:padding_in_case_err + expected_size]
            os.remove(f)
            dxchange.write_tiff(img, f, dtype='float32')

    flist = glob.glob(os.path.join('training_set', 'bad', '*.tiff'))
    for f in flist:
        if os.path.getsize(f) > expected_volume * 1.2:
            print('{} is oversized. Cropping.'.format(f))
            img = dxchange.read_tiff(f)
            img = np.squeeze(img)[padding_in_case_err:padding_in_case_err + expected_size,
                  padding_in_case_err:padding_in_case_err + expected_size]
            os.remove(f)
            dxchange.write_tiff(img, f, dtype='float32')