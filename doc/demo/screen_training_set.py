import os
import glob
import numpy as np

import dxchange

expected_size = 1920
padding_in_case_err = 1000
expected_volume = expected_size * expected_size * 4

print('Checking file sizes...')
flist = glob.glob(os.path.join('training_set', 'good', '*.tiff'))
for f in flist:
    print(f)
    if os.path.getsize(f) > expected_volume * 1.2:
        print('{} is oversized. Cropping.'.format(f))
        img = dxchange.read_tiff(f)
        img = img[padding_in_case_err:padding_in_case_err + expected_size,
              padding_in_case_err:padding_in_case_err + expected_size]
        dxchange.write_tiff(img, f, dtype='float32', overwrite=True)

flist = glob.glob(os.path.join('training_set', 'bad', '*.tiff'))
for f in flist:
    print(f)
    if os.path.getsize(f) > expected_volume * 1.2:
        print('{} is oversized. Cropping.'.format(f))
        img = dxchange.read_tiff(f)
        img = img[padding_in_case_err:padding_in_case_err + expected_size,
              padding_in_case_err:padding_in_case_err + expected_size]
        dxchange.write_tiff(img, f, dtype='float32', overwrite=True)