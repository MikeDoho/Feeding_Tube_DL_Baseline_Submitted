# %% Modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from skimage.transform import resize
from scipy.ndimage import interpolation, rotate


# %% Augment and normalize
class img_man:

    @staticmethod
    def clip_cbct_ct(ct, min_=-1000, max_=1000, input_shape=(150, 80, 80)):
        # print(np.shape(ct))
        assert np.shape(ct) == input_shape, f"confirm input shape. should be {input_shape}"
        return np.clip(ct, min_, max_)

    @staticmethod
    def bed_calc(dose, fractions, ab=3):
        return dose * (1 + (dose / fractions) / ab)

    @staticmethod
    def dose_bed(dose, fractions=35, ab=3, input_shape=(150, 80, 80)):

        assert np.shape(dose) == input_shape, f"confirm input shape should be {input_shape}"

        def bed_calc(dose, fractions=fractions, ab=ab):
            return dose * (1 + (dose / fractions) / ab)

        return np.clip(np.apply_along_axis(bed_calc, 2, dose), 0, np.inf)

    @staticmethod
    def norm_cbct_ctsim_dose(composite_npy_file,
                             min_max_key_dir=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Data/ft_data_5/min_max_key_train_active_norm/',
                             input_shape=(150, 80, 80, 3), filter_performed=False,
                             csv_filename='min_max_key.csv'):

        """
        :param composite_npy_file: composite file [cbct, ctsim, dose]
        :return: normalized composite file by channel
        """

        if filter_performed:
            assert np.shape(composite_npy_file) == input_shape, f"shape should be {input_shape}"

            min_max = pd.read_csv(os.path.join(min_max_key_dir, csv_filename))
            min_ = [min_max['cbct_min'][0], min_max['ct_min'][0], min_max['dose_min'][0]]
            max_ = [min_max['cbct_max'][0], min_max['ct_max'][0], min_max['dose_max'][0]]

            # min_ = [-1000, -1000, min_max['dose_min'][0]]
            # max_ = [1000, 1000, min_max['dose_max'][0]]

            # converting to np.array
            min_ = np.array(min_)
            max_ = np.array(max_)

            # print(min_)
            # print(max_)

            return (composite_npy_file - min_) / (max_ - min_)

        else:
            print("Need to make sure filter function aug_filter_cbct_ct_dose before applying normalization")

    @staticmethod
    def clip_norm_cbct_ctsim_dose_reorder(composite_npy_file,
                                          min_max_key_dir=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Data/ft_data_5/min_max_key_train_active_norm/',
                                          input_shape_dose=(150, 80, 80),
                                          input_shape_total=(150, 80, 80, 3),
                                          ct_clip_min=-1000, ct_clip_max=1000, fractions=None, ab=3,
                                          csv_filename='min_max_key.csv'):

        """
        :param composite_npy_file: composite file [cbct, ctsim, dose]
        :return: normalized composite file by channel
        """

        # if filter_performed:
        #     assert np.shape(composite_npy_file) == input_shape, f"shape should be {input_shape}"

        min_max = pd.read_csv(os.path.join(min_max_key_dir, csv_filename))
        # min_ = [min_max['cbct_min'][0], min_max['ct_min'][0], min_max['dose_min'][0]]
        # max_ = [min_max['cbct_max'][0], min_max['ct_max'][0], min_max['dose_max'][0]]

        min_ = [min_max['cbct_min'][0], min_max['ct_min'][0], min_max['dose_min'][0]]
        max_ = [min_max['cbct_max'][0], min_max['ct_max'][0], min_max['dose_max'][0]]

        # converting to np.array
        min_ = np.array(min_)
        max_ = np.array(max_)

        new_composite = np.zeros(input_shape_total)

        # clipping ct and cbct; converting dose to bed
        new_cbct = img_man.clip_cbct_ct(composite_npy_file[..., 0], min_=ct_clip_min, max_=ct_clip_max)
        new_ct = img_man.clip_cbct_ct(composite_npy_file[..., 1], min_=ct_clip_min, max_=ct_clip_max)
        new_dose = img_man.dose_bed(composite_npy_file[..., 2], fractions=fractions, ab=ab,
                                    input_shape=input_shape_dose)

        new_composite[..., 0] = new_cbct
        new_composite[..., 1] = new_ct
        new_composite[..., 2] = new_dose

        # print(min_)
        # print(max_)

        # final output normalizes data

        return (new_composite - min_) / (max_ - min_)


def obtain_fractions(filename, frac_key_dir=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Data/ft_data_5/fraction_key/'):
    # os.chdir(frac_key_dir)
    fraction_key = pd.read_csv(os.path.join(frac_key_dir, 'fraction_key.csv'))
    mrn = filename.split('_')[0]
    frac_idx = fraction_key.loc[fraction_key.MRN == int(mrn), 'Original Fractions'].index.tolist()[0]
    fractions = fraction_key.loc[fraction_key.MRN == int(mrn), 'Original Fractions'][frac_idx]

    return fractions


def obtain_global_min_max_after_manipulation(
        train_mrn_filenames=[],
        data_store_dir=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Data/ft_data_5/store_img_crop_npy/',
        store_dir=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Data/ft_data_5/min_max_key_train_active_norm/',
        csv_filename='min_max_key.csv',
        frac_key_dir=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Data/ft_data_5/fraction_key/'):

    # going to obtain the min and the max of the transformed images
    cbct_min = 0
    cbct_max = 0
    ct_min = 0
    ct_max = 0
    dose_min = 0
    dose_max = 0

    for train in train_mrn_filenames:
        # print(train)
        npy_dir = os.path.join(data_store_dir, train)
        load_file = np.load(npy_dir)

        # clipping images. Will need to clip images with the same values when we normalize and augment
        load_file[..., 0] = img_man.clip_cbct_ct(load_file[..., 0])
        load_file[..., 1] = img_man.clip_cbct_ct(load_file[..., 1])

        # need to load fractions to get proper BED calculation
        fractions = obtain_fractions(filename=train, frac_key_dir=frac_key_dir)
        load_file[..., 2] = img_man.dose_bed(load_file[..., 2], fractions=fractions)

        # # CT scans are clipped and dose map is converted to BED
        # load_file = img_man.aug_filter_cbct_ct_dose(load_file, fractions=fractions, aug=False)

        cbct_min_new, ct_min_new, dose_min_new = np.min(load_file, axis=(0, 1, 2))
        cbct_max_new, ct_max_new, dose_max_new = np.max(load_file, axis=(0, 1, 2))

        if cbct_min_new < cbct_min:
            cbct_min = cbct_min_new
        if ct_min_new < ct_min:
            ct_min = ct_min_new
        if dose_min_new < dose_min:
            dose_min = dose_min_new

        if cbct_max_new > cbct_max:
            cbct_max = cbct_max_new
        if ct_max_new > ct_max:
            ct_max = ct_max_new
        if dose_max_new > dose_max:
            # print(train)
            # print()
            dose_max = dose_max_new

    # os.chdir(store_dir)
    df = pd.DataFrame({"cbct_min": [cbct_min], "cbct_max": [cbct_max], "ct_min": [ct_min],
                       "ct_max": [ct_max], "dose_min": [dose_min], "dose_max": [dose_max]})

    print('')
    print('min/max of cbct, ct, dose')
    print(df)
    print('')
    
    df.to_csv(os.path.join(store_dir, csv_filename), index=False)
