# Python Modules
import os
import glob
import time
import numpy as np
import pandas as pd
import random
# Torch Modules
import torch
from torch.utils.data import Dataset
# Personal Modules
import lib.augment3D as augment3D
from lib.utils.logger import log
from lib.utils.evaluation_metrics import cbct_ctsim_dose_image_review
from lib.preprocess_normalize.aug_normalize import img_man, obtain_fractions


class FEEDTUBE(Dataset):
    """
    Code for reading the infant brain MICCAIBraTS2018 challenge
    """

    def __init__(self, args, mode, dataset_path='./datasets_main', label_path='./datasets', classes=2,
                 exclude_mrns=[], train_path='./datasets_train', val_path='./datasets_val', test_path='./datasets_test',
                 clinic_image_eval=False, tta=False):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param samples: number of sub-volumes that you want to create
        """
        self.mode = mode
        self.root = dataset_path
        self.label_path = label_path
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        # self.normalization = args.normalization
        self.augmentation = args.augmentation
        self.list = []
        # self.samples = samples
        self.full_volume = None
        self.classes = classes
        self.exclude_mrns = exclude_mrns
        self.args = args
        self.clinic_image_eval = clinic_image_eval
        self.tta = tta
        # self.aug_percent = aug_percent

        # self.train_mrn_list = train_mrn_list
        # self.val_mrn_list = val_mrn_list
        # self.test_mrn_list = test_mrn_list
        self.transform_tta = augment3D.RandomChoice(
            transforms=[augment3D.GaussianNoise(mean=0, std=0.01*self.args.increase_tta_factor),
                        augment3D.RandomShift_tta(max_percentage=0.15*self.args.increase_tta_factor), 
                        augment3D.RandomRotation(min_angle=-15*self.args.increase_tta_factor,
                                                 max_angle=15*self.args.increase_tta_factor)],
            p=1.0)

        if self.augmentation.lower() == 'true' and self.mode == 'train':
            self.transform = augment3D.RandomChoice(
                transforms=[augment3D.GaussianNoise(mean=0, std=0.01), augment3D.RandomShift(),
                            augment3D.RandomRotation(min_angle=-15, max_angle=15)], p=self.args.aug_percent)

        # print(self.root)
        # image = glob.glob(os.path.join(self.root, '*.npy'))
        train_image = glob.glob(os.path.join(self.train_path, '*.npy'))
        val_image = glob.glob(os.path.join(self.val_path, '*.npy'))
        test_image = glob.glob(os.path.join(self.test_path, '*.npy'))

        image = train_image + val_image + test_image
        print('len of all image directories: ', len(image))

        # print(f"\tLoading {self.mode} data from", os.path.join(self.root, '*.npy'))

        image = [x for x in image if x.split('/')[-1].split('_')[0] not in self.exclude_mrns]

        print('len of all image directories after excluding mrns: ', len(image))

        labels = []
        for path in image:
            temp = path.split('/')[-1]
            temp = temp.split('_')[0]
            labels += glob.glob(os.path.join(self.label_path, temp + '_new_outcome.npy'))

        labels = [x for x in labels if x.split('/')[-1].split('_')[0] not in self.exclude_mrns]

        ### Added to gather data for clinic model training and validation###
        if self.mode == 'train':
            # # Right now will not need. will save csv
            # train_mrn = [x.split('/')[-1].split('_')[0] for x in labels]
            # df_train = pd.DataFrame()
            # df_train['mrn'] = train_mrn
            #
            # # save file as csv
            # save_location = r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/RN_pretrain_clinic/'
            # train_mrn_str = 'train_mrn.csv'
            # df_train.to_csv(os.path.join(save_location, train_mrn_str), index=False)

            # this will be needed for current code (need to take into account split in Train_Transfer_Clinic)
            image = [x for x in image if x.split('/')[-1].split('_')[0] in self.args.train_mrn_list]
            labels = [x for x in labels if x.split('/')[-1].split('_')[0] in self.args.train_mrn_list]
            
            
            # adjustment made 01032022; set update above will create repeat(x3 of training MRN)
            image = [x for x in image if x.split('/')[-1].split('_')[0] in self.args.train_mrn_list and \
                     not x.split('/')[-1].split('.')[0].split('_')[-1].isnumeric()]
            image = sorted(list(set(image)), key=lambda x: x.split('/')[-1].split('_')[0])

            labels = [x for x in labels if x.split('/')[-1].split('_')[0] in self.args.train_mrn_list]
            labels = sorted(list(set(labels)), key=lambda x: int(x.split('/')[-1].split('_')[0]))

            # save file as csv
            # Right now will not need. will save csv
            train_mrn = [x.split('/')[-1].split('_')[0] for x in labels]
            df_train = pd.DataFrame()
            df_train['mrn'] = train_mrn

            save_location = r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/RN_pretrain_clinic/'
            train_mrn_str = 'train_mrn.csv'
            df_train.to_csv(os.path.join(save_location, train_mrn_str), index=False)
            

            print('Length of training images: ', len(image))
            print('Length of training image labels: ', len(labels))

        if self.mode == 'val':
            # Right now will not need. will save csv
            val_mrn = [x.split('/')[-1].split('_')[0] for x in labels]
            df_val = pd.DataFrame()
            df_val['mrn'] = val_mrn

            # save file as csv
            save_location = r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/RN_pretrain_clinic/'
            val_mrn_str = 'val_mrn.csv'
            df_val.to_csv(os.path.join(save_location, val_mrn_str), index=False)

            # this will be needed for current code (need to take into account split in Train_Transfer_Clinic)
            print('len val old: ', np.shape(image))
            image = [x for x in image if x.split('/')[-1].split('_')[0] in self.args.val_mrn_list and \
                     not x.split('/')[-1].split('.')[0].split('_')[-1].isnumeric()]

            image = sorted(list(set(image)), key=lambda x: x.split('/')[-1].split('_')[0])

            labels = [x for x in labels if x.split('/')[-1].split('_')[0] in self.args.val_mrn_list]
            labels = sorted(list(set(labels)), key=lambda x: int(x.split('/')[-1].split('_')[0]))

            print('Length of sorted validation images: ', len(image))
            print('Length of sorted validation image labels: ', len(labels))

        if self.mode == 'test':
            test_mrn = [x.split('/')[-1].split('_')[0] for x in labels]
            df_test = pd.DataFrame()
            df_test['mrn'] = test_mrn

            # save file as csv
            save_location = r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/RN_pretrain_clinic/'
            test_mrn_str = 'test_mrn.csv'
            df_test.to_csv(os.path.join(save_location, test_mrn_str), index=False)

            # this will be needed for current code (need to take into account split in Train_Transfer_Clinic)
            print('Length of old test images: ', len(image))
            image = [x for x in image if x.split('/')[-1].split('_')[0] in self.args.test_mrn_list and \
                     not x.split('/')[-1].split('.')[0].split('_')[-1].isnumeric()]

            image = sorted(list(set(image)), key=lambda x: x.split('/')[-1].split('_')[0])

            labels = [x for x in labels if x.split('/')[-1].split('_')[0] in self.args.test_mrn_list]
            labels = sorted(list(set(labels)), key=lambda x: int(x.split('/')[-1].split('_')[0]))

            print('Length of sorted test images: ', len(image))
            print('Length of sorted test image labels: ', len(labels))

        ### Added to gather data for clinic model training and validation###

        if self.mode == 'train':
            print('\tTraining data size: ', len(image))
            print('\tTraining label size: ', len(labels))
            self.list = []
            for i in range(len(image)):
                sub_list = []
                sub_list.append(image[i])
                sub_list.append(labels[i])
                self.list.append(tuple(sub_list))

        elif self.mode == 'val':
            print('\tValidation data size: ', len(image))
            print('\tValidation label size: ', len(labels))
            self.list = []
            for i in range(len(image)):
                sub_list = []
                sub_list.append(image[i])
                sub_list.append(labels[i])
                self.list.append(tuple(sub_list))

        elif self.mode == 'test':
            print('\tTest data size: ', len(image))
            print('\tTest label size: ', len(labels))
            self.list = []
            for i in range(len(image)):
                sub_list = []
                sub_list.append(image[i])
                sub_list.append(labels[i])
                self.list.append(tuple(sub_list))
                # print(self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):

        # note to Mike: will probably have to add in self.augmentation to code because I dont see it implemented

        f_img, f_label = self.list[index]
        assert f_img.split('/')[-1].split('_')[0] == f_label.split('/')[-1].split('_')[
            0], 'check that mrns are the same'

        # if self.mode != 'train':
        #     print(f_img.split('/')[-1], f_label.split('/')[-1])
        img, lab = np.load(f_img), np.load(f_label)

        if self.args.do_normalization.lower() == 'true':
            # fractions = obtain_fractions(file, frac_key_dir=frac_key_dir)
            fractions = obtain_fractions(filename=f_img.split('/')[-1].split('_')[0],
                                         frac_key_dir=self.args.fraction_key_path)

            # creating single nonaugmented file
            img = img_man.clip_norm_cbct_ctsim_dose_reorder(
                min_max_key_dir=self.args.min_max_key_path,
                input_shape_dose=(self.args.input_H, self.args.input_W, self.args.input_D),
                input_shape_total=(self.args.input_H, self.args.input_W, self.args.input_D, self.args.in_modality),
                fractions=fractions, ab=3,
                composite_npy_file=img)

            # cbct_ctsim_dose_image_review(img[..., 0], img[..., 1], img[..., 2],
            #                              save_fig_name=self.mode+'_norm', view=False,
            #                              fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/saved_images/')
        else:
            pass

        # img = torch.FloatTensor(img)
        # # Need to confirm w/ Kai but I believe this adjust the x,y,z, channel axis
        # img = img.permute(3, 2, 0, 1)

        if not self.clinic_image_eval:
            if self.mode == 'train' and self.augmentation.lower() == 'true':
                # print('did it get here')
                img, lab = self.transform(img, lab)

            else:
                pass

            # cbct_ctsim_dose_image_review(img[..., 0], img[..., 1], img[..., 2],
            #                              save_fig_name=self.mode+'_norm_aug', view=False,
            #                              fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/saved_images/')

            img = torch.FloatTensor(img)
            # Need to confirm w/ Kai but I believe this adjust the x,y,z, channel axis
            img = img.permute(3, 2, 0, 1)
            # print('1')

            return img, lab
        else:
            if self.mode == 'train' and self.augmentation.lower() == 'true':
                # print('did it get here')
                img, lab = self.transform(img, lab)
            elif self.tta:

                # # this will perform only 1 transformation
                # img, lab = self.transform_tta(img, lab)

                # performing all three transformations
                transforms_selection = [augment3D.GaussianNoise(mean=0, std=0.01 * self.args.increase_tta_factor),
                                        augment3D.RandomShift_tta(max_percentage=0.15 * self.args.increase_tta_factor),
                                        augment3D.RandomRotation(min_angle=-15 * self.args.increase_tta_factor,
                                                       max_angle=15 * self.args.increase_tta_factor)]

                # there is a random.sample function in ComposeTransforms_tta
                # will select 1, 2, or all 3 transformations and apply the transformations selected
                all_transform = augment3D.ComposeTransforms_tta(transforms=transforms_selection)
                img, lab = all_transform(img, lab)

                # print('1')
            else:
                pass

            # cbct_ctsim_dose_image_review(img[..., 0], img[..., 1], img[..., 2],
            #                              save_fig_name=self.mode, view=False,
            #                              fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/saved_images/')

            img = torch.FloatTensor(img)
            # Need to confirm w/ Kai but I believe this adjust the x,y,z, channel axis
            img = img.permute(3, 2, 0, 1)

            # if self.mode == 'test':
            #     print(f_img.split('/')[-1].split('_')[0])

            return img, lab, f_img.split('/')[-1].split('_')[0]
