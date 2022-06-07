# Python libraries
import os
import datetime
import time
import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

# Pytorch
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

# Lib files
import lib.utils as utils
import lib.Loading as medical_loaders
from lib.utils.setting import parse_opts
from lib.utils.model import generate_model
# from lib.utils.model_spatial import generate_model
from lib.Loading.feedtube_pretrain_clinic import FEEDTUBE
# import lib.Trainers.pytorch_trainer_pretrain_clinic as pytorch_trainer
import lib.Trainers.pytorch_trainer_pretrain_clinic_softmax_update_orig as pytorch_trainer
from lib.preprocess_normalize.aug_normalize import obtain_global_min_max_after_manipulation
from lib.utils.logger import log
from lib.medzoo.ResNet3DMedNet import generate_resnet3d
from lib.Models.resnet_fork import resnet50

# try to address speed issues?
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
cudnn.deterministic = True

'''

Current adjustments to the code from baseline: using model_spatial instead of model in the module section.
Will have to revert. Will also have to confirm new transfer learning layer selections is appropriate. 

if change back to prior method then will have to change # --new_layer_names avgpool dropout fc 

'''


def main():
    time_stamp = datetime.datetime.now()
    print("Time stamp " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'), '\n\n')

    print("Arguments Used")
    args = parse_opts()

    args.time_stamp = time_stamp.strftime('%Y.%m.%d')
    print('')


    if float(args.resnet_lr_factor) > 0 and args.batch_size > 16:
        print('\ndownsizing batch size because now we are training the whole model\n')
        args.batch_size = 16
        print(args.batch_size)
    else:
        pass

    print(f"Setting seed for reproducibility\n\tseed: {args.manual_seed}")
    utils.general.reproducibility(args, args.manual_seed)
    # print(f"Creating saved folder: {args.save}")
    # utils.general.make_dirs(args.save)

    print("\nCreating custom training and validation generators")
    print(f"\tIs data augmentation being utilized?: {args.augmentation}")
    # args.augmentation = ast.literal_eval(args.augmentation)

    print(f"\tBatch size: {args.batch_size}")
    print(f"selected clinical features: {args.selected_clinical_features}")

    # getting data
    args.phase = 'train'
    if args.no_cuda:
        args.pin_memory = False
    else:
        args.pin_memory = True

    ### ADDED
    # Loading csv file that contains list of MRN
    if args.exclude_mrn.lower() == 'true':
        exclude_dir = args.exclude_mrn_path
        exclude_csv_filename = args.exclude_mrn_filename
        exclude_mrns = pd.read_csv(os.path.join(exclude_dir, exclude_csv_filename))

        # Currently loaded as dataframe
        exclude_mrns = exclude_mrns['MRN'].tolist()
        exclude_mrns = [str(int(x)) for x in exclude_mrns]
    elif args.exclude_mrn.lower() == 'false':
        exclude_mrns = []

    args.exclude_mrns = exclude_mrns

    # LOADING DATA (MRN, LABEL) TO LATER PERFORM DATA SPLIT FOR CROSS-VALIDATION
    # train_label path, val_label_path, and test_label_path are all the same
    total_mrn_list = [x.split('/')[-1].split('_')[0] for x in os.listdir(args.train_label_path) if \
                      x.split('/')[-1].split('_')[0] not in exclude_mrns]

    total_label_list = [np.load(os.path.join(args.train_label_path, x)) for x in os.listdir(args.train_label_path) if \
                        x.split('/')[-1].split('_')[0] not in exclude_mrns]

    total_outcome_zip = list(zip(total_mrn_list, total_label_list))

    # print(total_outcome_zip)

    # need to add filter that handles missing input files w/ corresponding labels. More labels because not filtering out
    # data preprocessing step offline
    total_mrn_list = [x for x, k in total_outcome_zip if
                      any(xs.split('/')[-1].split('_')[0] in x for xs in os.listdir(args.train_dataset_path))]
    total_label_list = [k for x, k in total_outcome_zip if
                        any(xs.split('/')[-1].split('_')[0] in x for xs in os.listdir(args.train_dataset_path))]

    # print(total_mrn_list)
    # print(total_label_list)
    args.true_cv_count = 0
    # StratifiedKFold(n_splits=args.cv_num, rano  )
    skf = StratifiedKFold(n_splits=args.cv_num, random_state=args.manual_seed, shuffle=True)

    # print(total_mrn_list)
    total_mrn_array = np.array(total_mrn_list)
    total_label_array = np.array(total_label_list)

    for train_index, test_index in skf.split(total_mrn_array, total_label_array):
        #         print("TRAIN:", train_index, "TEST:", test_index)

        X_train, xtest_ = total_mrn_array[train_index], total_mrn_array[test_index]
        y_train, ytest_ = total_label_array[train_index], total_label_array[test_index]

        X_train = list(X_train)
        xtest_ = list(xtest_)
        y_train = list(y_train)
        ytest_ = list(ytest_)

        # print(X_train)
        # print(len(xtest_)/len(total_label_list))
        # print(y_train)

        ### Creating the Split for training and validation and test
        # (what I am about to do isnt the most correct way to do. will fix if good results)
        args.single_fold = args.true_cv_count
        #
        # train_test_split_fraction = 0.2
        train_val_split_fraction = 0.25

        # X_train, xtest_, y_train, ytest_ = train_test_split(total_mrn_list, total_label_list,
        #                                                     test_size=train_test_split_fraction,
        #                                                     random_state=args.manual_seed + cv)

        xtrain_, xval_, ytrain_, yval_ = train_test_split(X_train, y_train,
                                                          test_size=train_val_split_fraction,
                                                          random_state=args.manual_seed + args.true_cv_count,
                                                          stratify=y_train)

        print('train outcome ratio: ', np.sum(np.array(ytrain_)) / len(np.array(ytrain_)))
        print('val outcome ratio: ', np.sum(np.array(yval_)) / len(np.array(yval_)))
        print('test outcome ratio: ', np.sum(np.array(ytest_)) / len(np.array(ytest_)))

        print('seeing what xtrain_ is: ', xtrain_)
        train_mrn_filename_norm = [x + '_cropped_150-80-80_days_6-10.npy' for x in xtrain_]
        # obtain min and max for normalization; BED value is obtained inside function below
        obtain_global_min_max_after_manipulation(
            train_mrn_filenames=train_mrn_filename_norm,
            data_store_dir=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Data/ft_data_5/store_img_crop_npy/',
            store_dir=args.min_max_key_path,
            csv_filename='min_max_key.csv')

        ### ADDED
        # Loading csv file that contains list of MRN
        if args.exclude_mrn.lower() == 'true':
            exclude_dir = args.exclude_mrn_path
            exclude_csv_filename = args.exclude_mrn_filename
            exclude_mrns = pd.read_csv(os.path.join(exclude_dir, exclude_csv_filename))

            # Currently loaded as dataframe
            exclude_mrns = exclude_mrns['MRN'].tolist()
            exclude_mrns = [str(int(x)) for x in exclude_mrns]
        elif args.exclude_mrn.lower() == 'false':
            exclude_mrns = []

        # Repeated to exclude mrns obtained above
        args.train_mrn_list = [x for x in xtrain_ if x not in exclude_mrns]
        args.val_mrn_list = [x for x in xval_ if x not in exclude_mrns]
        args.test_mrn_list = [x for x in xtest_ if x not in exclude_mrns]

        print('\nmrn list')
        print(f"train: {args.train_mrn_list}")
        print(f"val: {args.val_mrn_list}")
        print(f"test: {args.test_mrn_list}")
        print('')

        # TRAINING
        training_dataset = FEEDTUBE(args, mode='train',
                                    train_path=args.train_dataset_path,
                                    val_path=args.val_dataset_path,
                                    test_path=args.test_dataset_path,
                                    # dataset_path=args.train_dataset_path,
                                    label_path=args.train_label_path,
                                    exclude_mrns=exclude_mrns)
        training_generator = DataLoader(training_dataset, 
                                        batch_size=args.batch_size,
                                        # batch_size=7,
                                        shuffle=True,
                                        pin_memory=args.pin_memory,
                                        num_workers=args.num_workers,
                                        drop_last=True)

        # VALIDATION
        validation_dataset = FEEDTUBE(args, mode='val',
                                      train_path=args.train_dataset_path,
                                      val_path=args.val_dataset_path,
                                      test_path=args.test_dataset_path,
                                      # dataset_path=args.train_dataset_path,
                                      label_path=args.val_label_path,
                                      exclude_mrns=exclude_mrns)
        val_generator = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0,
                                        drop_last=True)

        # TESTING
        test_dataset = FEEDTUBE(args, mode='test',
                                train_path=args.train_dataset_path,
                                val_path=args.val_dataset_path,
                                test_path=args.test_dataset_path,
                                # dataset_path=args.train_dataset_path,
                                label_path=args.test_label_path,
                                exclude_mrns=exclude_mrns)
        test_generator = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                        drop_last=True)

        # Setting model and optimizer
        print('')
        # torch.manual_seed(sets.manual_seed) # already set above

        if 'basicresnet50' not in args.short_note:
            model, parameters = generate_model(args)
            summary(model, (args.in_modality, args.input_D, args.input_H, args.input_W))
            # print(model)

            # optimizer
            if args.ci_test:
                params = [{'params': parameters, 'lr': args.learning_rate}]
            else:
                params = [
                    {'params': parameters['base_parameters'], 'lr': args.learning_rate * args.resnet_lr_factor},
                    {'params': parameters['new_parameters'], 'lr': args.learning_rate}
                ]

            print('rn lr: ', float(args.resnet_lr_factor))
            print('exclude type: ', type(args.exclude_mrn))
            # print('parameters: \n', params)

            for k, v in model.named_parameters():
                # print(k, v)
                temp = k[7:]
                # print('why here')
                # print(k[7:])
                weight = temp[:-7]
                bias = temp[:-5]
                # print('weight: {}, bias: {}'.format(weight, bias))
                if weight in args.new_layer_names:
                    # print('{} skipped'.format(k))
                    continue
                elif bias in args.new_layer_names:
                    # print('{} skipped'.format(k))
                    continue

                # needed to commit out to train resnet layers.

                else:
                    # print('RequireGrad for {} set False'.format(k))
                    if float(args.resnet_lr_factor) == 0.0:
                        v.requires_grad = False
                    else:
                        # v.requires_grad = True
                        pass

            optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)

            # print('\nCheck layers require grad:')
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print('{} requires grad'.format(name))
        else:
            print('Creating basic resnet50')
            # model = generate_resnet3d()
            model = resnet50(sample_input_W=args.input_W,
                             sample_input_H=args.input_H,
                             sample_input_D=args.input_D,
                             shortcut_type=args.resnet_shortcut,
                             no_cuda=args.no_cuda,
                             num_classes=args.n_classes)
            summary(model, (args.in_modality, args.input_D, args.input_H, args.input_W))
            params = model.parameters()
            optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
        # print('parameters (look here): \n', params)

        if args.resume_path:
            if os.path.isfile(args.resume_path):
                print("=> loading checkpoint '{}'".format(arg.resume_path))
                checkpoint = torch.load(arg.resume_path)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(arg.resume_path, checkpoint['epoch']))
        # print(f"\t# Trainable model parameters: {sum(param.numel() for param in model.parameters() if param.requires_grad)}")

        # Selecting loss function
        criterion_pre = nn.CrossEntropyLoss(ignore_index=-1,
                                            weight=torch.FloatTensor(ast.literal_eval(args.class_weights)))

        criterion_pre = criterion_pre.to(torch.device('cuda'))

        # tb_logger = SummaryWriter('./events/{}'.format(args.short_note))
        if args.use_tb.lower() == 'true':
            tb_logger = SummaryWriter(f"./pretrain_clinic/{args.short_note}_{args.resnet_lr_factor}_" \
                                      f"{args.input_H, args.input_W, args.input_D}_exclude_{str(args.exclude_mrn)}_" \
                                      f"{args.exclude_mrn_filename.split('_')[-2]}_{args.class_weights}-{args.time_stamp}-{args.true_cv_count + 1}_of_{args.cv_num}_"
                                      f"seed_{args.manual_seed}")
        else:
            tb_logger = None

        args.true_cv_count += 1

        print("Assessing GPU usage")
        if args.cuda:
            print(f"\tCuda set to {args.cuda}\n")
            model = model.to(torch.device('cuda'))
        # summary(model, (args.in_modality, args.input_D, args.input_H, args.input_W))

        print("Initializing training")
        trainer = pytorch_trainer.Trainer(args, model, criterion_pre, optimizer, train_data_loader=training_generator, \
                                          valid_data_loader=val_generator, test_data_loader=test_generator,
                                          lr_scheduler=scheduler, tb_logger=tb_logger)

        print("Start training!")
        trainer.training()


if __name__ == '__main__':
    main()
