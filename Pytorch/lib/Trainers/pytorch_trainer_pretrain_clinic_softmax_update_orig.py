# Python Modules
import numpy as np
import pandas as pd
import os
import time
import math
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score, classification_report

# Torch Modules
import torch
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

# Self Modules
from lib.utils.general import prepare_input
from lib.utils.logger import log
from lib.utils.evaluation_metrics import roc_auc_plot  # (y_true, y_pred)
from lib.Models.clinical_ft_model import clinical_model
from lib.Loading.clinical_data_loader import clinical_data
from lib.Loading.feedtube_pretrain_clinic import FEEDTUBE



def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
      E.g. for use with categorical_crossentropy.
      Args:
          y: class vector to be converted into a matrix
              (integers from 0 to num_classes).
          num_classes: total number of classes. If `None`, this would be inferred
            as the (largest number in `y`) + 1.
          dtype: The data type expected by the input. Default: `'float32'`.
      Returns:
          A binary matrix representation of the input. The classes axis is placed
          last.
      Example:
      >>> a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
      >>> a = tf.constant(a, shape=[4, 4])
      >>> print(a)
      tf.Tensor(
        [[1. 0. 0. 0.]
         [0. 1. 0. 0.]
         [0. 0. 1. 0.]
         [0. 0. 0. 1.]], shape=(4, 4), dtype=float32)
      >>> b = tf.constant([.9, .04, .03, .03,
      ...                  .3, .45, .15, .13,
      ...                  .04, .01, .94, .05,
      ...                  .12, .21, .5, .17],
      ...                 shape=[4, 4])
      >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
      >>> print(np.around(loss, 5))
      [0.10536 0.82807 0.1011  1.77196]
      >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
      >>> print(np.around(loss, 5))
      [0. 0. 0. 0.]
      Raises:
          Value Error: If input contains string value
      """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


def dr_friendly_measures(outputs, targets):
    with torch.no_grad():
        outputs = torch.argmax(F.softmax(outputs, dim=1), dim=1)

        try:
            tn, fp, fn, tp = confusion_matrix(targets.cpu().numpy(), outputs.cpu().numpy()).ravel()
            specificity = tn / (tn + fp + 1e-12)
            sensitivity = tp / (tp + fn + 1e-12)
            return specificity, sensitivity
        except:
            return np.nan, np.nan


# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))


def dr_friendly_measures_reg(outputs, targets):


    assert np.shape(outputs) == np.shape(targets), 'prediction and target outcomes should be same shape'

    try:
        tn, fp, fn, tp = confusion_matrix(targets, outputs).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        return specificity, sensitivity
    except:
        return np.nan, np.nan


def calculate_auc(outputs, targets):
    with torch.no_grad():
        # log.info(f"loss: {outputs.size()}")
        outputs = F.softmax(outputs, dim=1)
        # log.info(outputs)
        # log.info(outputs.size())
        try:
            auc = roc_auc_score(targets.cpu().numpy(), outputs.type(torch.FloatTensor).cpu().data.numpy()[:, 1])
            return auc
        except:
            return np.nan


def calculate_prauc(outputs, targets):
    with torch.no_grad():
        outputs = F.softmax(outputs, dim=1)
        try:
            prauc = average_precision_score(targets.cpu().numpy(),
                                            outputs.type(torch.FloatTensor).cpu().data.numpy()[:, 1])
            return prauc
        except:
            return np.nan


class Trainer:
    """
    Trainer class
    """

    def __init__(self, args, model, criterion_pre, optimizer, train_data_loader,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, tb_logger=None):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion_pre = criterion_pre
        self.train_data_loader = train_data_loader
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.test_data_loader = test_data_loader
        self.do_test = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.save_frequency = 20
        self.start_epoch = 1
        self.val_loss = 0

        self.print_batch_spacing = 50
        self.save_interval = args.save_intervals
        self.tb_logger = tb_logger
        self.train_count = 0
        self.val_count = 0

    def training(self):

        for epoch in range(self.start_epoch, (self.args.n_epochs + 1)):

            for param_group in self.optimizer.param_groups:
                lr_show = param_group['lr']
            log.info('\n########################################################################')
            log.info(f"Training epoch: {epoch}, Learning rate: {lr_show:.8f}")

            self.train_epoch_alex(epoch)

            if self.do_validation:
                log.info(f"Validation epoch: {epoch}")
                self.validate_epoch_alex(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # # comment out for speed test
            # if epoch % self.save_frequency == 0:
            #     self.model.save_checkpoint(self.args.save,
            #                            epoch, self.val_loss,
            #                            optimizer=self.optimizer)

            # print('\n\n')

        # Saving final model after training
        # if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
        model_save_path = '{}_epoch_{}_{}_fold_{}.pth.tar'.format(self.args.save_folder, epoch, self.args.short_note,
                                                             self.args.true_cv_count)
        model_save_dir = os.path.dirname(model_save_path)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        log.info('Save checkpoints: epoch = {}'.format(epoch))
        torch.save({
            'epoch': epoch,
            # 'batch_id': batch_idx,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()},
            model_save_path)

        ###WILL NEED TO TRAIN CLINICAL MODEL HERE###
        log.info('loading clinical data')
        x_train_clinic, y_train_clinic = \
            clinical_data(args=self.args,
                          train_mrn_list=self.args.train_mrn_list).load_train_data_batch(
                batch_size=len(self.args.train_mrn_list))

        look_at_train = pd.DataFrame(x_train_clinic)
        look_at_train.to_csv('/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/look_at_clinic/trainlook.csv')
        # log.info('train data: ', np.shape(x_train_clinic))

        x_val_clinic, y_val_clinic = \
            clinical_data(args=self.args,
                          val_mrn_list=self.args.val_mrn_list,
                          train_mrn_list=self.args.train_mrn_list).load_val_data_batch(
                batch_size=len(self.args.val_mrn_list))

        # log.info('val data: ', np.shape(x_val_clinic))

        x_test_clinic, y_test_clinic, mrn_list_test_clinic_pred = \
            clinical_data(args=self.args,
                          train_mrn_list=self.args.train_mrn_list,
                          test_mrn_list=self.args.test_mrn_list).load_test_data_batch(
                batch_size=len(self.args.test_mrn_list))

        # log.info('test data: ', np.shape(x_test_clinic))

        # log.info('data_loaded')
        clinical_model_ = clinical_model(args=self.args, X=x_train_clinic, y=y_train_clinic,
                                         model_type=self.args.clinical_model_type).create_model()
        # log.info('model created')

        y_pred_val_clinic = clinical_model_.predict_proba(x_val_clinic)
        y_pred_train_clinic = clinical_model_.predict_proba(x_train_clinic)
        y_pred_test_clinic = clinical_model_.predict_proba(x_test_clinic)

        # print('test predictions')
        # print(y_pred_test_clinic)
        #
        # log.info('predictions made')
        # log.info(os.getcwd())

        # Creating list to store prediction information from image and clinic based model
        auc_store = []
        auc_store_clinic = []

        # rand_num_cv = np.random.randint(0, 1000)

        ### ASSESSING CLINICAL MODEL ###

        print('\nVALIDATION INFORMATION - CLINICAL BASED MODEL\n')
        target_names = ['class ' + str(x) for x in range(self.args.n_classes)]
        print('\n', classification_report(np.argmax(y_val_clinic, axis=-1),
                                          np.argmax(y_pred_val_clinic, axis=-1),
                                          target_names=target_names))

        print('\nTRAIN INFORMATION - CLINIC BASED MODEL\n')
        target_names = ['class ' + str(x) for x in range(self.args.n_classes)]
        print('\n', classification_report(np.argmax(y_train_clinic, axis=-1),
                                          np.argmax(y_pred_train_clinic, axis=-1),
                                          target_names=target_names))

        print('\nTEST INFORMATION - CLINIC BASED MODEL\n')
        target_names = ['class ' + str(x) for x in range(self.args.n_classes)]
        print('\n', classification_report(np.argmax(y_test_clinic, axis=-1),
                                          np.argmax(y_pred_test_clinic, axis=-1),
                                          target_names=target_names))

        # Clinical model evaluation that will be saved to tensorboard
        # Train results
        # random integer correlate appropriate splits

        store_all_label_pred_dir = r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/saved_predictions/green_journal_revisions_feb2022'
        clinic_train_df = pd.DataFrame()
        clinic_train_df['true'] = [np.argmax(x) for x in y_train_clinic]
        clinic_train_df['pred'] = [x[1] for x in y_pred_train_clinic]

        filename = f"clinical_model_train_prediction_" \
                   f"{self.args.short_note}_{self.args.class_weights}_{self.args.single_fold + 1}_of_{self.args.cv_num}_" \
                   f"seed_{self.args.manual_seed}_{self.args.time_stamp}.csv"
        csv_dir = os.path.join(store_all_label_pred_dir, filename)
        clinic_train_df.to_csv(path_or_buf=csv_dir, index=False)
        # AUC values gathered for total prediction model
        auc_t = roc_auc_score(y_true=np.argmax(y_train_clinic, axis=-1), y_score=y_pred_train_clinic[:, 1])
        print('clinical model train auc_: ', auc_t)


        # Validation results
        # random integer correlate appropriate splits

        store_all_label_pred_dir = r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/saved_predictions/green_journal_revisions_feb2022'
        clinic_val_df = pd.DataFrame()
        clinic_val_df['true'] = [np.argmax(x) for x in y_val_clinic]
        clinic_val_df['pred'] = [x[1] for x in y_pred_val_clinic]

        filename = f"clinical_model_val_prediction_" \
                   f"{self.args.short_note}_{self.args.class_weights}_{self.args.single_fold + 1}_of_{self.args.cv_num}_" \
                   f"seed_{self.args.manual_seed}_{self.args.time_stamp}.csv"
        csv_dir = os.path.join(store_all_label_pred_dir, filename)
        clinic_val_df.to_csv(path_or_buf=csv_dir, index=False)

        if self.args.use_tb.lower() == 'true':
            fig_add = roc_auc_plot(y_val_clinic, y_pred_val_clinic, data_title=f"Clinical Model Val AUC")
            self.tb_logger.add_figure(f"Clinical Model Validation AUC", figure=fig_add)
        clinic_spec, clinic_sens = dr_friendly_measures_reg(np.argmax(y_pred_val_clinic, axis=-1),
                                                            np.argmax(y_val_clinic, axis=-1))
        print('val sensitivity: ', clinic_sens, 'val specificity: ', clinic_spec)

        if self.args.use_tb.lower() == 'true':
            self.tb_logger.add_scalar(f"clinic_model_val_spec", clinic_spec, 0)
            self.tb_logger.add_scalar(f"clinic_model_val_sens", clinic_sens, 0)
            self.tb_logger.flush()

        # AUC values gathered for total prediction model
        auc_ = roc_auc_score(y_true=np.argmax(y_val_clinic, axis=-1), y_score=y_pred_val_clinic[:, 1])
        print('clinical model validation auc_: ', auc_)
        auc_store_clinic.append(auc_)

        # Train results
        if self.args.use_tb.lower() == 'true':
            fig_add = roc_auc_plot(y_train_clinic, y_pred_train_clinic, data_title=f"Clinical Model Train AUC")
            self.tb_logger.add_figure(f"Clinical Model Train AUC", figure=fig_add)
        clinic_spec, clinic_sens = dr_friendly_measures_reg(np.argmax(y_pred_train_clinic, axis=-1),
                                                            np.argmax(y_train_clinic, axis=-1))

        if self.args.use_tb.lower() == 'true':
            self.tb_logger.add_scalar(f"clinic_model_train_spec", clinic_spec, 0)
            self.tb_logger.add_scalar(f"clinic_model_train_sens", clinic_sens, 0)
            self.tb_logger.flush()

        # Test results
        # storing test results in csv file. Will combine to make total CV ROC
        clinic_test_df = pd.DataFrame()
        clinic_test_df['true'] = [np.argmax(x) for x in y_test_clinic]
        clinic_test_df['pred'] = [x[1] for x in y_pred_test_clinic]
        clinic_test_df['mrn'] = [x for x in mrn_list_test_clinic_pred]

        filename = f"clinical_model_test_prediction_" \
                   f"{self.args.short_note}_{self.args.class_weights}_{self.args.single_fold + 1}_of_{self.args.cv_num}_" \
                   f"seed_{self.args.manual_seed}_{self.args.time_stamp}.csv"
        csv_dir = os.path.join(store_all_label_pred_dir, filename)
        clinic_test_df.to_csv(path_or_buf=csv_dir, index=False)

        if self.args.use_tb.lower() == 'true':
            fig_add = roc_auc_plot(y_test_clinic, y_pred_test_clinic, data_title=f"Clinical Model Test AUC")
            self.tb_logger.add_figure(f"Clinical Model Test AUC", figure=fig_add)

        clinic_spec, clinic_sens = dr_friendly_measures_reg(np.argmax(y_pred_test_clinic, axis=-1),
                                                            np.argmax(y_test_clinic, axis=-1))

        if self.args.use_tb.lower() == 'true':
            self.tb_logger.add_scalar(f"clinic_model_test_spec", clinic_spec, 0)
            self.tb_logger.add_scalar(f"clinic_model_test_sens", clinic_sens, 0)
            self.tb_logger.flush()

        ### FINISHED ASSESSING CLINICAL MODEL ###

        ### ASSESSING IMAGE MODEL ###

        target_ = []
        pred_ = []
        target2_ = []
        pred2_ = []

        # creating PR curve training/validation

        # cycle through data loader for training data
        for batch_idx, input_tuple in enumerate(self.train_data_loader):
            with torch.no_grad():
                # Gathering input data; prepare_input sends to gpu
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)

                # Model make prediction
                pred = self.model(input_tensor)
                # log.info(f"train: {F.softmax(pred, dim=1).size()}")
                # log.info(F.softmax(pred, dim=1))
                pred_.extend(torch.argmax(F.softmax(pred, dim=1), dim=1).cpu().numpy())
                pred2_.extend(np.array(F.softmax(pred, dim=1).cpu().numpy()).tolist())
                target_.extend(target.cpu().numpy())
                target2_.extend(np.array(F.one_hot(target).cpu().numpy()).tolist())

        # changed 12/24/2021; testing
        # target2_ = np.array(target2_)
        target2_ = np.array(np.eye(self.args.n_classes)[target_])
        assert all(
            [np.shape(x)[0] == self.args.n_classes for x in
             target2_]), f"make sure one hot predictions is correct: {np.shape(target2_)}"
        pred2_ = np.array(pred2_)

        # storing predictions for image model
        image_train_df = pd.DataFrame()
        image_train_df['true'] = [np.argmax(x) for x in target2_]
        image_train_df['pred'] = [x[1] for x in pred2_]
        filename = f"image_model_train_prediction_" \
                   f"{self.args.short_note}_{self.args.class_weights}_{self.args.single_fold + 1}_of_{self.args.cv_num}_" \
                   f"seed_{self.args.manual_seed}_{self.args.time_stamp}.csv"
        csv_dir = os.path.join(store_all_label_pred_dir, filename)
        image_train_df.to_csv(path_or_buf=csv_dir, index=False)

        auc_tm = roc_auc_score(y_true=np.argmax(target2_, axis=-1), y_score=pred2_[:, 1])
        print('image model train auc_: ', auc_tm)

        if self.args.use_tb.lower() == 'true':
            fig_add = roc_auc_plot(target2_, pred2_, data_title='Training ROC')
            self.tb_logger.add_figure(f"Image Model Train AUC", figure=fig_add)

        target_ = torch.from_numpy(np.array(target_).astype('int32'))
        pred_ = torch.from_numpy(np.array(pred_).astype('int32'))

        if self.args.use_tb.lower() == 'true':
            self.tb_logger.add_pr_curve(f"Image Model Training PR Curve", target_, pred_)
            self.tb_logger.flush()

        target_ = []
        pred_ = []
        target2_ = []
        pred2_ = []

        # cycle through data loader for validation data
        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            with torch.no_grad():
                # Gathering input data; prepare_input sends to gpu
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)

                # Model make prediction
                pred = self.model(input_tensor)
                pred_.extend(torch.argmax(F.softmax(pred, dim=1), dim=1).cpu().numpy())
                pred2_.extend(np.array(F.softmax(pred, dim=1).cpu().numpy()).tolist())

                target_.extend(target.cpu().numpy())
                target2_.extend(np.array(F.one_hot(target).cpu().numpy()).tolist())

        # changed 12/24/2021; testing
        # target2_ = np.array(target2_)
        target2_ = np.array(np.eye(self.args.n_classes)[target_])
        print('val length target: ', np.shape(target2_))
        assert all(
            [np.shape(x)[0] == self.args.n_classes for x in
             target2_]), f"make sure one hot predictions is correct: {np.shape(target2_)}"
        pred2_ = np.array(pred2_)
        print('val length pred2_: ', np.shape(pred2_))

        # storing predictions for image model
        image_val_df = pd.DataFrame()
        image_val_df['true'] = [np.argmax(x) for x in target2_]
        image_val_df['pred'] = [x[1] for x in pred2_]
        filename = f"image_model_val_prediction_" \
                   f"{self.args.short_note}_{self.args.class_weights}_{self.args.single_fold + 1}_of_{self.args.cv_num}_" \
                   f"seed_{self.args.manual_seed}_{self.args.time_stamp}.csv"
        csv_dir = os.path.join(store_all_label_pred_dir, filename)
        image_val_df.to_csv(path_or_buf=csv_dir, index=False)

        if self.args.use_tb.lower() == 'true':
            fig_add = roc_auc_plot(target2_, pred2_, data_title='Validation ROC')
            self.tb_logger.add_figure(f"Image Model Validation AUC", figure=fig_add)

        # AUC values gathered for total prediction model
        auc_ = roc_auc_score(y_true=np.argmax(target2_, axis=-1), y_score=pred2_[:, 1])
        auc_store.append(auc_)

        target_ = torch.from_numpy(np.array(target_).astype('int32'))
        pred_ = torch.from_numpy(np.array(pred_).astype('int32'))

        if self.args.use_tb.lower() == 'true':
            self.tb_logger.add_pr_curve(f"Image Model Validation PR Curve", target_, pred_)
            self.tb_logger.flush()

        target_ = []
        pred_ = []
        target2_ = []
        pred2_ = []

        # Testing
        # cycle through data loader for testing data
        for batch_idx, input_tuple in enumerate(self.test_data_loader):
            with torch.no_grad():
                # Gathering input data; prepare_input sends to gpu
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)

                # Model make prediction
                pred = self.model(input_tensor)
                pred_.extend(torch.argmax(F.softmax(pred, dim=1), dim=1).cpu().numpy())
                pred2_.extend(np.array(F.softmax(pred, dim=1).cpu().numpy()).tolist())

                target_.extend(target.cpu().numpy())
                target2_.extend(np.array(F.one_hot(target).cpu().numpy()).tolist())

        # testing 12/24/2021;
        # target2_ = np.array(target2_)
        target2_ = np.array(np.eye(self.args.n_classes)[target_])
        print('test length target: ', np.shape(target2_))
        assert all(
            [np.shape(x)[0] == self.args.n_classes for x in
             target2_]), f"make sure one hot predictions is correct: {np.shape(target2_)}"
        pred2_ = np.array(pred2_)
        print('test length pred2_: ', np.shape(pred2_))

        # storing predictions for image model
        image_test_df = pd.DataFrame()
        image_test_df['true'] = [np.argmax(x) for x in target2_]
        image_test_df['pred'] = [x[1] for x in pred2_]
        filename = f"image_model_test_prediction_" \
                   f"{self.args.short_note}_{self.args.class_weights}_{self.args.single_fold + 1}_of_{self.args.cv_num}_" \
                   f"seed_{self.args.manual_seed}_{self.args.time_stamp}.csv"
        csv_dir = os.path.join(store_all_label_pred_dir, filename)
        image_test_df.to_csv(path_or_buf=csv_dir, index=False)

        if self.args.use_tb.lower() == 'true':
            fig_add = roc_auc_plot(target2_, pred2_, data_title='Test ROC')
            self.tb_logger.add_figure(f"Image Model Test AUC", figure=fig_add)

        target_ = torch.from_numpy(np.array(target_).astype('int32'))
        pred_ = torch.from_numpy(np.array(pred_).astype('int32'))

        if self.args.use_tb.lower() == 'true':
            self.tb_logger.add_pr_curve(f"Image Model Test PR Curve", target_, pred_)
            self.tb_logger.flush()

        ### FINISHED ASSESSING IMAGE MODEL ###

        ### ASSESSING IMAGE + CLINIC MODEL ###

        # image factor / (image factor + clinic factor)
        alpha_image = auc_store[0] / (auc_store[0] + auc_store_clinic[0])

        # Creating new validation and testing generators for validation
        # VALIDATION
        validation_dataset_eval = FEEDTUBE(self.args, mode='val',
                                           train_path=self.args.train_dataset_path,
                                           val_path=self.args.val_dataset_path,
                                           test_path=self.args.test_dataset_path,
                                           # dataset_path=args.train_dataset_path,
                                           label_path=self.args.val_label_path,
                                           exclude_mrns=self.args.exclude_mrns,
                                           clinic_image_eval=True)
        val_generator_eval = DataLoader(validation_dataset_eval, batch_size=1, shuffle=False, num_workers=0)

        # For TTA
        validation_dataset_eval_tta = FEEDTUBE(self.args, mode='val',
                                               train_path=self.args.train_dataset_path,
                                               val_path=self.args.val_dataset_path,
                                               test_path=self.args.test_dataset_path,
                                               # dataset_path=args.train_dataset_path,
                                               label_path=self.args.val_label_path,
                                               exclude_mrns=self.args.exclude_mrns,
                                               clinic_image_eval=True, tta=True)
        val_generator_eval_tta = DataLoader(validation_dataset_eval_tta, batch_size=1, shuffle=False, num_workers=0)

        # TESTING
        test_dataset_eval = FEEDTUBE(self.args, mode='test',
                                     train_path=self.args.train_dataset_path,
                                     val_path=self.args.val_dataset_path,
                                     test_path=self.args.test_dataset_path,
                                     # dataset_path=args.train_dataset_path,
                                     label_path=self.args.test_label_path,
                                     exclude_mrns=self.args.exclude_mrns,
                                     clinic_image_eval=True)
        test_generator_eval = DataLoader(test_dataset_eval, batch_size=1, shuffle=False, num_workers=0)

        # For TTA
        test_dataset_eval_tta = FEEDTUBE(self.args, mode='test',
                                         train_path=self.args.train_dataset_path,
                                         val_path=self.args.val_dataset_path,
                                         test_path=self.args.test_dataset_path,
                                         # dataset_path=args.train_dataset_path,
                                         label_path=self.args.test_label_path,
                                         exclude_mrns=self.args.exclude_mrns,
                                         clinic_image_eval=True, tta=True)
        test_generator_eval_tta = DataLoader(test_dataset_eval_tta, batch_size=1, shuffle=False, num_workers=0)

        # Calculating MC Dropout
        if self.args.mc_do.lower() == 'true':
            # Epistemic Uncertainty
            from lib.utils.mc_dropout_pytorch import get_monte_carlo_predictions

            def set_dropout(model, drop_rate=0.1):
                # changing only the DO in the 3D layers.
                for name, child in model.named_children():
                    if isinstance(child, torch.nn.Dropout3d):
                        log.info(f"{child} old DO rate: {child.p}")
                        child.p = drop_rate
                        log.info(f"{child} new DO rate: {child.p}")
                    set_dropout(child, drop_rate=drop_rate)

            import copy
            model_do_copy = copy.deepcopy(self.model)

            # setting new dropout value
            log.info('Changing dropout for Epistemic Uncertainty')
            set_dropout(model_do_copy, drop_rate=0.1)
            print('REVIEW TO MAKE SURE EPISTEMIC*********************')
            print('reviewing the dropout copy model')
            for name, child in model_do_copy.named_children():
                print(name, child)
            log.info('')

            df_val_epistemic_entropy = get_monte_carlo_predictions(data_loader=val_generator_eval,
                                                                   forward_passes=self.args.mc_do_num,
                                                                   model=model_do_copy,
                                                                   n_classes=2,
                                                                   n_samples=len(val_generator_eval))

            df_test_epistemic_entropy = get_monte_carlo_predictions(data_loader=test_generator_eval,
                                                                    forward_passes=self.args.mc_do_num,
                                                                    model=model_do_copy,
                                                                    n_classes=2,
                                                                    n_samples=len(test_generator_eval))

            epistemic_uncertainty_store_dir = '/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/Uncertainty/Epistemic/'

            # Saving validation Epistemic uncertainty calculations
            filename_val_epistemic_entropy = f"image_model_val_epistemic_do_entropy_" \
                                             f"{self.args.short_note}_{self.args.class_weights}_{self.args.single_fold + 1}_of_{self.args.cv_num}_" \
                                             f"seed_{self.args.manual_seed}_{self.args.time_stamp}.csv"
            val_epistemic_entropy_csv_dir = os.path.join(epistemic_uncertainty_store_dir,
                                                         filename_val_epistemic_entropy)
            df_val_epistemic_entropy.to_csv(val_epistemic_entropy_csv_dir)

            # Saving test Epistemic uncertainty calculations
            filename_test_epistemic_entropy = f"image_model_test_epistemic_do_entropy_" \
                                              f"{self.args.short_note}_{self.args.class_weights}_{self.args.single_fold + 1}_of_{self.args.cv_num}_" \
                                              f"seed_{self.args.manual_seed}_{self.args.time_stamp}.csv"
            test_epistemic_entropy_csv_dir = os.path.join(epistemic_uncertainty_store_dir,
                                                          filename_test_epistemic_entropy)
            df_test_epistemic_entropy.to_csv(test_epistemic_entropy_csv_dir)

        if self.args.do_tta.lower() == 'true':
            # Aleatoric Uncertainty
            from lib.utils.tta_pytorch import get_tta_predictions

            df_val_aleatoric_entropy = get_tta_predictions(data_loader=val_generator_eval_tta,
                                                           forward_passes=self.args.tta_num,
                                                           model=self.model,
                                                           n_classes=2,
                                                           n_samples=len(val_generator_eval_tta))

            df_test_aleatoric_entropy = get_tta_predictions(data_loader=test_generator_eval_tta,
                                                            forward_passes=self.args.tta_num,
                                                            model=self.model,
                                                            n_classes=2,
                                                            n_samples=len(test_generator_eval_tta))

            aleatoric_uncertainty_store_dir = '/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/Uncertainty/Aleatoric/'

            # Saving validation aleatoric uncertainty calculations
            filename_val_aleatoric_entropy = f"image_model_val_aleatoric_tta_entropy_" \
                                             f"{self.args.short_note}_{self.args.class_weights}_{self.args.single_fold + 1}_of_{self.args.cv_num}_" \
                                             f"seed_{self.args.manual_seed}_{self.args.time_stamp}.csv"
            val_aleatoric_entropy_csv_dir = os.path.join(aleatoric_uncertainty_store_dir,
                                                         filename_val_aleatoric_entropy)
            df_val_aleatoric_entropy.to_csv(val_aleatoric_entropy_csv_dir)

            # Saving test aleatoric uncertainty calculations
            filename_test_aleatoric_entropy = f"image_model_test_aleatoric_tta_entropy_" \
                                              f"{self.args.short_note}_{self.args.class_weights}_{self.args.single_fold + 1}_of_{self.args.cv_num}_" \
                                              f"seed_{self.args.manual_seed}_{self.args.time_stamp}.csv"
            test_aleatoric_entropy_csv_dir = os.path.join(aleatoric_uncertainty_store_dir,
                                                          filename_test_aleatoric_entropy)
            df_test_aleatoric_entropy.to_csv(test_aleatoric_entropy_csv_dir)

        # Evaluating image and clinic model performance on validation data

        target_ = []
        pred_ = []
        target2_ = []
        pred2_ = []

        target2_c = []
        pred2_c = []

        single_val_mrn = []
        for batch_idx, input_tuple in enumerate(val_generator_eval):
            with torch.no_grad():
                # Gathering input data; prepare_input sends to gpu
                mrn = list(input_tuple[2])
                single_val_mrn.append(mrn)
                # print(mrn_str)
                input_tuple = input_tuple[0:2]
                # print(input_tuple)
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)

                # Model make prediction
                pred = self.model(input_tensor)
                pred_.extend(torch.argmax(F.softmax(pred, dim=1), dim=1).cpu().numpy())
                pred2_.extend(np.array(F.softmax(pred, dim=1).cpu().numpy()).tolist())

                target_.extend(target.cpu().numpy())
                target2_.extend(np.array(F.one_hot(target).cpu().numpy()).tolist())

                x_val_clinic, y_val_clinic = \
                    clinical_data(args=self.args,
                                  val_mrn_list=mrn,
                                  train_mrn_list=self.args.train_mrn_list).load_val_data_batch(batch_size=1)

                y_pred_val_clinic = clinical_model_.predict_proba(x_val_clinic)
                target2_c.extend(y_val_clinic)
                pred2_c.extend(np.array(y_pred_val_clinic).tolist())

        single_val_mrn_df = pd.DataFrame(single_val_mrn, columns=['MRN'])
        single_val_mrn_df.to_csv(os.path.join('/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/catching_splits_img_clin',
                                              f"val_{self.args.single_fold + 1}_of_{self.args.cv_num}_seed_"
                                              f"{self.args.manual_seed}_{self.args.time_stamp}.csv"))
        # testing 12/24/2021
        # target2_ = np.array(target2_)
        target2_ = np.array(np.eye(self.args.n_classes)[target_])
        assert all(
            [np.shape(x)[0] == self.args.n_classes for x in
             target2_]), f"make sure one hot predictions is correct: {np.shape(target2_)}"
        pred2_ = np.array(pred2_)

        pred2_c_1 = [x[1] for x in pred2_c]

        ### Evaluating loaded data above (Validation, Train, Test)
        # printing precision, recall, and fscore for each class for validation data
        print('\nVALIDATION INFORMATION - IMAGE/CLINIC BASED MODEL\n')

        pred2_[:, 0] = np.add(np.dot(1 - pred2_[:, 1], alpha_image),
                              np.dot([1 - x for x in pred2_c_1], (1 - alpha_image)))

        pred2_[:, 1] = np.add(np.dot(pred2_[:, 1], alpha_image),
                              np.dot(pred2_c_1, (1 - alpha_image)))

        assert [np.argmax(np.array(x).tolist()) for x in target2_] == [np.argmax(x) for x in np.array(target2_c)]

        print('COMBINED IMAGE AND CLINIC MODEL - VALIDATION DATA')

        target_names = ['class ' + str(x) for x in range(self.args.n_classes)]
        print('\n', classification_report([np.argmax(np.array(x).tolist()) for x in target2_],
                                          np.argmax(pred2_, axis=-1),
                                          target_names=target_names))

        # PRED2_ HAS BEEN UPDATED IN THE CODE ABOVE AND REPRESENTS PREDICTION SCORE OF IMAGE AND CLINIC COMBINED

        # storing predictions for image clinic model
        image_clinic_val_df = pd.DataFrame()
        image_clinic_val_df['true'] = [np.argmax(np.array(x)) for x in target2_]
        image_clinic_val_df['pred'] = [x[1] for x in pred2_]
        filename = f"image_clinic_model_val_prediction_" \
                   f"{self.args.short_note}_{self.args.class_weights}_{self.args.single_fold + 1}_of_{self.args.cv_num}_" \
                   f"seed_{self.args.manual_seed}_{self.args.time_stamp}.csv"
        csv_dir = os.path.join(store_all_label_pred_dir, filename)
        image_clinic_val_df.to_csv(path_or_buf=csv_dir, index=False)

        if self.args.use_tb.lower() == 'true':
            fig_add = roc_auc_plot(
                np.array([to_categorical(np.argmax(np.array(x)), self.args.n_classes).tolist() for x in target2_]),
                # target2_c,
                pred2_,
                data_title='Validation Image/Clinic ROC')
            self.tb_logger.add_figure(f"Validation_ROC_Image_Clinic", figure=fig_add)

        # print(np.array([np.argmax(np.array(x)).tolist() for x in target2_]))

        auc_ = roc_auc_score(
            y_true=np.array([np.argmax(np.array(x)).tolist() for x in target2_]),
            y_score=pred2_[:, 1])
        print('combine image and clinic validation auc: ', auc_)

        # Evaluating image and clinic model performance on TEST data

        target_ = []
        pred_ = []
        target2_ = []
        pred2_ = []

        target2_c = []
        pred2_c = []

        single_test_mrn = []

        for batch_idx, input_tuple in enumerate(test_generator_eval):
            with torch.no_grad():
                # Gathering input data; prepare_input sends to gpu
                mrn = list(input_tuple[2])
                single_test_mrn.append(mrn)
                # print(mrn_str)
                input_tuple = input_tuple[0:2]
                # print(input_tuple)
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)

                # Model make prediction
                pred = self.model(input_tensor)
                pred_.extend(torch.argmax(F.softmax(pred, dim=1), dim=1).cpu().numpy())
                pred2_.extend(np.array(F.softmax(pred, dim=1).cpu().numpy()).tolist())

                target_.extend(target.cpu().numpy())
                target2_.extend(np.array(F.one_hot(target).cpu().numpy()).tolist())

                x_test_clinic, y_test_clinic, test_mrn_list_store_2 = \
                    clinical_data(args=self.args,
                                  test_mrn_list=mrn,
                                  train_mrn_list=self.args.train_mrn_list).load_test_data_batch(batch_size=1)

                y_pred_test_clinic = clinical_model_.predict_proba(x_test_clinic)
                target2_c.extend(y_test_clinic)
                pred2_c.extend(np.array(y_pred_test_clinic).tolist())

        single_test_mrn_df = pd.DataFrame(single_test_mrn, columns=['MRN'])
        single_test_mrn_df.to_csv(
            os.path.join('/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/catching_splits_img_clin',
                         f"test_{self.args.single_fold + 1}_of_{self.args.cv_num}_seed_"
                         f"{self.args.manual_seed}_{self.args.time_stamp}.csv"))

        # testing 12/24/2021
        # target2_ = np.array(target2_)
        target2_ = np.array(np.eye(self.args.n_classes)[target_])
        assert all(
            [np.shape(x)[0] == self.args.n_classes for x in
             target2_]), f"make sure one hot predictions is correct: {np.shape(target2_)}"
        pred2_ = np.array(pred2_)

        pred2_c_1 = [x[1] for x in pred2_c]

        #### for howard analysis for paired t-test. predictions need to match
        store_all_label_pred_dir = r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/saved_predictions/green_journal_revisions_feb2022'
        clinic_test2_df = pd.DataFrame()
        clinic_test2_df['true'] = [np.argmax(x) for x in target2_]
        clinic_test2_df['pred'] = [x for x in pred2_c_1]

        filename = f"clinical_model_test_sorted_prediction_" \
                   f"{self.args.short_note}_{self.args.class_weights}_{self.args.single_fold + 1}_of_{self.args.cv_num}_" \
                   f"seed_{self.args.manual_seed}_{self.args.time_stamp}.csv"
        csv_dir = os.path.join(store_all_label_pred_dir, filename)
        clinic_test2_df.to_csv(path_or_buf=csv_dir, index=False)
        ###

        ### Evaluating loaded data above (Validation, Train, Test)
        # printing precision, recall, and fscore for each class for validation data
        print('\nTest INFORMATION - IMAGE/CLINIC BASED MODEL\n')

        pred2_[:, 0] = np.add(np.dot(1 - pred2_[:, 1], alpha_image),
                              np.dot([1 - x for x in pred2_c_1], (1 - alpha_image)))

        pred2_[:, 1] = np.add(np.dot(pred2_[:, 1], alpha_image),
                              np.dot(pred2_c_1, (1 - alpha_image)))

        assert [np.argmax(np.array(x).tolist()) for x in target2_] == [np.argmax(x) for x in np.array(target2_c)]

        print('COMBINED IMAGE AND CLINIC MODEL - Test DATA')

        target_names = ['class ' + str(x) for x in range(self.args.n_classes)]
        print('\n', classification_report([np.argmax(np.array(x).tolist()) for x in target2_],
                                          np.argmax(pred2_, axis=-1),
                                          target_names=target_names))

        # PRED2_ HAS BEEN UPDATED IN THE CODE ABOVE AND REPRESENTS PREDICTION SCORE OF IMAGE AND CLINIC COMBINED
        # storing predictions for image clinic model
        image_clinic_test_df = pd.DataFrame()
        image_clinic_test_df['true'] = [np.argmax(np.array(x)) for x in target2_]
        image_clinic_test_df['pred'] = [x[1] for x in pred2_]
        filename = f"image_clinic_model_test_prediction_" \
                   f"{self.args.short_note}_{self.args.class_weights}_{self.args.single_fold + 1}_of_{self.args.cv_num}_" \
                   f"seed_{self.args.manual_seed}_{self.args.time_stamp}.csv"
        csv_dir = os.path.join(store_all_label_pred_dir, filename)
        image_clinic_test_df.to_csv(path_or_buf=csv_dir, index=False)

        if self.args.use_tb.lower() == 'true':
            fig_add = roc_auc_plot(
                np.array([to_categorical(np.argmax(np.array(x)), self.args.n_classes).tolist() for x in target2_]),
                # target2_c,
                pred2_,
                data_title='Test Image/Clinic ROC')
            self.tb_logger.add_figure(f"Test_ROC_Image_Clinic", figure=fig_add)

        auc_ = roc_auc_score(
            y_true=np.array([np.argmax(np.array(x)).tolist() for x in target2_]),
            y_score=pred2_[:, 1])
        print('combine image and clinic test auc: ', auc_)

        if self.args.use_tb.lower() == 'true':
            self.tb_logger.close()

    def train_epoch_alex(self, epoch):

        # Creates once at the beginning of training

        def time_report(initial_time, time_name):
            get_time_diff = time.gmtime(time.time() - initial_time)
            readable_time = time.strftime("%M:%S", get_time_diff)
            print(f"{time_name} time: {readable_time} (min:seconds)")
            del get_time_diff
            del readable_time

        epoch_start_time = time.time()
        self.model.train()

        # Storing epoch values obtained from batch calculations
        loss_cum = []
        auc_cum = []
        prauc_cum = []
        spec_cum = []
        sens_cum = []

        log.info('-------------------------------------------------------------------------------------------')

        for batch_idx, input_tuple in enumerate(self.train_data_loader):
            # Gathering input data; prepare_input sends to gpu
            input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)

            # may need to turn on if want to train fully but off for transfer learning
            # input_tensor.requires_grad = True

            # Model make prediction
            pred = self.model(input_tensor)

            # calculating loss and metrics
            loss = self.criterion_pre(pred, target.long().view(-1))

            # need to calculate gradient
            self.model.zero_grad()

            loss.backward()

            def clip_gradient(optimizer, grad_clip):
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            param.grad.data.clamp_(-grad_clip, grad_clip)

            clip_gradient(self.optimizer, 5)
            self.optimizer.step()

            # Calculating and appending
            with torch.no_grad():
                auc = calculate_auc(pred, target)
                prauc = calculate_prauc(pred, target)
                spec, sens = dr_friendly_measures(pred, target)

                # storing loss and metrics
                loss_cum.append(loss.item())
                auc_cum.append(auc)
                prauc_cum.append(prauc)
                spec_cum.append(spec)
                sens_cum.append(sens)

                if (batch_idx + 1) % self.print_batch_spacing == 0:
                    log.info(f"\tBatch {batch_idx + 1} of {len(self.train_data_loader)}")
                    log.info(
                        f"\tLoss: {loss.item()}, PRAUC: {prauc}, AUC: {auc}, Sensitivity: {sens}, Specificity: {spec}")
                    log.info('\t**************************************************************************')
                else:
                    pass

            # self.tb_logger.add_pr_curve('training_PR_curve', target_, pred_, global_step=0)

            # test_str = f"training_loss-{self.args.short_note}"
            if self.args.use_tb.lower() == 'true':
                self.tb_logger.add_scalar(f"training_loss", loss.item(), self.train_count)
                self.tb_logger.add_scalar(f"training_auc", auc, self.train_count)
                self.tb_logger.add_scalar(f"training_prauc", prauc, self.train_count)
                self.tb_logger.add_scalar(f"training_sensitivity", sens, self.train_count)
                self.tb_logger.add_scalar(f"training_specificity", spec, self.train_count)

                self.tb_logger.flush()

            # train count for tensorboard logging
            self.train_count += 1

            if not self.args.ci_test:
                # save model
                if batch_idx == 0 and (epoch * len(self.train_data_loader)) != 0 and (
                        epoch * len(self.train_data_loader)) % self.save_interval == 0 and (
                        epoch > self.args.n_epochs - 40):

                    # if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    model_save_path = '{}_epoch_{}_batch_{}_fold_{}.pth.tar'.format(self.args.save_folder, epoch,
                                                                                    batch_idx, self.args.true_cv_count)
                    model_save_dir = os.path.dirname(model_save_path)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)

                    log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_idx))
                    torch.save({
                        'epoch': epoch,
                        'batch_id': batch_idx,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()},
                        model_save_path)

        # Calculating time per epoch
        ty_res = time.gmtime(time.time() - epoch_start_time)
        res = time.strftime("%M:%S", ty_res)
        log.info(
            f"Summary-----Loss: {np.round(sum(loss_cum) / len(loss_cum), 4)}, PRAUC: {np.round(np.nanmedian(prauc_cum), 4)}, "
            f"AUC: {np.round(np.nanmedian(auc_cum), 4)},\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tSensitivity: {np.round(np.nanmedian(sens_cum), 4)}, "
            f"Specificity: {np.round(np.nanmedian(spec_cum), 4)}\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t"
            f"Epoch: {res} (min:seconds), # nan: {sum(math.isnan(x) for x in auc_cum)}")
        log.info('-------------------------------------------------------------------------------------------')

    def validate_epoch_alex(self, epoch):
        self.model.eval()

        # Storing epoch values obtained from batch calculations
        loss_cum = []
        auc_cum = []
        prauc_cum = []
        spec_cum = []
        sens_cum = []

        # starting epoch timer
        epoch_start_time = time.time()

        log.info('-------------------------------------------------------------------------------------------')
        for batch_idx, input_tuple in enumerate(self.valid_data_loader):

            if (batch_idx + 1) % self.print_batch_spacing == 0:
                log.info('*************************************')
                log.info(f"\tBatch {batch_idx + 1} of {len(self.valid_data_loader)}")
            else:
                pass

            with torch.no_grad():
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                # input_tensor.requires_grad = False

                pred = self.model(input_tensor)

                loss = self.criterion_pre(pred, target.long().view(-1))
                auc = calculate_auc(pred, target)
                prauc = calculate_prauc(pred, target)
                spec, sens = dr_friendly_measures(pred, target)

                # storing loss and metrics
                loss_cum.append(loss.item())
                auc_cum.append(auc)
                prauc_cum.append(prauc)
                spec_cum.append(spec)
                sens_cum.append(sens)

                if (batch_idx + 1) % self.print_batch_spacing == 0:
                    # log.info('\t**************************************************************************')
                    log.info(f"\tBatch {batch_idx + 1} of {len(self.train_data_loader)}")
                    log.info(
                        f"\tLoss: {loss.item()}, PRAUC: {prauc}, AUC: {auc}, Sensitivity: {sens}, Specificity: {spec}")
                    log.info('\t**************************************************************************')
                else:
                    pass

                if self.args.use_tb.lower() == 'true':
                    self.tb_logger.add_scalar(f"val_loss", loss.item(), self.val_count)
                    self.tb_logger.add_scalar(f"val_auc", auc, self.val_count)
                    self.tb_logger.add_scalar(f"val_prauc", prauc, self.val_count)
                    self.tb_logger.add_scalar(f"val_sensitivity", sens, self.val_count)
                    self.tb_logger.add_scalar(f"val_specificity", spec, self.val_count)

                    self.tb_logger.flush()

                self.val_count += 1

        self.val_loss = sum(loss_cum) / len(loss_cum)

        ty_res = time.gmtime(time.time() - epoch_start_time)
        res = time.strftime("%M:%S", ty_res)
        log.info(
            f"Summary-----Loss: {np.round(sum(loss_cum) / len(loss_cum), 4)}, PRAUC: {np.round(np.nanmedian(prauc_cum), 4)}, "
            f"AUC: {np.round(np.nanmedian(auc_cum), 4)},\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tSensitivity: {np.round(np.nanmedian(sens_cum), 4)}, "
            f"Specificity: {np.round(np.nanmedian(spec_cum), 4)}\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t"
            f"Epoch: {res} (min:seconds), # nan: {sum(math.isnan(x) for x in auc_cum)}")
        log.info('-------------------------------------------------------------------------------------------')
