#!/bin/bash

#SBATCH --job-name=test2
#SBATCH --output=Logs/TFL_cR50_excl20pe_trainrn_1e-2_cw_2-0_aug0-7_cv5_03222022_s200_test2.log

#SBATCH --partition=dgxq
#SBATCH --gres=gpu:dgx:1

#SBATCH --ntasks=1
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=64

srun python Train_Transfer_Clinic_2_TrueCV.py \
--gpu_id 0 \
--is_transfer True \
--is_classi True \
--in_modality 3 \
--n_epochs 120 \
--manual_seed 200 \
--batch_size 10 \
--input_H 150 \
--input_W 80 \
--input_D 80 \
--model resnet \
--model_depth 50 \
--num_workers 0 \
--pretrain_path lib/Models/pretrain/resnet_50_23dataset.pth \
--resnet_shortcut B \
--new_layer_names avgpool dropout fc \
\
--use_tb "false" \
\
--train_dataset_path "/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Data/ft_data_5/store_img_crop_npy/" \
--train_label_path "/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Data/ft_data_5/outcome/" \
--val_dataset_path "/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Data/ft_data_5/store_img_crop_npy/" \
--val_label_path "/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Data/ft_data_5/outcome/" \
--test_dataset_path "/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Data/ft_data_5/store_img_crop_npy/" \
--test_label_path "/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Data/ft_data_5/outcome/" \
--min_max_key_path "/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Data/ft_data_5/min_max_key_train_active_norm/" \
--fraction_key_path "/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Data/ft_data_5/fraction_key" \
\
--clinical_data_path '/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Data/ft_data_5/clinical_data' \
--clinical_data_filename 'processed_unnormalized_data_350.csv' \
--clinical_model_type 'lr' \
\
--short_note "c_res50_150_80_80_TL0.01_excl20pe_aug0.7_ep1_03222022_s200_test2" \
--exclude_mrn 'true' \
--exclude_mrn_filename 'less_than_20_days_physician_edit.csv' \
--exclude_mrn_path '/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Data/ft_data_5/exclude_mrn_feeding_tube_days/' \
--resnet_lr_factor 0.01 \
--class_weights "[1.0, 2.0]" \
--cv_num 5 \
--augmentation 'true' \
--do_normalization 'true' \
--aug_percent 0.70

