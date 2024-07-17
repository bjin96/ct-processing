#!/bin/bash
#SBATCH --mail-user=your@mail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/path/to/output.log
#SBATCH --error=/path/to/error.log
#SBATCH --job-name=your-job-name
#SBATCH --partition=your-partition
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --mem=11G

CT_PROCESSING_CODE=/path/to/ct-processing
BASE_PATH=/path/to/working_directory
DATASET=your_dataset_name
DICOM_DIRECTORY=/path/to/dicom_directory
CLINICAL_INFORMATION_FILE_PATH=/path/to/clinical_information_file
MICROMAMBA=/path/to/bin/micromamba

eval "$($MICROMAMBA shell hook --shell=bash)"
micromamba activate ct-processing

python $CT_PROCESSING_CODE/run.py create-dcm-folder-list \
--dicom_directory=$DICOM_DIRECTORY \
--base_path=$BASE_PATH \
--dataset=$DATASET

python $CT_PROCESSING_CODE/run.py convert-dcm-to-nifti \
--base_path=$BASE_PATH \
--dicom_directory=$DICOM_DIRECTORY

python $CT_PROCESSING_CODE/run.py read-nifti-information \
--base_path=$BASE_PATH \
--dataset=$DATASET

python $CT_PROCESSING_CODE/run.py clean-nifti-information \
--base_path=$BASE_PATH \

python $CT_PROCESSING_CODE/run.py filter-nifti-information \
--base_path=$BASE_PATH \
--dataset=$DATASET

python $CT_PROCESSING_CODE/run.py determine-patient-age-group \
--base_path=$BASE_PATH \
--dataset=$DATASET \
--clinical_information_file_path=$CLINICAL_INFORMATION_FILE_PATH

python $CT_PROCESSING_CODE/run.py register-to-template \
--base_path=$BASE_PATH \
--dataset=$DATASET

python $CT_PROCESSING_CODE/run.py check-registration \
--base_path=$BASE_PATH \
--dataset=$DATASET

python $CT_PROCESSING_CODE/run.py invert-registration-matrices \
--base_path=$BASE_PATH \
--dataset=$DATASET

python $CT_PROCESSING_CODE/run.py calculate-scan-completeness \
--base_path=$BASE_PATH \
--dataset=$DATASET

python $CT_PROCESSING_CODE/run.py control-similarity-to-template \
--base_path=$BASE_PATH \
--dataset=$DATASET

python $CT_PROCESSING_CODE/run.py control-low-similarity-series \
--base_path=$BASE_PATH \
--dataset=$DATASET

python $CT_PROCESSING_CODE/run.py control-superimposed-series \
--base_path=$BASE_PATH \

python $CT_PROCESSING_CODE/run.py create-image-series-roi-masks \
--base_path=$BASE_PATH \

python $CT_PROCESSING_CODE/run.py create-rois \
--base_path=$BASE_PATH \

python $CT_PROCESSING_CODE/run.py create-ground-truth-rois \
--base_path=$BASE_PATH

python $CT_PROCESSING_CODE/run.py check-contains-roi \
--base_path=$BASE_PATH \
--dataset=$DATASET
