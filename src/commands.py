from pathlib import Path
from typing import List, Tuple

import click

from src.create_nnunet_dataset import create_nnunet_from_preprocessed
from src.region_of_interest_definition import RegionOfInterest
from src.completeness_group_similarity import plot_all_group_similarities
from src.co_registration import run_register_to_template, run_invert_registration_matrices
from src.convert_dicom_to_nifti import convert_all_dcms_to_niftis
from src.create_completeness_groups import create_completeness_groups
from src.create_region_of_interest import apply_transformation_matrices_to_roi, run_create_region_of_interests
from src.datasets.datasets import CtDatasets, dataset_to_structure
from src.image_similarity import calculate_template_similarity
from src.nifti_selection import clean_nifti_info, filter_nifti_files, check_registration_files, \
    manual_superimpose_check_results, manual_low_similarity_check_results, select_scans_with_at_least_one_roi
from src.read_nifti_information import read_json_sidecar_files, read_nifti_headers
from src.scan_completeness import run_calculate_completeness
from src.create_dataset_json import write_dataset_json


@click.command()
@click.option('--base_path', required=True, type=click.Path(path_type=Path))
@click.option('--dicom_directory', required=True, type=click.Path(path_type=Path))
def convert_dcm_to_nifti(
        base_path: Path,
        dicom_directory: Path,
):
    convert_all_dcms_to_niftis(
        base_path=base_path,
        dicom_path=dicom_directory
    )


@click.command()
@click.option('--base_path', required=True, type=click.Path(path_type=Path))
@click.option('--dataset', required=True, type=click.Choice([d.name.lower() for d in CtDatasets]))
def read_nifti_information(
        base_path: Path,
        dataset: str,
):
    dataset = CtDatasets(dataset)
    dataset_folder_structure = dataset_to_structure[dataset]

    read_json_sidecar_files(
        base_path=base_path,
        dataset_folder_structure=dataset_folder_structure,
    )

    read_nifti_headers(
        base_path=base_path,
        dataset_folder_structure=dataset_folder_structure,
    )


@click.command()
@click.option('--base_path', required=True, type=click.Path(path_type=Path))
@click.option('--column_missing_data_threshold', required=False, type=click.INT, default=5)
def clean_nifti_information(
        base_path: Path,
        column_missing_data_threshold: int
):
    clean_nifti_info(
        base_path=base_path,
        column_missing_data_threshold=column_missing_data_threshold,
    )


@click.command()
@click.option('--base_path', required=True, type=click.Path(path_type=Path))
@click.option('--dataset', required=True, type=click.Choice([d.name.lower() for d in CtDatasets]))
def filter_nifti_information(
        base_path: Path,
        dataset: str
):
    dataset = CtDatasets(dataset)
    dataset_folder_structure = dataset_to_structure[dataset]

    filter_nifti_files(
        base_path=base_path,
        dataset_folder_structure=dataset_folder_structure
    )


@click.command(name='determine-patient-age-group')
@click.option('--clinical_information_file_path', required=True, type=click.Path(path_type=Path))
@click.option('--base_path', required=True, type=click.Path(path_type=Path))
@click.option('--age_bin', required=False, multiple=True, type=click.FLOAT, default=[72.5],
              help='Thresholds between age groups. Important age templates from different ages are used.')
@click.option('--dataset', required=True, type=click.Choice([d.name.lower() for d in CtDatasets]))
def determine_patient_age_group(
        clinical_information_file_path: Path,
        base_path: Path,
        age_bin: List[float],
        dataset: str
):
    age_bins = [0, *sorted(age_bin), float('inf')]

    dataset = CtDatasets(dataset)
    dataset_folder_structure = dataset_to_structure[dataset]

    dataset_folder_structure.determine_age_group_from_xlsx(
        input_file_path=clinical_information_file_path,
        base_output_path=base_path,
        age_bins=age_bins
    )


@click.command(name='create-dcm-folder-list')
@click.option('--dicom_directory', required=True, type=click.Path(path_type=Path))
@click.option('--base_path', required=True, type=click.Path(path_type=Path))
@click.option('--dataset', required=True, type=click.Choice([d.name.lower() for d in CtDatasets]))
def create_dcm_folder_list(
        dicom_directory: Path,
        base_path: Path,
        dataset: str
):
    dataset = CtDatasets(dataset)
    dataset_folder_structure = dataset_to_structure[dataset]

    dataset_folder_structure.write_dicom_folders_to_file(
        base_path=base_path,
        dicom_directory=dicom_directory
    )


@click.command()
@click.option('--base_path', required=True, type=click.Path(path_type=Path))
@click.option('--dataset', required=True, type=click.Choice([d.name.lower() for d in CtDatasets]))
def register_to_template(
        base_path: Path,
        dataset: str
):
    dataset_structure = dataset_to_structure[CtDatasets(dataset)]
    run_register_to_template(
        base_path=base_path,
        dataset_structure=dataset_structure
    )


@click.command()
@click.option('--base_path', required=True, type=click.Path(path_type=Path))
@click.option('--dataset', required=True, type=click.Choice([d.name.lower() for d in CtDatasets]))
def invert_registration_matrices(
        base_path: Path,
        dataset: str
):
    dataset_structure = dataset_to_structure[CtDatasets(dataset)]
    run_invert_registration_matrices(
        base_path=base_path,
        dataset_structure=dataset_structure
    )


@click.command()
@click.option('--base_path', required=True, type=click.Path(path_type=Path))
@click.option('--dataset', required=True, type=click.Choice([d.name.lower() for d in CtDatasets]))
def check_registration(
        base_path: Path,
        dataset: str
):
    dataset_structure = dataset_to_structure[CtDatasets(dataset)]

    check_registration_files(
        base_path=base_path,
        dataset_folder_structure=dataset_structure
    )


@click.command()
@click.option('--base_path', required=True, type=click.Path(path_type=Path))
@click.option('--dataset', required=True, type=click.Choice([d.name.lower() for d in CtDatasets]))
def calculate_scan_completeness(
        base_path: Path,
        dataset: str
):
    dataset_structure = dataset_to_structure[CtDatasets(dataset)]

    completeness_information_files = run_calculate_completeness(
        base_path=base_path,
        dataset_structure=dataset_structure
    )

    for age_group, completeness_information_file in completeness_information_files.items():
        create_completeness_groups(
            completeness_information_json=completeness_information_file,
            output_file=completeness_information_file.parent / f'{age_group}_completeness_group.json'
        )


@click.command()
@click.option('--base_path', required=True, type=click.Path(path_type=Path))
@click.option('--dataset', required=True, type=click.Choice([d.name.lower() for d in CtDatasets]))
@click.option('--inspect_percentage', required=False, type=click.FLOAT, default=0.05)
@click.option('--automate', required=False, type=click.BOOL, default=False,
              help='Automatically filter the worst 5% of registrations and registrations that are suspected to have failed.')
def control_similarity_to_template(
        base_path: Path,
        dataset: str,
        inspect_percentage: float,
        automate: bool,
):
    dataset_structure = dataset_to_structure[CtDatasets(dataset)]

    calculate_template_similarity(
        base_path=base_path,
        dataset_structure=dataset_structure
    )

    plot_all_group_similarities(
        base_path=base_path,
        dataset_structure=dataset_structure,
        inspect_percentage=inspect_percentage,
        automate=automate
    )


@click.command()
@click.option('--base_path', required=True, type=click.Path(path_type=Path))
@click.option('--dataset', required=True, type=click.Choice([d.name.lower() for d in CtDatasets]))
def control_low_similarity_series(
        base_path: Path,
        dataset: str
):
    dataset_structure = dataset_to_structure[CtDatasets(dataset)]

    manual_low_similarity_check_results(
        base_path=base_path,
        dataset_structure=dataset_structure
    )


@click.command()
@click.option('--base_path', required=True, type=click.Path(path_type=Path))
def control_superimposed_series(
        base_path: Path,
):
    manual_superimpose_check_results(base_path=base_path)


@click.command()
@click.option('--base_path', required=True, type=click.Path(path_type=Path))
@click.option('--distinguish_left_right', required=False, default=False, type=click.Path(path_type=Path))
def create_image_series_roi_masks(
        base_path: Path,
        distinguish_left_right: bool
):
    apply_transformation_matrices_to_roi(base_path=base_path, distinguish_left_right=distinguish_left_right)


@click.command()
@click.option('--base_path', required=True, type=click.Path(path_type=Path))
@click.option('--distinguish_left_right', required=False, default=False, type=click.Path(path_type=Path))
def create_rois(
        base_path: Path,
        distinguish_left_right: bool
):
    run_create_region_of_interests(base_path=base_path, distinguish_left_right=distinguish_left_right)


@click.command()
@click.option('--base_path', required=True, type=click.Path(path_type=Path))
@click.option('--ground_truth_suffix', required=True, type=str, default='_ground_truth',
              help='Suffix of the ground truth image. Default: Input image is called "<ID>.nii.gz" and the ground truth'
                   'mask is then called "<ID>_ground_truth.nii.gz"')
def create_ground_truth_rois(
        base_path: Path,
        ground_truth_suffix: str,
):
    run_create_region_of_interests(base_path=base_path, ground_truth_suffix=ground_truth_suffix)


@click.command()
@click.option('--base_path', required=True, type=click.Path(path_type=Path))
@click.option('--dataset', required=True, type=click.Choice([d.name.lower() for d in CtDatasets]))
def check_contains_roi(
        base_path: Path,
        dataset: str
):
    dataset_structure = dataset_to_structure[CtDatasets(dataset)]
    select_scans_with_at_least_one_roi(
        base_path,
        dataset_structure
    )


@click.command()
@click.option('--base_path', required=True, type=click.Path(path_type=Path))
@click.option('--region_of_interest', multiple=True, required=True,
              type=click.Choice([roi.value for roi in RegionOfInterest]), default=(RegionOfInterest.CAROTID.value,))
def create_dataset_json(
        base_path: Path,
        region_of_interest: Tuple[str]
):
    regions_of_interest = [RegionOfInterest(roi) for roi in region_of_interest]
    write_dataset_json(
        regions_of_interest=regions_of_interest,
        base_path=base_path,
    )


@click.command()
@click.option('--dataset_json', required=True, type=click.Path(path_type=Path))
@click.option('--nnunet_dataset_path', required=True, type=click.Path(path_type=Path))
@click.option('--nnunet_mapping_path', required=True, type=click.Path(path_type=Path))
def create_nnunet_dataset(
        dataset_json: Path,
        nnunet_dataset_path: Path,
        nnunet_mapping_path: Path,
):
    create_nnunet_from_preprocessed(
        dataset_json=dataset_json,
        nnunet_dataset_images_path=nnunet_dataset_path / 'imagesTr',
        nnunet_dataset_labels_path=nnunet_dataset_path / 'labelsTr',
        from_nnunet_mapping=nnunet_mapping_path / 'from_nnunet_mapping.json',
        to_nnunet_mapping=nnunet_mapping_path / 'to_nnunet_mapping.json',
    )
