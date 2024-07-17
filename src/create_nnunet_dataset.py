import json
import shutil
from pathlib import Path
from typing import Dict, List

from src.create_region_of_interest import create_region_of_interest, create_annotation_region_of_interest, \
    run_function_for_list
from src.ct_windowing import DEFAULT_CT_RANGE, window_ct_scan
from src.region_of_interest_definition import RegionOfInterest


def create_ica_roi_dataset(
        dataset_json: Path,
        nnunet_dataset_images_path: Path,
        nnunet_dataset_labels_path: Path,
        roi_mask_file: Path,
        from_nnunet_mapping: Path,
        to_nnunet_mapping: Path,
        nnunet_dataset_name: str = 'ICAROIS',
        roi_image_suffix: str = '_ROI_ICA_cavernous.nii.gz',
        voi_suffix: str = '_ICA_voi.nii.gz',
):
    with open(dataset_json) as file:
        dataset_files = json.load(file)

    _create_ica_roi_dataset(
        dataset_files=dataset_files,
        base_path=dataset_json.parent,
        nnunet_dataset_images_path=nnunet_dataset_images_path,
        nnunet_dataset_labels_path=nnunet_dataset_labels_path,
        roi_mask_file=roi_mask_file,
        from_nnunet_mapping=from_nnunet_mapping,
        to_nnunet_mapping=to_nnunet_mapping,
        nnunet_dataset_name=nnunet_dataset_name,
        roi_image_suffix=roi_image_suffix,
    )


def _create_ica_roi_dataset(
        dataset_files: List[Dict[str, str]],
        base_path: Path,
        nnunet_dataset_images_path: Path,
        nnunet_dataset_labels_path: Path,
        roi_mask_file: Path,
        from_nnunet_mapping: Path,
        to_nnunet_mapping: Path,
        nnunet_dataset_name: str = 'ICAROIS',
        roi_image_suffix: str = '_ROI_ICA_cavernous.nii.gz',
):
    nnunet_dataset_file_map = {}
    reverse_nnunet_dataset_file_map = {}

    for index, case in enumerate(dataset_files):
        original_image_series_path = Path(base_path) / case['image']
        original_annotation_path = Path(base_path) / case['mask']

        roi_image_series_path = Path(str(original_image_series_path).replace('.nii.gz', roi_image_suffix))

        if not roi_image_series_path.exists():
            print(f'{roi_image_series_path} does not exist')
            continue

        # The second id is always 000, because CT only has one channel. Could potentially give different windows.
        # Attention: We are now giving a different window.
        nnunet_image_name = f'{nnunet_dataset_name}_{index:03}_0000.nii.gz'
        nnunet_label_name = f'{nnunet_dataset_name}_{index:03}.nii.gz'

        nnunet_windowed_image_name = f'{nnunet_dataset_name}_{index:03}_0001.nii.gz'

        nnunet_image_path = nnunet_dataset_images_path / nnunet_image_name
        nnunet_windowed_image_path = nnunet_dataset_images_path / nnunet_windowed_image_name
        nnunet_label_path = nnunet_dataset_labels_path / nnunet_label_name

        # Create file map to join nnunet file to DICOM information
        nnunet_dataset_file_map[str(nnunet_image_path)] = str(original_image_series_path)
        reverse_nnunet_dataset_file_map[str(original_image_series_path)] = str(nnunet_image_path)
        create_annotation_region_of_interest(
            mask_path=roi_mask_file,
            ct_scan_path=original_image_series_path,
            annotation_path=original_annotation_path,
            output_path=nnunet_label_path,
            window=(0, 300)
        )
        create_region_of_interest(
            mask_path=roi_mask_file,
            ct_scan_path=original_image_series_path,
            output_path=nnunet_windowed_image_path,
            window=(0, 300),
            original_ct_scan_path=original_image_series_path
        )
        shutil.copy(roi_image_series_path, nnunet_image_path)

    with open(from_nnunet_mapping, 'w') as file:
        json.dump(nnunet_dataset_file_map, file)

    with open(to_nnunet_mapping, 'w') as file:
        json.dump(reverse_nnunet_dataset_file_map, file)


def create_rss_regions_of_interest(
        valid_cases_json: Path,
        mask_roi_cavernous: Path
):
    """
    Create regions of interest for the Rotterdam study scans.

    Args:
        valid_cases_json: JSON file listing the niftis from which to crop the regions of interest.
        mask_roi_cavernous: Path to the binary mask file describing the region of interest.
    """
    with open(valid_cases_json, 'r') as f:
        nifti_files = json.load(f)
    args_list = []
    for case in nifti_files:
        ct_scan_path = Path(case['image'])
        args_list.append([
            mask_roi_cavernous,
            ct_scan_path,
            ct_scan_path.parent / ct_scan_path.name.replace('.nii.gz', f'_{RegionOfInterest.CAROTID.value}.nii.gz'),
            DEFAULT_CT_RANGE,
        ])
    run_function_for_list(create_region_of_interest, args_list)


def create_nnunet_from_preprocessed(
        dataset_json: Path,
        nnunet_dataset_images_path: Path,
        nnunet_dataset_labels_path: Path,
        from_nnunet_mapping: Path,
        to_nnunet_mapping: Path,
        nnunet_dataset_name: str = 'ICAROIS_TRAUMA',
):
    with open(dataset_json) as file:
        dataset_files = json.load(file)

    nnunet_dataset_file_map = {}
    reverse_nnunet_dataset_file_map = {}

    for index, case in enumerate(dataset_files):
        roi_image_series_path = case['image']
        roi_label_series_path = case['mask']

        # The second id is always 000, because CT only has one channel. Could potentially give different windows.
        # Attention: We are now giving a different window.
        nnunet_image_name = f'{nnunet_dataset_name}_{index:03}_0000.nii.gz'
        nnunet_label_name = f'{nnunet_dataset_name}_{index:03}.nii.gz'

        nnunet_windowed_image_name = f'{nnunet_dataset_name}_{index:03}_0001.nii.gz'

        nnunet_image_path = nnunet_dataset_images_path / nnunet_image_name
        nnunet_windowed_image_path = nnunet_dataset_images_path / nnunet_windowed_image_name
        nnunet_label_path = nnunet_dataset_labels_path / nnunet_label_name

        # Create file map to join nnunet file to DICOM information
        nnunet_dataset_file_map[str(nnunet_image_path)] = str(roi_image_series_path)
        reverse_nnunet_dataset_file_map[str(roi_image_series_path)] = str(nnunet_image_path)

        window_ct_scan(roi_image_series_path, nnunet_windowed_image_path, 0, 300)
        shutil.copy(roi_label_series_path, nnunet_label_path)
        shutil.copy(roi_image_series_path, nnunet_image_path)

    with open(from_nnunet_mapping, 'w') as file:
        json.dump(nnunet_dataset_file_map, file)

    with open(to_nnunet_mapping, 'w') as file:
        json.dump(reverse_nnunet_dataset_file_map, file)
