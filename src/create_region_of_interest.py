from multiprocessing import Pool
from pathlib import Path
from typing import Tuple, Callable, List

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image

from src.ct_windowing import DEFAULT_CT_RANGE
from src.read_annotation_file import filter_annotation_mask
from src.datasets.structure import DatasetStructure
from src.read_nifti_information import _add_voxel_size, NIFTI_SELECTION_FILES
from src.co_registration import apply_list_of_matrices
from src.ct_windowing import binarize_ct_scan
from src.region_of_interest_definition import REGION_OF_INTEREST_SIZE, RegionOfInterest, get_template_roi_regions


def create_annotation_region_of_interest(
        mask_path: Path,
        annotation_path: Path,
        output_path: Path,
        window: Tuple[int, int],
        ct_scan_path: Path,
):
    """
    Create a region of interest (defined by mask_path) for an annotation of an entire CT scan. The annotation will be
    filtered according to a threshold for the corresponding voxel HU. The threshold is defined in
    filter_annotation_mask.

    Args:
        mask_path: Path to the mask that defines the region of interest.
        annotation_path: Path to the annotation of an entire CT scan.
        ct_scan_path: Path to the CT scan.
        window: Parameter to ensure consistency with create_region_of_interest.
        output_path: Path where the cropped annotation region of interest will be saved.
    """
    annotation = image.load_img(annotation_path)
    ct_scan = image.load_img(ct_scan_path)
    mask = image.load_img(mask_path)

    annotation = filter_annotation_mask(annotation, ct_scan)

    return _create_region_of_interest(annotation, mask, output_path, None)


def run_create_region_of_interests(
        base_path: Path,
        ground_truth_suffix: str | None = None,
        distinguish_left_right: bool = False,
):
    registered_nifti_files_info = pd.read_pickle(base_path / NIFTI_SELECTION_FILES[5])
    args_list = []
    for index, row in list(registered_nifti_files_info.iterrows()):
        template_regions = get_template_roi_regions(age_group=row['age_group'])
        ct_scan_path = base_path / 'niftis' / row['registered_file_path']
        original_ct_scan_path = base_path / 'niftis' / row['registered_file_path']

        if ground_truth_suffix is not None:
            ct_scan_path = Path(str(ct_scan_path).replace('.nii.gz', f'{ground_truth_suffix}.nii.gz'))

        for region_name, _ in template_regions.items():
            mask_path = base_path / 'registered_niftis' \
                        / (row['registered_file_path'].replace('.nii.gz', f'_mask_{region_name}.nii.gz'))
            output_path = base_path / 'registered_niftis' \
                          / (row['registered_file_path'].replace('.nii.gz', f'_{region_name}.nii.gz'))

            if ground_truth_suffix is not None:
                output_path = Path(str(output_path).replace('.nii.gz', f'{ground_truth_suffix}.nii.gz'))

            args_list.append((
                mask_path,
                ct_scan_path,
                output_path,
                DEFAULT_CT_RANGE,
                original_ct_scan_path
            ))

            lr_mask_path = Path(str(mask_path).replace('_mask', '_LR_mask'))
            if distinguish_left_right and lr_mask_path.exists():
                output_path = base_path / 'registered_niftis' \
                              / (row['registered_file_path'].replace('.nii.gz', f'_LR_{region_name}.nii.gz'))
                args_list.append((
                    mask_path,
                    lr_mask_path,
                    output_path,
                    DEFAULT_CT_RANGE,
                    original_ct_scan_path
                ))

    if ground_truth_suffix is not None:
        results = run_function_for_list(create_annotation_region_of_interest, args_list)
    else:
        print(args_list)
        results = run_function_for_list(create_region_of_interest, args_list)
    if ground_truth_suffix is None:
        pd.DataFrame(results, columns=['roi_path', 'roi_size', 'image_size', 'roi_slice_count']).to_csv(
            base_path / DatasetStructure.REGION_OF_INTEREST_SIZES_CSV, index=False)


def create_region_of_interest(
        mask_path: Path,
        ct_scan_path: Path,
        output_path: Path,
        window: Tuple[int, int],
        original_ct_scan_path: Path,
):
    """
    Create a region of interest based on a mask and a CT scan.

    Args:
        ct_scan_path: Path to the CT scan to be cropped to the region of interest.
        mask_path: Path to region of interest mask.
        output_path: Path to where the region of interest will be saved.
        window: Window for the region of interest.
        original_ct_scan_path: Parameter to ensure consistency with create_annotation_region_of_interest.
    """
    print(f'Creating region of interest {output_path}')
    ct_scan = image.load_img(ct_scan_path)
    mask = image.load_img(mask_path)

    return _create_region_of_interest(ct_scan, mask, output_path, window)


def _create_region_of_interest(ct_scan, mask, output_path, window):
    scan_data = ct_scan.get_fdata()
    raw_mask_data = mask.get_fdata()
    mask_data = binarize_ct_scan(raw_mask_data, threshold=0.5)

    any_x = np.any(mask_data, axis=(1, 2))
    any_y = np.any(mask_data, axis=(0, 2))
    any_z = np.any(mask_data, axis=(0, 1))

    if np.any(any_x) and np.any(any_y) and np.any(any_z):
        x_min, x_max, = np.where(any_x)[0][[0, -1]]
        y_min, y_max, = np.where(any_y)[0][[0, -1]]
        z_min, z_max, = np.where(any_z)[0][[0, -1]]

        roi_data = scan_data[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]

        if window is not None:
            roi_data = roi_data.clip(window[0], window[1])

        # Should we blacken parts of the roi_data that is zero in the mask?
        # pro: regions should be more consistent, con: gantry tilt is showing in the image -> possibility of bias
        # roi_data = roi_data.where()

        roi_image = nib.Nifti1Image(roi_data, ct_scan.affine, ct_scan.header)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            roi_image.to_filename(output_path)
        return output_path, int(np.sum(mask_data)), int(np.prod(mask_data.shape)), np.sum(any_z)
    print('Did not create region of interest, because region was not found in the CT image series.')
    return output_path, 0, int(np.prod(mask_data.shape)), 0


def run_function_for_list(function: Callable, args_list: List):
    with Pool(processes=8) as p:
        return p.starmap(function, args_list)


def select_regions_of_interest(registered_nifti_info, base_path, dataset_structure):
    registered_nifti_info = _add_voxel_size(registered_nifti_info, base_path, dataset_structure)

    roi_info = pd.read_csv(base_path / DatasetStructure.REGION_OF_INTEREST_SIZES_CSV)
    roi_info['roi_proportion'] = roi_info['roi_size'] / roi_info['image_size']
    roi_info.sort_values('roi_proportion', inplace=True, ascending=False)

    roi_info['registered_file_path'] = roi_info['roi_path'].apply(lambda x: str(dataset_structure.get_relative_path_from_absolute(Path(x))).split('_ROI_')[0] + '.nii.gz')

    merged = pd.merge(roi_info, registered_nifti_info, on=['registered_file_path'], how='inner')
    merged['absolute_roi_size'] = merged['roi_size'] * merged['voxel_size'].apply(lambda x: x[0]) * merged['voxel_size'].apply(lambda x: x[1]) * merged['voxel_size'].apply(lambda x: x[2])


    merged['region_of_interest_type'] = merged['roi_path'].apply(lambda x: [roi.value for roi in RegionOfInterest if roi.value in x][0])
    merged['template_absolute_roi_size'] = merged['roi_path'].apply(lambda x: REGION_OF_INTEREST_SIZE[[roi for roi in RegionOfInterest if roi.value in x][0]])
    merged['relative_roi_size'] = merged['absolute_roi_size'] / merged['template_absolute_roi_size']
    merged.sort_values('relative_roi_size', inplace=True, ascending=False)

    # Have at least 50% of the RoI present in the scan. Potentially questionable because of separate skull base/vault
    # where one part of the RoI is in the base and the other in the vault. But can be argued with volumetric
    # experiments, where there needs to be a certain number of slices to apply a 3d neural network.
    available_rois = merged.loc[merged['relative_roi_size'] > 0.5]
    available_ct_series = available_rois.groupby(['registered_file_path'])
    available_patients = available_rois.groupby('patient_id')

    print(f'Available CT-series: {len(available_ct_series)}')
    print(f'Available ROIs: {len(available_rois)}')
    print(f'Available patients: {len(available_patients)}')

    return available_rois


def apply_transformation_matrices_to_roi(base_path: Path, distinguish_left_right: bool = False):
    registered_nifti_files_info = pd.read_pickle(base_path / NIFTI_SELECTION_FILES[5])
    args_list = []
    for index, row in registered_nifti_files_info.iterrows():
        matrix_path = base_path / 'registered_niftis' / row['registered_file_path'].replace('.nii.gz', '.mat')
        inverse_matrix_path = matrix_path.parent / ('inverted_' + str(matrix_path.name))
        template_regions = get_template_roi_regions(age_group=row['age_group'])
        for template_region in template_regions.values():
            reference_volume = base_path / 'niftis' / row['registered_file_path']
            input_volume = base_path / template_region
            output_path = matrix_path.parent / (str(matrix_path.stem) + '_mask_' + str(input_volume.name))
            args_list.append((
                reference_volume,
                input_volume,
                inverse_matrix_path,
                output_path
            ))
            if distinguish_left_right:
                input_volume = Path(str(input_volume).replace('.nii.gz', '_LR.nii.gz'))
                output_path = Path(str(output_path).replace('_mask', '_LR_mask'))
                args_list.append((
                    reference_volume,
                    input_volume,
                    inverse_matrix_path,
                    output_path
                ))
    apply_list_of_matrices(args_list)
