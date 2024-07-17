"""Create the dataset.json file with paths to image and ground truth NIfTIs."""
import json
from pathlib import Path
from typing import List

import pandas as pd

from src.read_nifti_information import NIFTI_SELECTION_FILES
from src.region_of_interest_definition import RegionOfInterest


def write_dataset_json(
        regions_of_interest: List[RegionOfInterest],
        base_path: Path
) -> None:
    """
    Reads pre-processed files and writes image and segmentation paths to JSON.

    Args:
        regions_of_interest (List[RegionOfInterest]): List of regions of interest to write to JSON.
        base_path (Path): Path to the folder where the JSON files are located.
    """
    nifti_info = pd.read_pickle(base_path / NIFTI_SELECTION_FILES[6])

    regions_of_interest = [roi.value for roi in regions_of_interest]
    nifti_info = nifti_info.loc[nifti_info['region_of_interest_type'].isin(regions_of_interest)]

    dataset_json = []
    for index, row in nifti_info.iterrows():
        dataset_json.append({
            'image': str(base_path / row['roi_path']),
            'mask': str(base_path / row['roi_path']).replace('.nii.gz', '_ground_truth.nii.gz'),
        })

    rois_string = '_'.join(regions_of_interest)
    with open(base_path / f'nifti_selection_files/dataset_{rois_string}.json', 'w') as file:
        json.dump(dataset_json, file, indent=4)