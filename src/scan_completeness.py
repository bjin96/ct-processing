"""Calculating scan completeness, relates to 'information presence' in the MICCAI workshop paper."""
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from nilearn import image

from src.datasets.structure import DatasetStructure
from src.read_nifti_information import NIFTI_SELECTION_FILES


def calculate_completeness(image_path):
    """
    Calculate a percentage of the template that is covered by the scan by counting zero pixels along the z-axis to
    determine which parts of the template are not covered by the scan.

    Args:
        image_path: Scan that was previously registered to a template.

    Return:
        List of completeness percentages (for each slice along the z-axis percentage of coverage)
    """
    image_data = image.load_img(image_path).get_fdata()
    return count_zero_pixels(image_data)


def count_zero_pixels(image_data):
    if len(image_data.shape) != 3:
        raise ValueError('Image data should have exactly 3 dimensions')

    # Define minimum pixel intensities as lowest 5% of pixels.
    range_percentage = 0.05
    attenuated_minimum_intensity = image_data.min() + (image_data.max() - image_data.min()) * range_percentage

    attenuated_minimum_pixels = np.sum(image_data < attenuated_minimum_intensity, axis=(0, 1))

    # Invert for better intuition.
    return 1 - (attenuated_minimum_pixels / (image_data.shape[0] * image_data.shape[1]))


def calculate_completeness_of_list(image_path_list, output_json_path):
    completeness_information = {}

    for index, image_path in enumerate(image_path_list):
        if index % 100 == 0:
            print(f'{index}/{len(image_path_list)}')
        try:
            completeness_percentage = calculate_completeness(image_path)
            completeness_information[str(image_path)] = list(completeness_percentage)
        except ValueError as e:
            print(f'Could not calculate completeness of {image_path} with {e}')
        except EOFError as e:
            print(f'Could not calculate completeness of {image_path} with {e}')

    with open(output_json_path, 'w') as file:
        json.dump(completeness_information, file)


def run_calculate_completeness(
        base_path: Path,
        dataset_structure: DatasetStructure
) -> Dict[int, Path]:
    completeness_information_files = {}

    paths_by_age_group = path_for_templates_pickle(
        pickle_path=base_path / NIFTI_SELECTION_FILES[3],
        root_folder_path=base_path / 'registered_niftis',
        dataset_structure=dataset_structure
    )

    completeness_information_folder = base_path / dataset_structure.COMPLETENESS_INFORMATION_PATH
    completeness_information_folder.mkdir(parents=True, exist_ok=True)

    for age_group, image_series_paths in paths_by_age_group.items():
        completeness_information_file = completeness_information_folder / f'{age_group}_completeness_information.json'
        completeness_information_files[age_group] = completeness_information_file
        calculate_completeness_of_list(
            image_series_paths,
            completeness_information_file
        )

    return completeness_information_files


def path_for_templates_pickle(pickle_path, root_folder_path, dataset_structure: DatasetStructure):
    path_age_group_info = path_age_group_from_pickle(pickle_path, root_folder_path, dataset_structure)

    paths_by_age_group = {}

    for age_group in path_age_group_info['age_group'].unique():
        paths_by_age_group[age_group] = path_age_group_info.loc[path_age_group_info['age_group'] == age_group]['full_path'].to_list()

    return paths_by_age_group


def path_age_group_from_pickle(pickle_path, root_folder_path, dataset_structure: DatasetStructure):
    data = pd.read_pickle(pickle_path)
    data['file'] = data['registered_file']
    paths = []
    age_group = []

    for index, row in data.iterrows():
        paths.append(
            dataset_structure.create_path_from_csv_row(root_folder_path, row)
        )
        age_group.append(row['age_group'])

    data['full_path'] = paths
    data['age_group'] = age_group

    return data
