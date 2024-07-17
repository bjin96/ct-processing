"""Reading NIfTI header and sidecar JSON files"""
import json
import pickle
from pathlib import Path

import nibabel
import pandas as pd

from src.datasets.structure import DatasetStructure

ALL_NIFTI_HEADERS_FILE = Path('all_nifti_headers.pkl')
NIFTI_SELECTION_FILES = [
    Path('nifti_selection_files/00_nifti_metadata.pkl'),
    Path('nifti_selection_files/01_cleaned_nifti_metadata.pkl'),
    Path('nifti_selection_files/02_filtered_nifti_metadata.pkl'),
    Path('nifti_selection_files/03_registered_nifti_metadata.pkl'),
    Path('nifti_selection_files/04_ssim_qc_nifti_metadata.pkl'),
    Path('nifti_selection_files/05_superimpose_qc_nifti_metadata.pkl'),
    Path('nifti_selection_files/06_at_least_one_roi_metadata.pkl')
]


def get_corresponding_json(nifti_file, json_files):
    matches = []
    for json_file in json_files:
        if str(json_file.name).replace('.json', '') in str(nifti_file.name):
            matches.append(json_file)
    if len(matches) > 1:
        longest_match = Path('')
        for match in matches:
            if len(str(match.name)) > len(str(longest_match.name)):
                longest_match = match
        return longest_match
    elif len(matches) == 1:
        return matches[0]
    else:
        raise ValueError(f'No matching JSON file found for {nifti_file=}. {json_files=}')


def read_json_sidecar_files(
        base_path: Path,
        dataset_folder_structure: DatasetStructure,
):
    relative_dcm_folder_paths_json = base_path / dataset_folder_structure.DICOM_FOLDERS_JSON

    with open(relative_dcm_folder_paths_json, 'r') as file:
        nifti_folders = json.load(file)

    all_meta_data = {'patient_id': [], 'study_subdir': [], 'series_subdir': [], 'json_file': [], 'nifti_file': []}

    base_nifti_path = base_path / 'niftis'
    for index, nifti_folder in enumerate(nifti_folders):
        nifti_folder = Path(nifti_folder)
        if index % 100 == 0:
            print(f'{index=}')
        json_files = list((base_nifti_path / nifti_folder).glob('*.json'))
        nifti_files = (base_nifti_path / nifti_folder).glob('*.nii.gz')
        for nifti_file in nifti_files:
            if str(Path(nifti_file.name)).startswith('._'):
                continue

            json_path = get_corresponding_json(nifti_file, json_files)
            with open(json_path, 'r') as file:
                meta_data = json.load(file)
            for key, value in dict(meta_data).items():
                if key not in all_meta_data.keys():
                    all_meta_data[key] = [None for _ in range(len(all_meta_data['patient_id']))]
                while len(all_meta_data[key]) < len(all_meta_data['patient_id']):
                    all_meta_data[key].append(None)
                if value != '':
                    all_meta_data[key].append(value)
                else:
                    all_meta_data[key].append(None)
            image_series_identifier = dataset_folder_structure.get_identifiers_from_folder_structure(nifti_folder)
            all_meta_data['patient_id'].append(image_series_identifier.patient_id)
            all_meta_data['study_subdir'].append(image_series_identifier.study_id)
            all_meta_data['series_subdir'].append(image_series_identifier.series_id)
            all_meta_data['json_file'].append(Path(json_path).name)
            all_meta_data['nifti_file'].append(Path(nifti_file).name)

    for key, value in all_meta_data.items():
        while len(value) < len(all_meta_data['patient_id']):
            all_meta_data[key].append(None)

    all_meta_data_df = pd.DataFrame(all_meta_data)
    (base_path / NIFTI_SELECTION_FILES[0]).parent.mkdir(parents=False, exist_ok=True)
    all_meta_data_df.to_pickle(base_path / NIFTI_SELECTION_FILES[0])


def get_nifti_headers(
        base_path: Path
) -> dict:
    nifti_headers_pkl_file = base_path / ALL_NIFTI_HEADERS_FILE
    if not nifti_headers_pkl_file.exists():
        return {}

    with open(nifti_headers_pkl_file, 'rb') as file:
        nifti_headers = pickle.load(file)

    return nifti_headers


def read_nifti_headers(
        base_path: Path,
        dataset_folder_structure: DatasetStructure
):
    nifti_headers_pkl_file = base_path / ALL_NIFTI_HEADERS_FILE
    nifti_info_file = base_path / NIFTI_SELECTION_FILES[0]

    all_headers = get_nifti_headers(base_path)
    all_niftis = pd.read_pickle(nifti_info_file)
    for index, row in all_niftis.iterrows():
        if index % 100 == 0:
            print(f'index: {index}/{len(all_niftis.index)}')

        nifti_path = dataset_folder_structure.create_path_from_csv_row(
            base_path / 'niftis',
            row,
            'nifti_file'
        )

        if str(nifti_path) not in all_headers.keys():
            image = nibabel.load(nifti_path)
            all_headers[str(nifti_path)] = image.header

    with open(nifti_headers_pkl_file, 'wb') as file:
        pickle.dump(all_headers, file)


def _add_voxel_size(df, base_path: Path, dataset_folder_structure: DatasetStructure):
    voxel_sizes = []
    nifti_headers = get_nifti_headers(base_path)

    for index, row in df.iterrows():
        file_path = dataset_folder_structure.create_path_from_csv_row(base_path / 'niftis', row, 'nifti_file')
        zooms = nifti_headers[str(file_path)].get_zooms()
        voxel_sizes.append(zooms)

    df['voxel_size'] = voxel_sizes
    return df
