"""Creating a single file containing DICOM header information"""
import pandas as pd
import pydicom
from pydicom.multival import MultiValue


def create_dicom_information_pkl(dcm_root_path, output_csv_path):
    plain_cts = pd.read_csv(output_csv_path / '000PlainCT_folders.csv')

    all_meta_data = {'patient_id': [], 'study_subdir': [], 'series_subdir': [], 'dcm_file': []}

    for index, row in plain_cts.iterrows():
        if index % 100 == 0:
            print(f'{index=}')
        dcm_files = (
                    dcm_root_path / f'{str(row["patient_id"]).zfill(5)}/{row["study_subdir"]}/{row["series_subdir"]}').glob(
            '*.dcm')
        for dcm_path in dcm_files:
            meta_data = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
            for key, value in dict(meta_data).items():
                if value.name not in all_meta_data.keys():
                    all_meta_data[value.name] = [None for _ in range(len(all_meta_data['patient_id']))]
                while len(all_meta_data[value.name]) < len(all_meta_data['patient_id']):
                    all_meta_data[value.name].append(None)
                if value.value != '':
                    if isinstance(value.value, MultiValue):
                        all_meta_data[value.name].append(list(value.value))
                    else:
                        all_meta_data[value.name].append(value.value)
                else:
                    all_meta_data[value.name].append(None)
            all_meta_data['patient_id'].append(row['patient_id'])
            all_meta_data['study_subdir'].append(row['study_subdir'])
            all_meta_data['series_subdir'].append(row['series_subdir'])
            all_meta_data['dcm_file'].append(dcm_path)

    for key, value in all_meta_data.items():
        while len(value) < len(all_meta_data['patient_id']):
            all_meta_data[key].append(None)

    all_meta_data_df = pd.DataFrame(all_meta_data)
    all_meta_data_df.to_pickle('all_dicom_metadata.pkl')


def read_dicom_information_pkl():
    return pd.read_pickle('all_dicom_metadata.pkl')

