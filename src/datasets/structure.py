import json
import re
from abc import abstractmethod, ABC
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class ImageSeriesIdentifier:
    patient_id: str
    study_id: str
    series_id: str


class DatasetStructure(ABC):

    AGE_GROUP_CSV = Path('age_group.csv')
    DICOM_FOLDERS_JSON = Path('dicom_folders.json')
    AGE_GROUP_TEMPLATES_PATH = Path('age_group_templates')
    COMPLETENESS_INFORMATION_PATH = Path('quality_control/completeness_information')
    SIMILARITY_INFORMATION_PATH = Path('quality_control/similarity_information')
    REGION_OF_INTEREST_SIZES_CSV = Path('nifti_selection_files/regions_size.csv')
    REGISTRATION_FAILURES_CSV = SIMILARITY_INFORMATION_PATH / 'registration_failures.csv'

    @staticmethod
    @abstractmethod
    def get_identifiers_from_folder_structure(
            relative_path: Path
    ) -> ImageSeriesIdentifier:
        pass

    @staticmethod
    @abstractmethod
    def get_relative_path_from_csv_row(row):
        pass

    @classmethod
    @abstractmethod
    def get_relative_file_path_from_csv_row(cls, row, file_column_name='file'):
        pass

    @classmethod
    def create_path_from_csv_row(cls, root_folder_path, row, file_column_name='file'):
        return Path(root_folder_path).absolute() \
            / cls.get_relative_file_path_from_csv_row(row, file_column_name=file_column_name)

    @property
    @abstractmethod
    def patient_id_pattern(self):
        pass

    @property
    @abstractmethod
    def study_time_pattern(self):
        pass

    @property
    @abstractmethod
    def relative_path_length(self):
        """
        Length of the file path from the base folder to the dicom files. E.g.
        ./<patient_id>/<study_id>/<series_id>/0.dcm would be relative_path_length = 3. The dicom folder structure is
        mirrored in the nifti folder structure.
        """
        pass

    @classmethod
    def choose_nifti_file(cls, df):
        """
        Choose tilt and equidistant files if there are multiple nifti files produced by dcm2niix. Advantage: proper
        format in NIfTI standard, disadvantage: additional resampling/interpolation.
        """
        if len(df) == 1:
            return df

        drop_files = []

        for index, row in df.iterrows():
            non_corrected_file = cls.patient_id_pattern + cls.study_time_pattern + r'_([0-9]+)((_i[0-9]{5})|([a-z]))?\.nii\.gz$'
            if re.compile(non_corrected_file).search(row['nifti_file']):
                corrected_file = fr'^({row["nifti_file"].replace(".nii.gz", "")})_((Tilt_1)|(Eq_1)|(Tilt_Eq_1))\.nii\.gz$'
                if any([re.compile(corrected_file).search(f) for f in df['nifti_file']]):
                    drop_files.append(row['nifti_file'])

        cleaned = df.loc[df['nifti_file'].apply(lambda x: x not in drop_files)]
        return cleaned

    @staticmethod
    @abstractmethod
    def read_age_group_file(
            input_file_path: Path
    ):
        pass

    @classmethod
    def determine_age_group_from_xlsx(
            cls,
            input_file_path: Path,
            base_output_path: Path,
            age_bins: List[int]
    ):
        age_info = cls.read_age_group_file(input_file_path)
        age_info['age_group'] = pd.cut(age_info['age'], bins=age_bins, labels=range(len(age_bins) - 1))
        age_info = age_info[['patient_id', 'age_group']]
        age_info.to_csv(base_output_path / cls.AGE_GROUP_CSV, index=False)

    @classmethod
    def get_age_groups(cls, base_path: Path):
        age_group_info = pd.read_csv(base_path / cls.AGE_GROUP_CSV)
        return age_group_info['age_group'].unique()


    @classmethod
    @abstractmethod
    def get_dicom_folders(cls, dicom_directory: Path):
        pass

    @classmethod
    def write_dicom_folders_to_file(
            cls,
            base_path: Path,
            dicom_directory: Path
    ):
        dicom_folders = cls.get_dicom_folders(dicom_directory)
        with open(base_path / cls.DICOM_FOLDERS_JSON, 'w') as file:
            json.dump(dicom_folders, file)

    @classmethod
    def get_relative_path_from_absolute(
            cls,
            path: Path,
            relative_path_length: int = None
    ) -> Path:
        if relative_path_length is None:
            relative_path_length = cls.relative_path_length
        return path.relative_to(path.parents[relative_path_length])

    @classmethod
    def get_age_group_template_paths(
            cls,
            base_path: Path
    ) -> List[Path]:
        template_file_pattern = r'^[0-9]+.nii.gz$'
        template_path = base_path / DatasetStructure.AGE_GROUP_TEMPLATES_PATH
        return sorted([p for p in template_path.iterdir() if re.match(template_file_pattern, p.name)])

    @classmethod
    def add_relative_path_to_dataframe(cls, df, relative_path_column_name, file_column_name='file'):
        relative_paths = []
        for index, row in df.iterrows():
            relative_paths.append(str(cls.get_relative_file_path_from_csv_row(row, file_column_name)))
        return_copy = df.copy()
        return_copy[relative_path_column_name] = relative_paths
        return return_copy


class Ist3DatasetStructure(DatasetStructure):

    patient_id_pattern = r'^[0-9]{5}'
    study_time_pattern = r'_([0-9]{14})'

    @staticmethod
    def get_identifiers_from_folder_structure(
            relative_path: Path
    ) -> ImageSeriesIdentifier:
        return ImageSeriesIdentifier(
            patient_id=relative_path.parents[2].name,
            study_id=relative_path.parents[1].name,
            series_id=relative_path.name
        )

    @staticmethod
    def get_relative_path_from_csv_row(row):
        return Path(f'{str(row["patient_id"]).zfill(5)}/{row["study_subdir"]}/{row["series_subdir"]}')

    @classmethod
    def get_relative_file_path_from_csv_row(cls, row, file_column_name='file'):
        return cls.get_relative_path_from_csv_row(row) / row[file_column_name]

    @staticmethod
    def read_age_group_file(
            input_file_path: Path
    ):
        raise NotImplementedError

    @classmethod
    def get_dicom_folders(cls, dicom_directory: Path):
        raise NotImplementedError


class Mss3DatasetStructure(DatasetStructure):

    patient_id_pattern = r'^(MSS3_ED)_([0-9]{3})'
    study_time_pattern = r'_([0-9]{14})'
    relative_path_length = 3

    @staticmethod
    def get_identifiers_from_folder_structure(
            relative_path: Path
    ) -> ImageSeriesIdentifier:
        patient_id = relative_path.parents[1].name

        if len(patient_id) == 11:
            study_id = 0
        elif len(patient_id) > 11:
            study_id = patient_id[11:].replace('_', '')
            patient_id = patient_id[:11]
        else:
            raise ValueError(f'Patient ID {patient_id} has invalid length.')

        return ImageSeriesIdentifier(patient_id=patient_id, study_id=study_id, series_id=relative_path.name)

    @staticmethod
    def get_relative_path_from_csv_row(row):
        patient_id = row['patient_id']
        if row['study_subdir'] != 0:
            patient_id = f'{patient_id}_{row["study_subdir"]}'
        return Path(f'{patient_id}/CT/{row["series_subdir"]}')

    @classmethod
    def get_relative_file_path_from_csv_row(cls, row, file_column_name='file'):
        return cls.get_relative_path_from_csv_row(row) / row[file_column_name]

    @staticmethod
    def read_age_group_file(
            input_file_path: Path
    ):
        patient_info = pd.read_excel(input_file_path)
        return pd.DataFrame({
            'patient_id': patient_info['MSS3_Study_Number'],
            'age': patient_info['ageAtIndexStroke']
        })

    @classmethod
    def get_dicom_folders(cls, dicom_directory: Path):
        patients_folders = [p for p in dicom_directory.iterdir() if p.is_dir()]
        image_series = []
        for patient_folder in patients_folders:
            data_folder = patient_folder / 'CT'
            image_series_folders = [
                str(cls.get_relative_path_from_absolute(p, cls.relative_path_length - 1))
                for p in data_folder.iterdir()
                if p.is_dir()
            ]
            image_series.extend(image_series_folders)
        image_series = sorted(image_series)
        return image_series


class RotterdamTraumaDatasetStructure(DatasetStructure):

    patient_id_pattern = r'^([0-9]+)'
    study_time_pattern = r'(?:_[0-9]{14})?'
    relative_path_length = 2

    @staticmethod
    def get_identifiers_from_folder_structure(
            relative_path: Path
    ) -> ImageSeriesIdentifier:
        patient_id = relative_path.parent.name.split('_')[0]
        study_id = relative_path.parent.name.split('_')[1]
        return ImageSeriesIdentifier(patient_id=patient_id, study_id=study_id, series_id=relative_path.name)

    @staticmethod
    def get_relative_path_from_csv_row(row):
        patient_id = row['patient_id']
        return Path(f'{patient_id}_{row["study_subdir"]}/{row["series_subdir"]}')

    @classmethod
    def get_relative_file_path_from_csv_row(cls, row, file_column_name='file'):
        return cls.get_relative_path_from_csv_row(row) / row[file_column_name]

    @staticmethod
    def read_age_group_file(
            input_file_path: Path
    ):
        patient_info = pd.read_csv(input_file_path)
        print('No age information for trauma cohort. Assigning age 65.')
        return pd.DataFrame({
            'patient_id': patient_info['ID'].apply(lambda x: x.split('_')[0]),
            'age': 65
        })

    @classmethod
    def get_dicom_folders(cls, dicom_directory: Path):
        patients_folders = [p for p in dicom_directory.iterdir() if p.is_dir()]
        image_series = []
        for patient_folder in patients_folders:
            data_folder = patient_folder
            image_series_folders = [
                str(cls.get_relative_path_from_absolute(p, cls.relative_path_length - 1))
                for p in data_folder.iterdir()
                if p.is_dir()
            ]
            image_series.extend(image_series_folders)
        image_series = sorted(image_series)
        return image_series

    @classmethod
    def choose_nifti_file(cls, df):
        """
        Don't choose corrected file because resampling for the annotations failed in some instances...
        """
        if len(df) == 1:
            return df

        drop_files = []

        for index, row in df.iterrows():
            if 'Tilt_1' in row['nifti_file'] or 'Eq_1' in row['nifti_file'] or 'Tilt_Eq_1' in row['nifti_file']:
                drop_files.append(row['nifti_file'])

        cleaned = df.loc[df['nifti_file'].apply(lambda x: x not in drop_files)]
        return cleaned