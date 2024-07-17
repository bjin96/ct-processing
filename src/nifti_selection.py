"""Pipeline step summaries"""
import json
from pathlib import Path

import pandas as pd

from src.create_region_of_interest import select_regions_of_interest
from src.datasets.structure import DatasetStructure
from src.determine_patient_orientation import get_patient_orientation, PatientOrientation
from src.read_nifti_information import get_nifti_headers, NIFTI_SELECTION_FILES

NON_INFORMATIVE_NIFTI_COLUMNS = [
    'ConversionSoftware',
    'ConversionSoftwareVersion',
    'SeriesNumber',
    'ImageComments'
]


def clean_nifti_info(
        base_path: Path,
        column_missing_data_threshold: int = 5,
):
    """
    Clean the nifti info dataframe.

    Args:
        column_missing_data_threshold: Maximum percentage of missing data in a column before it is rejected.
    """
    nifti_info_file_path = base_path / NIFTI_SELECTION_FILES[0]
    cleaned_nifti_info_file_path = base_path / NIFTI_SELECTION_FILES[1]

    all_nifti_info = pd.read_pickle(nifti_info_file_path)

    # Convert vector to PatientOrientation
    all_nifti_info['ImageOrientationPatientDICOM'] = all_nifti_info['ImageOrientationPatientDICOM'].apply(
        get_patient_orientation)

    # Drop unnecessary columns
    attribute_frequencies = all_nifti_info.isnull().sum()
    five_percent_of_data = len(all_nifti_info) / 100 * column_missing_data_threshold
    # Keep columns that have less than 5% missing data
    usually_available_attributes = all_nifti_info[
        attribute_frequencies[attribute_frequencies <= five_percent_of_data].index]

    usually_available_attributes = usually_available_attributes.drop(NON_INFORMATIVE_NIFTI_COLUMNS, axis=1, errors='ignore')

    usually_available_attributes.to_pickle(cleaned_nifti_info_file_path)


def add_statistics(
        summary: dict,
        step_df: pd.DataFrame,
        step_name: str,
        explanation: str
):
    summary['step_name'].append(step_name)
    summary['explanation'].append(explanation)

    summary['nifti_file_count'].append(len(step_df))
    summary['original_version_count'].append(
        len(step_df.groupby(['patient_id', 'study_subdir', 'series_subdir']).count()))
    summary['patient_count'].append(len(step_df.groupby(['patient_id']).count()))

    return summary


def add_roi_statistics(summary_object, step_df, step_name, explanation):
    summary_object['step_name'].append(step_name)
    summary_object['explanation'].append(explanation)

    summary_object['roi_count'].append(len(step_df))
    summary_object['nifti_file_count'].append(len(step_df.groupby('registered_file_path').count()))
    summary_object['original_version_count'].append(
        len(step_df.groupby(['patient_id', 'study_subdir', 'series_subdir']).count()))
    summary_object['patient_count'].append(len(step_df.groupby(['patient_id']).count()))

    return summary_object


def filter_nifti_files(
        base_path: Path,
        dataset_folder_structure: DatasetStructure,
):
    cleaned_nifti_info_file_path = base_path / NIFTI_SELECTION_FILES[1]
    filtered_nifti_info_file_path = base_path / NIFTI_SELECTION_FILES[2]

    summary_statistics = {
        'step_name': [],
        'nifti_file_count': [],
        'original_version_count': [],
        'patient_count': [],
        'explanation': []
    }

    cleaned_nifti_info: pd.DataFrame = pd.read_pickle(cleaned_nifti_info_file_path)
    summary_statistics = add_statistics(
        summary_statistics,
        cleaned_nifti_info,
        'Original nifti files produced by dcm2niix v1.0.20230411',
        'For a subset of the data, dcm2niix creates multiple nifti files per DICOM. See https://github.com/rordenlab/dcm2niix/blob/master/FILENAMING.md'
    )

    # Filter for only axial orientations
    only_axial = cleaned_nifti_info.loc[cleaned_nifti_info['ImageOrientationPatientDICOM'] == PatientOrientation.AXIAL]
    summary_statistics = add_statistics(
        summary_statistics,
        only_axial,
        'Only axial files',
        'Remove files where the Image Orientation (Patient) is not axial.'
    )

    # Filter for non-localizers
    only_axial = _add_slice_count(
        only_axial,
        base_path,
        dataset_folder_structure
    )
    no_localizer_axial = only_axial.loc[
        (only_axial['ImageType'].apply(lambda x: 'LOCALIZER' not in x if x is not None else False))
        & (only_axial['nifti_slice_count'] > 2)
        ]
    summary_statistics = add_statistics(
        summary_statistics,
        no_localizer_axial,
        'Without localizer files',
        '''Remove files where the Image Type contains "LOCALIZER" and volumes with less than three slices as
        indicated in the NIfTI file header.'''
    )

    # Filter for ROI nifti files
    no_roi_no_localizer_axial = no_localizer_axial.loc[no_localizer_axial['nifti_file'].apply(lambda x: 'ROI' not in x)]
    summary_statistics = add_statistics(
        summary_statistics,
        no_roi_no_localizer_axial,
        'Without ROI files',
        'DICOM files can contain binary overlays. dcm2niix creates additional ROI files for these overlays.'
    )

    # Choose a single nifti if there are multiple ones per directory
    unique_version_no_roi_no_localizer_axial = no_roi_no_localizer_axial \
        .groupby(['patient_id', 'study_subdir', 'series_subdir']) \
        .apply(dataset_folder_structure.choose_nifti_file, include_groups=False) \
        .reset_index()
    summary_statistics = add_statistics(
        summary_statistics,
        unique_version_no_roi_no_localizer_axial,
        'Choose unique version from duplicates',
        '''dcm2niix creates additional files for gantry tilt correction (Tilt_1), resliced equidistance
        (Eq_1) or both (Tilt_Eq_1) in addition to non-corrected files. We choose the corrected files here.'''
    )

    summary_statistics_df = pd.DataFrame(summary_statistics)
    print(summary_statistics_df)
    summary_statistics_df.to_csv(cleaned_nifti_info_file_path.parent / '02_summary_statistics.csv')

    unique_version_no_roi_no_localizer_axial.to_pickle(filtered_nifti_info_file_path)


def _add_slice_count(df, base_path: Path, dataset_folder_structure: DatasetStructure):
    slice_counts = []
    nifti_headers = get_nifti_headers(base_path)

    with_slice_count = df.copy()

    for index, row in with_slice_count.iterrows():
        file_path = dataset_folder_structure.create_path_from_csv_row(base_path / 'niftis', row, file_column_name='nifti_file')
        slice_count = nifti_headers[str(file_path)].get_data_shape()[2]
        slice_counts.append(slice_count)

    with_slice_count.loc[:, 'nifti_slice_count'] = slice_counts
    return with_slice_count


def check_registration_files(base_path: Path, dataset_folder_structure: DatasetStructure):
    filtered_nifti_info_file_path = base_path / NIFTI_SELECTION_FILES[2]
    registered_nifti_info_file_path = base_path / NIFTI_SELECTION_FILES[3]

    filtered_nifti_info = pd.read_pickle(filtered_nifti_info_file_path)
    age_group_info = pd.read_csv(base_path / 'age_group.csv', dtype={'age_group': 'int32', 'patient_id': 'str'})

    nifti_files_with_age_groups = filtered_nifti_info.merge(
        age_group_info,
        on='patient_id',
        how='inner',
    )
    registered_files = []

    for index, row in nifti_files_with_age_groups.iterrows():
        if index % 100 == 0:
            print(f'Processing {index} / {nifti_files_with_age_groups.shape[0]}')
        registered_file = Path(
            str(
                dataset_folder_structure.create_path_from_csv_row(
                    root_folder_path=base_path / 'registered_niftis',
                    row=row,
                    file_column_name='nifti_file'
                )
            )
        )
        if registered_file.is_file():
            registered_files.append(registered_file.name)
        else:
            registered_files.append(None)
    print(registered_files)

    nifti_files_with_age_groups['registered_file'] = registered_files

    success_registration_files = nifti_files_with_age_groups.loc[
        ~(nifti_files_with_age_groups['registered_file'].isnull())]
    success_registration_files.to_pickle(registered_nifti_info_file_path)
    write_registered_nifti_files_summary(registered_nifti_info_file_path)

    failed_registration_files = nifti_files_with_age_groups.loc[nifti_files_with_age_groups['registered_file'].isnull()]
    failed_registration_files.to_csv(
        registered_nifti_info_file_path.parent / '03_failed_registration_files.csv',
        index=False
    )


def write_registered_nifti_files_summary(registered_nifti_info_file_path):
    registered_nifti_files_info = pd.read_pickle(registered_nifti_info_file_path)
    summary_statistics = {
        'step_name': [],
        'nifti_file_count': [],
        'original_version_count': [],
        'patient_count': [],
        'explanation': []
    }
    add_statistics(
        summary_statistics,
        registered_nifti_files_info,
        'Co-registration with FLIRT v. 6.0',
        '''Registration seems to fail especially for the Siemens (Manufacturer) SOMATOM PLUS 4 
        (Manufacturers Model Name) with Software Version VC10C.'''
    )
    summary_statistics_df = pd.DataFrame(summary_statistics)
    print(summary_statistics_df)
    summary_statistics_df.to_csv(registered_nifti_info_file_path.parent / '03_summary_statistics.csv', index=False)


def manual_low_similarity_check_results(
        base_path: Path,
        dataset_structure: DatasetStructure,
):
    registered_nifti_info = pd.read_pickle(base_path / NIFTI_SELECTION_FILES[3])

    registration_failures_path = base_path / dataset_structure.REGISTRATION_FAILURES_CSV
    if registration_failures_path.exists():
        manual_exclusions = pd.read_csv(registration_failures_path)
    else:
        manual_exclusions = pd.DataFrame(columns=['image_path'])

    registered_nifti_info = dataset_structure.add_relative_path_to_dataframe(
        df=registered_nifti_info,
        relative_path_column_name='registered_file_path',
        file_column_name='registered_file'
    )

    accurate_registrations = registered_nifti_info.loc[~registered_nifti_info['registered_file_path'].isin(manual_exclusions['image_path'])]
    inaccurate_registrations = registered_nifti_info.loc[registered_nifti_info['registered_file_path'].isin(manual_exclusions['image_path'])]

    accurate_registration_path = base_path / NIFTI_SELECTION_FILES[4]
    accurate_registrations.to_pickle(accurate_registration_path)

    # List of accurate registrations for superimpose QC.
    full_path = accurate_registrations['registered_file_path'].apply(lambda relative_path: str(base_path / 'registered_niftis' / relative_path))
    with open(accurate_registration_path.parent / '04_accurate_registration_files.json', 'w') as file:
        json.dump(full_path.to_list(), file, indent=4)

    inaccurate_registrations.to_csv(
        accurate_registration_path.parent / '04_inaccurate_registration_files.csv',
        index=False
    )
    write_ssim_qc_nifti_files_summary(accurate_registration_path)


def write_ssim_qc_nifti_files_summary(accurate_registration_path):
    registered_nifti_files_info = pd.read_pickle(accurate_registration_path)
    summary_statistics = {
        'step_name': [],
        'nifti_file_count': [],
        'original_version_count': [],
        'patient_count': [],
        'explanation': []
    }
    add_statistics(
        summary_statistics,
        registered_nifti_files_info,
        'Manual check of low SSIM score registration files.',
        '''The co-registration can produce inaccurate co-registrations even if the FLIRT call was successful.
        This step identifies five groups per age group: complete, skull_base, skull_vault, middle, and incomplete to
        group similar scans together (done by a calculation of the information available across the x-axis for the
        co-registered scans). Afterwards the SSIM score to measure similarity between CT scan and MRI template is
        calculated and low similarity score registrations were examined manually.'''
    )
    summary_statistics_df = pd.DataFrame(summary_statistics)
    print(summary_statistics_df)
    summary_statistics_df.to_csv(accurate_registration_path.parent / '04_summary_statistics.csv', index=False)


def manual_superimpose_check_results(
        base_path: Path,
):
    registered_nifti_info = pd.read_pickle(base_path / NIFTI_SELECTION_FILES[4])

    blacklist_path = base_path / 'quality_control/superimpose_information/blacklist.json'
    if blacklist_path.exists():
        with open(blacklist_path, 'r') as file:
            blacklist_information = json.load(file)
        manual_exclusions = [entry['file'] for entry in blacklist_information]
    else:
        manual_exclusions = []

    accurate_registrations = registered_nifti_info.loc[~registered_nifti_info['registered_file_path'].isin(manual_exclusions)]
    inaccurate_registrations = registered_nifti_info.loc[registered_nifti_info['registered_file_path'].isin(manual_exclusions)]

    accurate_registrations_path = base_path / NIFTI_SELECTION_FILES[5]
    accurate_registrations.to_pickle(accurate_registrations_path)
    inaccurate_registrations.to_csv(
        accurate_registrations_path.parent / '05_inaccurate_registration_files.csv',
        index=False
    )

    write_superimpose_qc_nifti_files_summary(base_path)


def write_superimpose_qc_nifti_files_summary(base_path: Path):
    registered_nifti_files_info = pd.read_pickle(base_path / NIFTI_SELECTION_FILES[5])
    summary_statistics = {
        'step_name': [],
        'nifti_file_count': [],
        'original_version_count': [],
        'patient_count': [],
        'explanation': []
    }
    add_statistics(
        summary_statistics,
        registered_nifti_files_info,
        'Manual check superimposed dataset of registration files.',
        '''This is an additional quality control step. The registered CT scans were superimposed and
        outliers where identified and inspected manually to find inaccurate registrations. Inaccurate registrations
        have regions of interest for the arteries that are not at the actual position of the arteries. These were
        excluded.'''
    )
    summary_statistics_df = pd.DataFrame(summary_statistics)
    print(summary_statistics_df)
    summary_statistics_df.to_csv((base_path / NIFTI_SELECTION_FILES[5]).parent / '05_summary_statistics.csv', index=False)


def select_scans_with_at_least_one_roi(base_path: Path, dataset_structure: DatasetStructure):
    registered_nifti_info = pd.read_pickle(base_path / NIFTI_SELECTION_FILES[5])
    selected_rois = select_regions_of_interest(registered_nifti_info, base_path, dataset_structure)
    selected_rois.to_pickle(base_path / NIFTI_SELECTION_FILES[6])

    write_at_least_one_roi_nifti_files_summary(base_path)


def write_at_least_one_roi_nifti_files_summary(base_path: Path):
    registered_nifti_files_info = pd.read_pickle(base_path / NIFTI_SELECTION_FILES[6])
    summary_statistics = {
        'step_name': [],
        'roi_count': [],
        'nifti_file_count': [],
        'original_version_count': [],
        'patient_count': [],
        'explanation': []
    }
    add_roi_statistics(
        summary_statistics,
        registered_nifti_files_info,
        'Check if at least half of one ROI is present in the scans.',
        '''Not all scans contain the ROIs we defined in the MRI template. This step filters scans where
        there is no ROI present, e.g. skull vault scans, that don't reach low enough. This step filters by calculating
        the size of the ROI mask in the native CT space in absolute (mm^2) values compared to the original size of the
        ROI. The condition of inclusion for CT scans is at least one ROI with a size of half of the original
        template ROI.'''
    )
    summary_statistics_df = pd.DataFrame(summary_statistics)
    print(summary_statistics_df)
    summary_statistics_df.to_csv((base_path / NIFTI_SELECTION_FILES[6]).parent / '06_summary_statistics.csv', index=False)
