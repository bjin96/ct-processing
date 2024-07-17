"""Co-registering two scans with FLIRT"""
import subprocess
from multiprocessing import Pool
from pathlib import Path
from typing import List

import pandas as pd

from src.call_command_line import call_command_line_verbose
from src.datasets.structure import DatasetStructure
from src.read_nifti_information import NIFTI_SELECTION_FILES


def run_register_to_template(
        base_path: Path,
        dataset_structure: DatasetStructure
):
    age_group_info = pd.read_csv(base_path / DatasetStructure.AGE_GROUP_CSV)
    age_group_info['patient_id'] = age_group_info['patient_id'].astype(str)
    filtered_nifti_info: pd.DataFrame = pd.read_pickle(base_path / NIFTI_SELECTION_FILES[2])

    filtered_nifti_info = filtered_nifti_info.merge(age_group_info, on='patient_id', how='left')

    age_group_templates = dataset_structure.get_age_group_template_paths(base_path=base_path)
    for template_path in age_group_templates:
        age_group = int(str(template_path.stem).split('.')[0])
        filtered = filtered_nifti_info.loc[filtered_nifti_info['age_group'] == age_group]

        co_registration_list = [
            dataset_structure.create_path_from_csv_row(base_path / 'niftis', row, 'nifti_file')
            for _, row
            in filtered.iterrows()
        ]
        output_path_list = [
            dataset_structure.create_path_from_csv_row(base_path / 'registered_niftis', row, 'nifti_file')
            for _, row
            in filtered.iterrows()
        ]

        co_register_list_to_template(
            co_registration_list=co_registration_list,
            template_path=template_path,
            output_path_list=output_path_list,
        )


def run_invert_registration_matrices(
        base_path: Path,
        dataset_structure: DatasetStructure,
):
    registered_nifti_info_file_path = base_path / NIFTI_SELECTION_FILES[3]
    filtered_nifti_info = pd.read_pickle(registered_nifti_info_file_path)

    invert_arguments = []
    for index, row in filtered_nifti_info.iterrows():
        registered_file = Path(
            str(
                dataset_structure.create_path_from_csv_row(
                    root_folder_path=base_path / 'registered_niftis',
                    row=row,
                    file_column_name='nifti_file'
                )
            )
        )
        matrix_file = Path(str(registered_file).replace('.nii.gz', '.mat'))
        inverted_matrix_file = Path(matrix_file).parent / f'inverted_{matrix_file.name}'
        if matrix_file.is_file():
            invert_arguments.append((matrix_file, inverted_matrix_file))
        else:
            print(f'No matrix found for {row["nifti_file"]}')

    invert_list_of_matrices(invert_arguments)


def co_register_list_to_template(
        co_registration_list: List[Path],
        template_path: Path,
        output_path_list: List[Path]
) -> None:
    """
    Co-register a list of paths to scans to a template.

    Args:
        co_registration_list: List of paths to files to co-register.
        template_path: Path to the template to which to co-register to.
        output_path_list: List of paths to the output.
    """
    if len(co_registration_list) != len(output_path_list):
        raise ValueError('Input and output path lists should have the same length.')

    args_list = []
    for input_path, output_path in zip(co_registration_list, output_path_list):
        args_list.append((input_path, template_path, output_path))

    with Pool() as p:
        p.starmap(co_register, args_list)


def co_register(path_to_register, path_register_to, registered_file_name):
    """
    Co-register a template at template_path to a CT scan at ct_path. Output are the registered_binarized template and a
    .mat file containing the transformation matrix.

    Args:
        path_to_register: Path to the file to be co-registered_binarized.
        path_register_to: Path to the file to which to co-register.
        registered_file_name: Output name for the registration files.
    """
    # FLIRT command explanation:
    # -in: input volume
    # -ref: reference volume
    # -out: output volume
    # -omat: matrix filename
    # -bins: number of histogram bins (default 256...)
    # -cost: cost function to optimize against (default is corratio...)
    # -searchrx/y/z: min angle – max angle
    # -dof: number of transform dof (default is 12...)
    # -interp: final interpolation (default is trilinear...)
    commandline = f'''
        flirt
        -in "{path_to_register}"
        -ref "{path_register_to}"
        -out "{registered_file_name}"
        -omat "{registered_file_name.parent / registered_file_name.name.replace('.nii.gz', '')}.mat"
        -bins 256
        -cost mutualinfo
        -searchrx -90 90
        -searchry -90 90
        -searchrz -90 90
        -dof 12
        -interp trilinear
    '''
    registered_file_name.parent.mkdir(parents=True, exist_ok=True)
    if registered_file_name.exists():
        print(f'File {registered_file_name} already exists.')
        return
    try:
        call_command_line_verbose(commandline)
        print(f'Co-registration of {path_to_register} to {path_register_to} successful.')
    except subprocess.CalledProcessError as e:
        print(f'Calling flirt to co-register {path_to_register} to {path_register_to} failed with: {e.stderr}')


def apply_transformation_matrix(reference_volume, input_volume, transform_matrix_path, output_path: Path):
    # FLIRT command explanation:
    # -in: input volume
    # -ref: reference volume
    # -out: output volume
    # -omat: matrix filename
    # -bins: number of histogram bins (default 256...)
    # -cost: (default is corratio...)
    # -searchrx/y/z: min angle – max angle
    # -dof: number of transform dof (default is 12...)
    # -interp: final interpolation (default is trilinear...)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f'File {output_path} already exists.')
        return
    commandline = f'''
        flirt
        -in "{input_volume}"
        -out "{output_path}"
        -ref "{reference_volume}"
        -applyxfm
        -init "{transform_matrix_path}"
        '''
    try:
        call_command_line_verbose(commandline)
        print(
            f'Application of transformation matrix at {transform_matrix_path} to {input_volume} successful.')
    except subprocess.CalledProcessError as e:
        print(
            f'''
            Calling flirt to apply transformation matrix at {transform_matrix_path} to {input_volume}'
            failed with: {e.stderr}
            '''
        )


def apply_list_of_matrices(args_list):
    with Pool() as p:
        p.starmap(apply_transformation_matrix, args_list)


def invert_transformation_matrix(matrix_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f'File {output_path} already exists.')
        return
    commandline = f'convert_xfm -inverse "{matrix_path}" -omat "{output_path}"'
    try:
        call_command_line_verbose(commandline)
    except subprocess.CalledProcessError as e:
        print(f'Calling flirt to invert transformation matrix at {matrix_path} failed with: {e.stderr}')


def invert_list_of_matrices(args_list):
    with Pool() as p:
        p.starmap(invert_transformation_matrix, args_list)
