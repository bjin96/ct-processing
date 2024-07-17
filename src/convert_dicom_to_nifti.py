"""Converting DICOM to NIfTI with dcm2niix."""
import glob
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib

from src.call_command_line import call_command_line_verbose
from src.datasets.structure import DatasetStructure


def anonymise_study_time_with_gdcmanon(
    in_dir: Path,
    out_dir: Path,
    replace_with: str = "000000.000000",
) -> None:
    """
    Anonymize Study Time (0008,0030) for all DICOMs in a directory using gdcmanon.
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # (0008,0030) = Study Time
    commandline = f'gdcmanon --dumb --replace 0008,0030={replace_with} -i "{str(in_dir)}" -o "{str(out_dir)}"'
    call_command_line_verbose(commandline)


def call_dcm2niix(nii_path, dcm_path):
    """
    Calls dcm2niix to convert the DICOM files at dcm_path into NIfTI format.

    Args:
        nii_path: Output path for the NIfTI files.
        dcm_path: Path to the folder where the DICOM files lie.

    Returns
        stdout and stderr strings.
    """
    # Create a temporary directory with anonymized study times to avoid study splitting by dcm2niix. The -m option did
    # not work...
    with TemporaryDirectory() as tmp:
        try:
            temporary_dcm_path = Path(tmp)
            anonymise_study_time_with_gdcmanon(dcm_path, temporary_dcm_path)

            # -o: output path
            # -f: filename with %i = ID of patient, %t = time, %s = series number
            # -z: gz compress images with y = pigz
            # -m: Join slices from a series together regardless of changing parameters (e.g. study time, orientation)
            # -i: Ignore derived, localizer and 2D images (y/n, default n) with y = yes
            # <input folder>: folder where the .dcm files to be converted lie.
            commandline = f'dcm2niix -o "{nii_path}" -f %i_%x_%s -z y -m y "{temporary_dcm_path}"'
            print(f'{nii_path=}, {dcm_path=}')
            call_command_line_verbose(commandline)
        except subprocess.CalledProcessError as e:
            print(f'Calling dcm2niix failed with: {e.stderr} and {e.stdout}')


def convert_dcm_to_nifti(dcm_path: Path, nii_path: Path):
    if os.path.exists(nii_path):
        print(f'NIFTI path {nii_path} already existed.')
    else:
        os.makedirs(nii_path)
        call_dcm2niix(nii_path, dcm_path)

    all_nii_file_paths = glob.glob(str(nii_path) + '/*.nii.gz')
    print(f'{len(all_nii_file_paths)} .nii.gz files have been generated/found.')
    return all_nii_file_paths


def convert_all_dcms_to_niftis(
        base_path: Path,
        dicom_path: Path
):
    relative_dcm_folder_paths_json = base_path / DatasetStructure.DICOM_FOLDERS_JSON
    nii_base_path = base_path / 'niftis'

    with open(relative_dcm_folder_paths_json, 'r') as file:
        relative_dcm_folder_paths = json.load(file)
    for index, relative_dcm_folder_path in enumerate(relative_dcm_folder_paths):
        if index % 100 == 0:
            print(f'{index=}')

        convert_dcm_to_nifti(
            dcm_path=dicom_path / relative_dcm_folder_path,
            nii_path=nii_base_path / relative_dcm_folder_path,
        )


def read_nifti_header(path):
    nifti = nib.load(path)
    return nifti.header
