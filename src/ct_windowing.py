from pathlib import Path

import nibabel as nib
from nibabel import Nifti1Image
from nilearn import image


DEFAULT_CT_RANGE = (-1024, 3071)


def window_ct_scan(ct_scan_path, windowed_output_path, window_minimum, window_maximum) -> None:
    """
    Window a given CT scan to the given window_minimum and window_maximum.

    Args:
    ct_scan_path: Path to the CT scan to be windowed.
    windowed_output_path: Path to where the windowed scan will be saved.
    window_minimum: Lower limit of the window. Attenuation values below will be clipped to the minimum value.
    window_maximum: Upper limit of the window. Attenuation values above will be clipped to the maximum value.
    """
    ct_scan = image.load_img(ct_scan_path)
    clipped_image_data = ct_scan.get_fdata().clip(window_minimum, window_maximum)
    clipped_image = nib.Nifti1Image(clipped_image_data, ct_scan.affine, ct_scan.header)
    clipped_image.to_filename(windowed_output_path)


def filter_window_ct_scan(ct_scan_path, windowed_output_path, window_minimum, window_maximum) -> None:
    """
    Window a given CT scan to the given window_minimum and window_maximum. Anything outside the boundaries is 0.

    :param ct_scan_path: Path to the CT scan to be windowed.
    :param windowed_output_path: Path to where the windowed scan will be saved.
    :param window_minimum: Lower limit of the window. Attenuation values below will be clipped to the minimum value.
    :param window_maximum: Upper limit of the window. Attenuation values above will be clipped to the maximum value.
    """
    ct_scan = image.load_img(ct_scan_path)
    clipped_image_data = ct_scan.get_fdata()
    clipped_image_data[clipped_image_data < window_minimum] = window_minimum
    clipped_image_data[clipped_image_data > window_maximum] = window_minimum
    clipped_image = nib.Nifti1Image(clipped_image_data, ct_scan.affine, ct_scan.header)
    clipped_image.to_filename(windowed_output_path)


def shift_ct_scan(ct_scan_path, shifted_output_path):
    ct_scan = image.load_img(ct_scan_path)
    clipped_image_data = ct_scan.get_fdata()
    clipped_image_data = clipped_image_data - clipped_image_data.min()
    clipped_image = nib.Nifti1Image(clipped_image_data, ct_scan.affine, ct_scan.header)
    clipped_image.to_filename(shifted_output_path)


def invert_ct_scan(ct_scan_path, inverted_output_path):
    ct_scan = image.load_img(ct_scan_path)
    clipped_image_data = ct_scan.get_fdata()
    clipped_image_data = -clipped_image_data
    clipped_image = nib.Nifti1Image(clipped_image_data, ct_scan.affine, ct_scan.header)
    clipped_image.to_filename(inverted_output_path)


def binarize_ct_scan_files(
        ct_scan_path,
        windowed_output_path: Path,
        threshold
):
    if windowed_output_path.exists():
        return
    windowed_output_path.parent.mkdir(parents=True, exist_ok=True)
    ct_scan = image.load_img(ct_scan_path)
    ct_scan_data = ct_scan.get_fdata()
    clipped_image = binarize_ct_scan(ct_scan_data, threshold)
    clipped_image = nib.Nifti1Image(clipped_image, clipped_image.affine, clipped_image.header)
    clipped_image.to_filename(windowed_output_path)


def binarize_ct_scan(
        ct_scan_data: Nifti1Image,
        threshold
):
    ct_scan_data[ct_scan_data < threshold] = 0
    ct_scan_data[ct_scan_data >= threshold] = 1
    return ct_scan_data
