from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import image
import SimpleITK as sitk
from scipy.ndimage import laplace


def harmonize_kernel_ct_scan(
        ct_scan_path: Path,
        smoothed_output_path: Path,
        time_step=0.03125,
        conductance=3,
        iterations=5
) -> None:
    """
    Harmonize soft tissue and bone kernel CT scans.

    Args:
        ct_scan_path: Path to the CT scan to be harmonized.
        smoothed_output_path: Path to where the harmonized scan will be saved.
        time_step: Parameter of the curvature anisotropic diffusion smoothing.
        conductance: Parameter of the curvature anisotropic diffusion smoothing.
        iterations: Parameter of the curvature anisotropic diffusion smoothing.
    """
    ct_scan = image.load_img(ct_scan_path)
    clipped_image = _harmonize_kernel_ct_scan(
        original_ct_scan=ct_scan,
        time_step=time_step,
        conductance=conductance,
        iterations=iterations
    )
    clipped_image.to_filename(smoothed_output_path)


def _harmonize_kernel_ct_scan(
        original_ct_scan: nib.Nifti1Image,
        time_step=0.03125,
        conductance=3,
        iterations=5
):
    image_data = original_ct_scan.get_fdata()

    sigma = median_absolute_deviation(laplace(image_data[(image_data <= -950) & (image_data > -1050)]))
    if sigma > 30:
        image_data = sitk.GetArrayFromImage(
            sitk.CurvatureAnisotropicDiffusion(
                sitk.GetImageFromArray(image_data.astype(np.float32)),
                timeStep=time_step,
                conductanceParameter=conductance,
                numberOfIterations=iterations,
            )
        )

    return nib.Nifti1Image(image_data, original_ct_scan.affine, original_ct_scan.header)


def median_absolute_deviation(x):
    x = np.asarray(x)
    return np.median(np.abs(x - np.mean(x)))