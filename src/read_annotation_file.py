import numpy as np
import nibabel as nib
from scipy.ndimage import convolve


def load_filtered_annotation_mask(annotation_path, image_series_path, threshold=130):
    annotation = nib.load(annotation_path)
    image = nib.load(image_series_path)
    return filter_annotation_mask(annotation, image, threshold)


def filter_annotation_mask(annotation, image, threshold=130):
    annotation_data = annotation.get_fdata()
    image_data = image.get_fdata()

    image_threshold_mask = image_data > threshold
    filtered_annotation_data = np.logical_and(image_threshold_mask, annotation_data).astype(float)

    neighbor_count = remove_single_voxel_mask(filtered_annotation_data)
    filtered_annotation_data = np.where(neighbor_count == 0, 0, filtered_annotation_data)

    return nib.Nifti1Image(filtered_annotation_data, annotation.affine, annotation.header)


def remove_single_voxel_mask(mask):
    # Should eliminate single pixels because of noise.
    kernel = np.ones((3, 3, 3), np.uint8)
    kernel[1, 1, 1] = 0
    neighbor_count = convolve(mask, kernel, mode='constant', cval=0)
    return neighbor_count