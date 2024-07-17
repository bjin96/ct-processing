"""Regions of interest definition for both MRI templates."""
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Tuple

import numpy as np
import nibabel as nib
from nilearn.image import image

# In millimeters
T1_TEMPLATE_VOXEL_SIZE = (1., 1., 1.)


def get_template_roi_regions(age_group):
    return {
        'ROI_vertebral': f'age_group_templates/{age_group}_template_regions/ROI_vertebral.nii.gz',
        'ROI_basilar': f'age_group_templates/{age_group}_template_regions/ROI_basilar.nii.gz',
        'ROI_ICA_cavernous': f'age_group_templates/{age_group}_template_regions/ROI_ICA_cavernous.nii.gz',
        'ROI_MCA_left': f'age_group_templates/{age_group}_template_regions/ROI_MCA_left.nii.gz',
        'ROI_MCA_right': f'age_group_templates/{age_group}_template_regions/ROI_MCA_right.nii.gz',
    }


class RegionOfInterest(Enum):
    VERTEBRAL = 'ROI_vertebral'
    BASILAR = 'ROI_basilar'
    CAROTID = 'ROI_ICA_cavernous'
    LEFT_MCA = 'ROI_MCA_left'
    RIGHT_MCA = 'ROI_MCA_right'


@dataclass
class RegionOfInterestDefinition:
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    z_min: int
    z_max: int


VERTEBRAL_ARTERIES_ROI = RegionOfInterestDefinition(
    x_min=59,
    x_max=123,
    y_min=70,
    y_max=125,
    z_min=1,
    z_max=25
)
BASILAR_ARTERY_ROI = RegionOfInterestDefinition(
    x_min=67,
    x_max=115,
    y_min=88,
    y_max=130,
    z_min=26,
    z_max=62
)
ICA_CAVERNOUS_SEGMENT = RegionOfInterestDefinition(
    x_min=68,
    x_max=114,
    y_min=114,
    y_max=148,
    z_min=29,
    z_max=62
)
# These are coordinates for the 0.5^3mm space.
# ROI center in that space is: (128, 172, 74)
# 96 * 0.44 / 0.5 = 84, 32 * 1 / 0.5 = 64
# With ViT roi center = (145, 195, 34)
# Mirroring ViT roi would be 128±84/2 = (86, 170), 172±84/2 = (130, 214), 68±64/2 = (36, 100)
ICA_CAVERNOUS_SEGMENT_RSS_SMALL = RegionOfInterestDefinition(
    x_min=86,
    x_max=170,
    y_min=130,
    y_max=214,
    z_min=36,
    z_max=100
)
ICA_CAVERNOUS_SEGMENT_RSS_LARGE = RegionOfInterestDefinition(
    x_min=76,
    x_max=180,
    y_min=120,
    y_max=224,
    z_min=26,
    z_max=100
)
MCA_LEFT_SEGMENT = RegionOfInterestDefinition(
    x_min=113,
    x_max=157,
    y_min=115,
    y_max=155,
    z_min=42,
    z_max=65
)
MCA_RIGHT_SEGMENT = RegionOfInterestDefinition(
    x_min=26,
    x_max=69,
    y_min=115,
    y_max=155,
    z_min=42,
    z_max=65
)
ALL_REGION_OF_INTEREST = [
    VERTEBRAL_ARTERIES_ROI,
    BASILAR_ARTERY_ROI,
    ICA_CAVERNOUS_SEGMENT,
    MCA_RIGHT_SEGMENT,
    MCA_LEFT_SEGMENT
]


def calculate_roi_size(
        roi: RegionOfInterestDefinition,
        pixel_size: Tuple[float, float, float] = (T1_TEMPLATE_VOXEL_SIZE[:3])
) -> float:
    return (
            (roi.x_max - roi.x_min + 1) * pixel_size[0]
            * (roi.y_max - roi.y_min + 1) * pixel_size[1]
            * (roi.z_max - roi.z_min + 1) * pixel_size[2]
    )


REGION_OF_INTEREST_SIZE = {
    RegionOfInterest.VERTEBRAL: calculate_roi_size(VERTEBRAL_ARTERIES_ROI),
    RegionOfInterest.BASILAR: calculate_roi_size(BASILAR_ARTERY_ROI),
    RegionOfInterest.CAROTID: calculate_roi_size(ICA_CAVERNOUS_SEGMENT),
    RegionOfInterest.LEFT_MCA: calculate_roi_size(MCA_LEFT_SEGMENT),
    RegionOfInterest.RIGHT_MCA: calculate_roi_size(MCA_RIGHT_SEGMENT),
}


def get_roi_mask(roi_definition: RegionOfInterestDefinition, template_shape: tuple) -> np.ndarray:
    roi_mask = np.zeros(shape=template_shape)
    roi_mask[
        roi_definition.x_min - 1:roi_definition.x_max,
        roi_definition.y_min - 1:roi_definition.y_max,
        roi_definition.z_min - 1:roi_definition.z_max
    ] = 1
    return roi_mask


def create_roi_nifti_image(
        region_of_interest: RegionOfInterestDefinition,
        original_image_path: Path,
        output_path: Path,
):
    """
    Create a mask for the specified region_of_interest in the given original_image_path and saves it as a .nii.gz.
    """
    original_ct_scan = image.load_img(original_image_path)

    roi_mask = np.zeros(shape=original_ct_scan.header.get_data_shape())

    roi_mask += get_roi_mask(region_of_interest, original_ct_scan.header.get_data_shape())

    roi_mask = roi_mask.clip(0, 1)

    nib.Nifti1Image(roi_mask, affine=original_ct_scan.affine).to_filename(output_path)
