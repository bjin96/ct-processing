"""Calculating patient orientation (axial, coronal, sagittal) from Image Orientation (Patient) values"""
import numpy as np
from typing import List
from enum import Enum


class PatientOrientation(Enum):
    AXIAL = 'axial'
    SAGITTAL = 'sagittal'
    CORONAL = 'coronal'


def get_patient_orientation(
        dcm_patient_orientation: List[float]
) -> PatientOrientation:
    """
    Calculate the orientation of the patient by calculating the cross product of the x, y direction vectors. The cross
    product vector is perpendicular to the x and y vectors.

    Args:
        dcm_patient_orientation: List of floats of the DICOM patient orientation as found in the field
            (0020, 0037) Image Orientation (Patient) in the DICOM header.

    Returns:
        Patient orientation showing the orientation of the scan.
    """
    if dcm_patient_orientation is None:
        return None

    image_y = dcm_patient_orientation[:3]
    image_x = dcm_patient_orientation[3:]
    image_z = np.cross(image_x, image_y)
    absolute_image_z = abs(image_z)
    main_index = list(absolute_image_z).index(max(absolute_image_z))

    match main_index:
        case 0:
            return PatientOrientation('sagittal')
        case 1:
            return PatientOrientation('coronal')
        case _:
            return PatientOrientation('axial')
