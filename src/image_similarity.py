"""Measuring similarity between two scans"""
from pathlib import Path

import pandas as pd
from skimage.metrics import structural_similarity
from nilearn import image
import nibabel as nib

from src.datasets.structure import DatasetStructure
from src.read_nifti_information import NIFTI_SELECTION_FILES
from src.scan_completeness import path_for_templates_pickle


# from src.plotting.plot_co_registrations import plot_co_registration_interactive


def compare_image_similarity(image_path1, image_path2, ssim_image_path=None):
    image1 = image.load_img(image_path1)
    image2 = image.load_img(image_path2)

    data1 = image1.get_fdata()
    data2 = image2.get_fdata()

    if ssim_image_path is None:
        # 4000 indicates the range of Hounsfield units -1000 to 3000
        try:
            return structural_similarity(data1, data2, data_range=4000)
        except ValueError as e:
            print(f'Failed to compute structural similarity for {image_path1} with: {e}')

    else:
        ssim, ssim_image = structural_similarity(data1, data2, data_range=4000)
        nifti_ssi_image = nib.Nifti1Image(ssim_image, image1.affine, image1.header)
        # plot_co_registration_interactive(nifti_ssi_image, None, ssim_image_path)
        return ssim


def compare_similarity_of_image_list_to_template(
        base_path: Path,
        image_path_list,
        template_path,
        age_group
):
    print(f'Comparing similarity to {template_path}')

    ssim_information = []

    for index, image_path in enumerate(image_path_list):
        if index % 100 == 0:
            print(f'{index} / {len(image_path_list)}')

        try:
            ssim = compare_image_similarity(image_path, template_path)
            ssim_information.append([image_path, template_path, ssim])
        except EOFError as e:
            print(f'Could not calculate similarity of {image_path} with {e}')

    similarity_information_folder = base_path / f'quality_control/similarity_information'
    similarity_information_folder.mkdir(parents=True, exist_ok=True)

    ssim_information_df = pd.DataFrame(ssim_information, columns=['image_path', 'template_path', 'ssim'])
    ssim_information_df.to_csv(similarity_information_folder / f'{age_group}_ssim.csv', index=False)


def calculate_template_similarity(
        base_path: Path,
        dataset_structure: DatasetStructure,
):
    paths_by_age_group = path_for_templates_pickle(
        pickle_path=base_path / NIFTI_SELECTION_FILES[3],
        root_folder_path=base_path / 'registered_niftis',
        dataset_structure=dataset_structure
    )

    for age_group, image_paths in paths_by_age_group.items():
        templates = dataset_structure.get_age_group_template_paths(base_path)

        compare_similarity_of_image_list_to_template(
            base_path=base_path,
            image_path_list=image_paths,
            template_path=templates[age_group],
            age_group=age_group,
        )
