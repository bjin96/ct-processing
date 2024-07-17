import click

from src.commands import convert_dcm_to_nifti, read_nifti_information, clean_nifti_information, \
    filter_nifti_information, determine_patient_age_group, create_dcm_folder_list, register_to_template, \
    check_registration, calculate_scan_completeness, invert_registration_matrices, create_image_series_roi_masks, \
    create_rois, control_similarity_to_template, control_low_similarity_series, control_superimposed_series, \
    check_contains_roi, create_ground_truth_rois, create_dataset_json, create_nnunet_dataset


@click.group()
def cli():
    pass


if __name__ == '__main__':
    cli.add_command(convert_dcm_to_nifti)
    cli.add_command(read_nifti_information)
    cli.add_command(clean_nifti_information)
    cli.add_command(filter_nifti_information)
    cli.add_command(determine_patient_age_group)
    cli.add_command(create_dcm_folder_list)
    cli.add_command(register_to_template)
    cli.add_command(invert_registration_matrices)
    cli.add_command(check_registration)
    cli.add_command(calculate_scan_completeness)
    cli.add_command(control_similarity_to_template)
    cli.add_command(control_low_similarity_series)
    cli.add_command(control_superimposed_series)
    cli.add_command(create_image_series_roi_masks)
    cli.add_command(create_rois)
    cli.add_command(create_ground_truth_rois)
    cli.add_command(check_contains_roi)
    cli.add_command(create_dataset_json)
    cli.add_command(create_nnunet_dataset)
    cli()
