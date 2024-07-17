import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from nilearn import plotting

from src.datasets.structure import DatasetStructure
from src.create_completeness_groups import CompletenessGroup


def plot_histogram(
        similarity_df,
        completeness_paths,
        plot_title,
        plot_x_axis_label,
        output_path,
        plot_lowest=True,
        plot_all=False,
        cutoff_percentage=0.05,
):
    """
    We visually inspect the low 5% of SSIM score (with age-group template) CT scans.
    """
    if plot_all:
        plt.rcParams.update({'figure.figsize': [16, 5]})
    else:
        plt.rcParams.update({'figure.figsize': [8, 5]})
    plt.rcParams.update({'font.size': 20})

    default_bins = np.arange(similarity_df['ssim'].min(), similarity_df['ssim'].max(),
                             0.005) if plot_all else np.arange(similarity_df['ssim'].min(), similarity_df['ssim'].max(),
                                                               0.007)

    similarity_df.sort_values(by=['ssim'], inplace=True)
    similarity_df = similarity_df[similarity_df['image_path'].isin(completeness_paths)]

    if len(similarity_df) == 0:
        print('Empty similarity dataframe.')
        return None, None

    cutoff_index = round(cutoff_percentage * len(similarity_df))

    low_ssim_df = similarity_df.iloc[:cutoff_index]
    high_ssim_df = similarity_df.iloc[cutoff_index:]

    n, bins, patches = plt.hist(
        similarity_df['ssim'],
        bins=default_bins,
        color='tab:blue' if plot_all else 'tab:orange',
        alpha=0.75,
        log=False
    )

    if plot_lowest:
        first_color_index = np.digitize(similarity_df.iloc[cutoff_index]['ssim'], bins)
        for i in range(first_color_index):
            patches[i].set_facecolor('tab:orange')
        for i in range(first_color_index, len(patches)):
            patches[i].set_facecolor('tab:green')

    plt.xlabel(plot_x_axis_label)
    plt.ylabel('Frequency')
    plt.title(plot_title)

    if plot_lowest:
        legend_elements = [Line2D([0], [0], color='tab:orange', lw=4, label='Lowest 5%'),
                           Line2D([0], [0], color='tab:green', lw=4, label='Remaining 95%')]
        plt.legend(handles=legend_elements)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    return low_ssim_df, high_ssim_df


def plot_example_scans(low_ssim_df, output_path, number=None):
    output_path.mkdir(parents=True, exist_ok=True)

    number = number if number is not None else len(low_ssim_df)
    low_ssim_df.sort_values('ssim', inplace=True)
    for sorted_index, (index, row) in enumerate(list(low_ssim_df.iterrows())[:number]):
        plotting.plot_roi(
            roi_img=row['image_path'],
            bg_img=row['template_path'],
            title=f'SSIM = {row["ssim"]:.2f}, {Path(row["image_path"]).name}',
            vmin=-100,
            vmax=1000,
            colorbar=True,
            annotate=False,
            draw_cross=False,
            alpha=0.55
        )
        plt.savefig(output_path / f'{str(sorted_index).zfill(5)}.png')
        plt.close()


def plot_histogram_and_lowest_ssim_examples(
        ssim_information_csv,
        age_group,
        base_path,
        completeness_group_name,
        output_path,
        inspect_percentage: float = 0.05,
        automate=False,
):
    completeness_paths = set()
    for group_name in completeness_group_name:
        completeness_paths = completeness_paths.union(set(json.load(open(
            base_path / f'quality_control/completeness_information/{group_name}_group_{age_group}_completeness_group.json'))))

    similarity_df = pd.read_csv(ssim_information_csv)

    likely_incomplete_groups = [CompletenessGroup.MIDDLE.value, CompletenessGroup.INCOMPLETE.value]
    plot_lowest = len(completeness_group_name) == 1 and completeness_group_name[0] not in likely_incomplete_groups

    plot_title = completeness_group_name[0] if len(completeness_group_name) == 1 else 'All groups'
    plot_title = plot_title.replace('_', ' ').capitalize()

    low_ssim_df, high_ssim = plot_histogram(
        similarity_df=similarity_df,
        completeness_paths=completeness_paths,
        plot_title=plot_title,
        plot_x_axis_label='SSIM',
        output_path=output_path,
        plot_lowest=plot_lowest,
        plot_all=len(completeness_group_name) > 1,
        cutoff_percentage=inspect_percentage
    )

    if len(completeness_group_name) > 1 or low_ssim_df is None or similarity_df is None:
        return

    if completeness_group_name[0] in likely_incomplete_groups:
        plot_example_scans(similarity_df, output_path.parent / f'{completeness_group_name[0]}_group_low_ssim_examples')
        append_faulty_registrations_file(similarity_df[['image_path']], output_path.parents[1], automate)
    else:
        plot_example_scans(low_ssim_df, output_path.parent / f'{completeness_group_name[0]}_group_low_ssim_examples')
        append_faulty_registrations_file(low_ssim_df[['image_path']], output_path.parents[1], automate)


def append_faulty_registrations_file(
        similarity_df: pd.DataFrame,
        output_path: Path,
        automate: bool = False,
):
    faulty_registrations_file = 'registration_failures.csv' if automate else 'candidate_registration_failures.csv'
    faulty_registrations_file = output_path / faulty_registrations_file

    if faulty_registrations_file.exists():
        existing_faulty_registrations = pd.read_csv(faulty_registrations_file)
        faulty_registrations = pd.concat([existing_faulty_registrations, similarity_df]).drop_duplicates()
    else:
        faulty_registrations = similarity_df.drop_duplicates()

    faulty_registrations.to_csv(faulty_registrations_file, index=False, mode='w')


def plot_all_group_similarities(
        base_path: Path,
        dataset_structure: DatasetStructure,
        inspect_percentage: float = 0.05,
        automate=False,
):
    age_groups = dataset_structure.get_age_groups(base_path)

    # First plot behaves weird, so executed beforehand and overwritten afterward.
    # plot_histogram_and_lowest_ssim_examples(
    #     ssim_information_csv=dataset_structure.SIMILARITY_INFORMATION_PATH / f'{age_groups[0]}_ssim.csv',
    #     age_group=age_groups[0],
    #     base_path=base_path,
    #     completeness_group_name=[CompletenessGroup.COMPLETE.value],
    #     output_path=dataset_structure.SIMILARITY_INFORMATION_PATH / f'plots/{age_groups[0]}_{CompletenessGroup.COMPLETE.value}_group_similarity.pdf'
    # )

    for age_group in age_groups:

        plot_histogram_and_lowest_ssim_examples(
            base_path / dataset_structure.SIMILARITY_INFORMATION_PATH / f'{age_group}_ssim.csv',
            age_group=age_group,
            base_path=base_path,
            completeness_group_name=[completeness_group.value for completeness_group in CompletenessGroup],
            output_path=base_path / dataset_structure.SIMILARITY_INFORMATION_PATH / f'plots/{age_group}_all_group_similarity.pdf',
            inspect_percentage=inspect_percentage,
            automate=automate
        )

        for completeness_group in CompletenessGroup:
            output_path = base_path / dataset_structure.SIMILARITY_INFORMATION_PATH / f'plots/{age_group}_{completeness_group.value}_group_similarity.pdf'
            plot_histogram_and_lowest_ssim_examples(
                base_path / dataset_structure.SIMILARITY_INFORMATION_PATH / f'{age_group}_ssim.csv',
                age_group=age_group,
                base_path=base_path,
                completeness_group_name=[completeness_group.value],
                output_path=output_path,
                inspect_percentage=inspect_percentage,
                automate=automate
            )
