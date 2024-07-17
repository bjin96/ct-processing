"""Classify scans into groups based on completeness measures (calculated in scan_completeness.py)"""
import json
from enum import Enum
from pathlib import Path
from statistics import median

MEDIAN_THRESHOLD = 0.4
MIDDLE_TROUGH_THRESHOLD = 0.6


class CompletenessGroup(Enum):
    COMPLETE = 'complete'
    SKULL_BASE = 'skull_base'
    SKULL_VAULT = 'skull_vault'
    MIDDLE = 'middle'
    INCOMPLETE = 'incomplete'


def create_completeness_groups(completeness_information_json, output_file: Path):
    completeness_information = json.load(open(completeness_information_json))

    complete_group = _iterate_paths(_complete_group_comparison, completeness_information)
    skull_base_group = _iterate_paths(_skull_base_group_comparison, completeness_information)
    skull_vault_group = _iterate_paths(_skull_vault_group_comparison, completeness_information)
    middle_group = _iterate_paths(_middle_group_comparison, completeness_information)
    incomplete_group = _iterate_paths(_incomplete_group_comparison, completeness_information)

    create_group_json(complete_group, output_file.parent / ('complete_group_' + output_file.name))
    create_group_json(skull_base_group, output_file.parent / ('skull_base_group_' + output_file.name))
    create_group_json(skull_vault_group, output_file.parent / ('skull_vault_group_' + output_file.name))
    create_group_json(middle_group, output_file.parent / ('middle_group_' + output_file.name))
    create_group_json(incomplete_group, output_file.parent / ('incomplete_group_' + output_file.name))


def create_group_json(group_information, output_file):
    with open(output_file, 'w') as file:
        json.dump(group_information, file)


def _complete_group_comparison(value, middle_index):
    return (
            _has_lower_data(value, middle_index)
            and _has_upper_data(value, middle_index)
            and not _has_middle_dip(value, middle_index)
    )


def _skull_base_group_comparison(value, middle_index):
    return _has_lower_data(value, middle_index) and not _has_upper_data(value, middle_index)


def _skull_vault_group_comparison(value, middle_index):
    return not _has_lower_data(value, middle_index) and _has_upper_data(value, middle_index)


def _middle_group_comparison(value, middle_index):
    """
    Create a group of scans with a median value above the MEDIAN_THRESHOLD for the upper half of completeness values
    that are not contained in skull base or vault groups.
    """
    sorted_value = sorted(value)
    return (
            not _has_lower_data(value, middle_index)
            and not _has_upper_data(value, middle_index)
            and _has_upper_data(sorted_value, middle_index)
    )


def _incomplete_group_comparison(value, middle_index):
    return (
        not _complete_group_comparison(value, middle_index)
        and not _skull_base_group_comparison(value, middle_index)
        and not _skull_vault_group_comparison(value, middle_index)
        and not _middle_group_comparison(value, middle_index)
    )


def _iterate_paths(comparison_function, completeness_information):
    group_information = {}
    for key, value in completeness_information.items():
        middle_index = round(len(value) / 2)
        if comparison_function(value, middle_index):
            group_information[key] = value
    return group_information


def _has_lower_data(value, middle_index):
    return median(value[:middle_index]) >= MEDIAN_THRESHOLD


def _has_upper_data(value, middle_index):
    return median(value[middle_index:]) >= MEDIAN_THRESHOLD


def _has_middle_dip(value, middle_index):
    return value[middle_index] < MIDDLE_TROUGH_THRESHOLD
