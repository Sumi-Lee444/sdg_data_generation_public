from itertools import product
from math import floor
from typing import Any

from sdg_data_generation.feature_type_hinting import (
    DatasetFeaturesMetadata,
    FeatureCountsMap,
    FeatureValueMetadata,
    FeatureValuesCountsMap,
)


def frequencies_valid(freq_lst: list[float]) -> bool:
    """Check that all frequencies values are valid.

    Valid frequencies are such that:
    - they are non-negative
    - their sum equals 1, or, are frequencies are equal to 0

    Args:
        freq_lst: List of frequency values.

    Returns:
        True if all frequencies are non-negative and either sum to 1,
        or are all 0.

    Raises:
        ValueError: If any frequency is negative or the sum of frequencies is not 1.
    """
    for freq in freq_lst:
        if freq < 0:
            raise ValueError(f"Received negative frequency: '{freq}'")

    freq_sum = sum(freq_lst)
    if freq_sum == 0 or freq_sum == 1:
        return True
    else:
        raise ValueError(
            f"Frequencies for a given feature must sum to 1, but are instead '{sum(freq_lst)}'."
        )


def allocate_counts(
    combo_records: list[dict[str, Any]], num_to_generate: int
) -> list[int]:
    """
    Distribute counts across combinations proportionally to their frequencies.

    Args:
        combo_records (List[Dict[str, Any]]): List of combination records, each containing
            'freq_product' (float) among other keys.
        num_to_generate (int): Total number of counts to allocate.

    Returns:
        List[int]: Integer counts assigned to each combination, summing to `num_to_generate`.

    Example:
        >>> combo_records = [
             {'freq_product': 0.42, 'dataset_features_by_type': {}, 'features_name_flag_map': {}},
             {'freq_product': 0.18, 'dataset_features_by_type': {}, 'features_name_flag_map': {}},
             {'freq_product': 0.28, 'dataset_features_by_type': {}, 'features_name_flag_map': {}},
             {'freq_product': 0.12, 'dataset_features_by_type': {}, 'features_name_flag_map': {}}
         ]
        >>> allocate_counts(combo_records, 10)
        [4, 2, 3, 1]
    """
    total_freq = sum(r["freq_product"] for r in combo_records)
    counts_raw = [
        num_to_generate * (r["freq_product"] / total_freq)
        for r in combo_records
    ]
    counts_int = [floor(c) for c in counts_raw]
    remainders = [c - floor(c) for c in counts_raw]

    remainder_total = num_to_generate - sum(counts_int)
    indices_sorted = sorted(
        range(len(remainders)), key=lambda i: remainders[i], reverse=True
    )
    for i in indices_sorted[:remainder_total]:
        counts_int[i] += 1
    return counts_int


def validate_feature_frequencies(
    dataset_features_metadata: DatasetFeaturesMetadata,
) -> None:
    """
    Ensure all features have valid frequency distributions.

    Checks that every feature hasfrequencies are non-negative
    and sum to 1.

    Args:
        dataset_features_metadata (DatasetFeaturesMetadata):
            Dataset feature metadata to validate.

    Returns:
        None: This function performs validation only and does not return
        a value. Successful completion indicates that all frequencies are valid.

    Raises:
        ValueError: if a feature's frequency list is invalid.

    """
    for (
        feature_type,
        features_metadata_list,
    ) in dataset_features_metadata.items():
        for feature_metadata in features_metadata_list:
            for (
                feature_name,
                feature_values_metadata,
            ) in feature_metadata.items():
                freq_lst = [fv["frequency"] for fv in feature_values_metadata]
                try:
                    frequencies_valid(freq_lst)
                except Exception as e:
                    raise ValueError(
                        f"Invalid frequencies for feature '{feature_name}' "
                        f"in type '{feature_type}': {e}"
                    ) from e


def drop_zero_frequency_items(
    dataset_features_metadata: DatasetFeaturesMetadata,
) -> DatasetFeaturesMetadata:
    """Drop feature metadata items with frequency 0.

    Return a cleaned version of dataset_features_metadata where:
      - all feature metadata items with frequency 0 are removed
      - any feature that becomes empty is removed
      - any feature type that becomes empty is removed

    Args:
        dataset_features_metadata (DatasetFeaturesMetadata):
            The input dataset feature metadata.

    Returns:
        DatasetFeaturesMetadata:
            A cleaned up version of the dataset feature metadata.

    Example:
        >>> input = {
             "rule": [
                 {"color": [
                     {"flag": "R", "value": "red", "frequency": 0.6},
                     {"flag": "B", "value": "blue", "frequency": 0.4},
                     {"flag": "G", "value": "green", "frequency": 0},
                 ]},
                 {"type": [
                     {"flag": "p", "value": "pen", "frequency": 0},
                     {"flag": "e", "value": "eraser", "frequency": 0}
                 ]}
             ],
             "var": [
                 {"size": [
                     {"flag": "S", "value": "small", "frequency": 0},
                     {"flag": "M", "value": "medium", "frequency": 0},
                     {"flag": "L", "value": "large", "frequency": 0}
                 ]}
             ]
         }
        >>> drop_zero_frequency_items(input)
            {
                "rule": [
                    {
                        "color": [
                        {"flag": "R", "value": "red", "frequency": 0.6},
                        {"flag": "B", "value": "blue", "frequency": 0.4}
                        ]
                    }
                ]
            }
    """
    cleaned = {}

    for feature_type, feature_list in dataset_features_metadata.items():
        cleaned_features = []

        for feature in feature_list:
            cleaned_feature = {}
            for feature_name, feature_metadata_lst in feature.items():
                # Drop only zero-frequency values
                filtered_values = [
                    f_metadata
                    for f_metadata in feature_metadata_lst
                    if f_metadata["frequency"] != 0
                ]
                # Only keep the feature if some values remain
                if filtered_values:
                    cleaned_feature[feature_name] = filtered_values
            # Keep this feature block only if it has at least one feature left
            if cleaned_feature:
                cleaned_features.append(cleaned_feature)
        # Keep the whole feature type only if it has any remaining features
        if cleaned_features:
            cleaned[feature_type] = cleaned_features

    return cleaned


def compute_joint_combinations(
    dataset_features_metadata: DatasetFeaturesMetadata,
) -> list[dict[str, Any]]:
    """
    Generate all cross-type feature combinations with their joint frequencies.

    Args:
        feature_type_groups (DatasetFeaturesMetadata): Features metadata grouped by feature type.

    Returns:
        List[Dict[str, Any]]: Each record contains:
            - 'freq_product' (float): product of feature frequencies
            - 'dataset_features_by_type' (Dict[str, List[Dict[str, Any]]]): feature values grouped by type
            - 'features_name_flag_map' (Dict[str, str]): feature name → flag mapping

    Example:
        >>> dataset_features_metadata = {
                "rule": [
                    {
                        "label": [
                            {"flag": 1, "value": "LoF yes", "frequency": 0.5},
                            {"flag": 0, "value": "LoF no", "frequency": 0.5},
                        ]
                        },
                    {
                            "safe": [
                            {"flag": 1, "value": "Safe.", "frequency": 0.5},
                            {"flag": 0, "value": "Unsafe.", "frequency": 0.5},
                        ]
                    }
                ],
                "var": [
                    {
                        "length": [
                            {"flag": "short", "value": "10-15", "frequency": 0.4},
                            {"flag": "medium", "value": "20-30", "frequency": 0.3},
                            {"flag": "long", "value": "35-50", "frequency": 0.3},
                        ]
                    }
                ]
            }

        >>> compute_joint_combinations(feature_type_groups)
        [
            {'freq_product': 0.1, 'dataset_features_by_type': {'rule': [{'label': 'LoF yes'}, {'safe': 'Safe.'}], 'var': [{'length': '10-15'}]}, 'features_name_flag_map': {'label': 1, 'safe': 1, 'length': 'short'}},
            {'freq_product': 0.075, 'dataset_features_by_type': {'rule': [{'label': 'LoF yes'}, {'safe': 'Safe.'}], 'var': [{'length': '20-30'}]}, 'features_name_flag_map': {'label': 1, 'safe': 1, 'length': 'medium'}},
            {'freq_product': 0.075, 'dataset_features_by_type': {'rule': [{'label': 'LoF yes'}, {'safe': 'Safe.'}], 'var': [{'length': '35-50'}]}, 'features_name_flag_map': {'label': 1, 'safe': 1, 'length': 'long'}},
            {'freq_product': 0.1, 'dataset_features_by_type': {'rule': [{'label': 'LoF yes'}, {'safe': 'Unsafe.'}], 'var': [{'length': '10-15'}]}, 'features_name_flag_map': {'label': 1, 'safe': 0, 'length': 'short'}},
            {'freq_product': 0.075, 'dataset_features_by_type': {'rule': [{'label': 'LoF yes'}, {'safe': 'Unsafe.'}], 'var': [{'length': '20-30'}]}, 'features_name_flag_map': {'label': 1, 'safe': 0, 'length': 'medium'}},
            {'freq_product': 0.075, 'dataset_features_by_type': {'rule': [{'label': 'LoF yes'}, {'safe': 'Unsafe.'}], 'var': [{'length': '35-50'}]}, 'features_name_flag_map': {'label': 1, 'safe': 0, 'length': 'long'}},
            {'freq_product': 0.1, 'dataset_features_by_type': {'rule': [{'label': 'LoF no'}, {'safe': 'Safe.'}], 'var': [{'length': '10-15'}]}, 'features_name_flag_map': {'label': 0, 'safe': 1, 'length': 'short'}},
            {'freq_product': 0.075, 'dataset_features_by_type': {'rule': [{'label': 'LoF no'}, {'safe': 'Safe.'}], 'var': [{'length': '20-30'}]}, 'features_name_flag_map': {'label': 0, 'safe': 1, 'length': 'medium'}},
            {'freq_product': 0.075, 'dataset_features_by_type': {'rule': [{'label': 'LoF no'}, {'safe': 'Safe.'}], 'var': [{'length': '35-50'}]}, 'features_name_flag_map': {'label': 0, 'safe': 1, 'length': 'long'}},
            {'freq_product': 0.1, 'dataset_features_by_type': {'rule': [{'label': 'LoF no'}, {'safe': 'Unsafe.'}], 'var': [{'length': '10-15'}]}, 'features_name_flag_map': {'label': 0, 'safe': 0, 'length': 'short'}},
            {'freq_product': 0.075, 'dataset_features_by_type': {'rule': [{'label': 'LoF no'}, {'safe': 'Unsafe.'}], 'var': [{'length': '20-30'}]}, 'features_name_flag_map': {'label': 0, 'safe': 0, 'length': 'medium'}},
            {'freq_product': 0.075, 'dataset_features_by_type': {'rule': [{'label': 'LoF no'}, {'safe': 'Unsafe.'}], 'var': [{'length': '35-50'}]}, 'features_name_flag_map': {'label': 0, 'safe': 0, 'length': 'long'}},
        ]
    """
    feature_types = list(dataset_features_metadata.keys())

    # Step 1: Compute all combinations within each feature type
    feature_combinations_by_type = {}
    for ft in feature_types:
        features = dataset_features_metadata[ft]
        per_feature_values = []
        for feature in features:
            for feature_name, feature_metadata_lst in feature.items():
                per_feature_values.append(
                    [
                        {
                            feature_name: f_metadata["value"],
                            "flag": f_metadata["flag"],
                            "frequency": f_metadata["frequency"],
                        }
                        for f_metadata in feature_metadata_lst
                    ]
                )
        # print("per_feature_values \n")
        # pprint(per_feature_values, indent = 4)
        feature_combinations_by_type[ft] = list(product(*per_feature_values))
        # print("feature_combinations_by_type \n")
        # pprint(feature_combinations_by_type, indent=4)

    # Step 2: Cartesian product across feature types
    all_combos = list(product(*feature_combinations_by_type.values()))
    # print("all_combos \n")
    # pprint(all_combos, indent=4)

    combo_records = []
    for combo_across_types in all_combos:
        freq_product = 1.0
        dataset_features_by_type = {}
        features_name_flag_map = {}

        for ft, type_combo in zip(feature_types, combo_across_types):
            dataset_features_by_type[ft] = []
            for feature_dict in type_combo:
                feature_name = next(
                    k for k in feature_dict if k not in ("flag", "frequency")
                )
                dataset_features_by_type[ft].append(
                    {feature_name: feature_dict[feature_name]}
                )
                features_name_flag_map[feature_name] = feature_dict["flag"]
                freq_product *= feature_dict["frequency"]

        combo_records.append(
            {
                "freq_product": freq_product,
                "dataset_features_by_type": dataset_features_by_type,
                "features_name_flag_map": features_name_flag_map,
            }
        )

    return combo_records


def stratified_single_feature_value_counts(
    feature_values_metadata: list[FeatureValueMetadata],
    num_to_generate: int,
    feature_nm: str,
) -> list[FeatureValuesCountsMap]:
    """
    Compute stratified int counts for feature values based on frequency.

    This function allocates a total number of items (`num_to_generate`) across a set of
    feature values, in proportion to their specified frequencies.

    Args:
        feature_values_metadata (list[FeatureValueMetadata]):
            A list of FeatureValueMetadata describing each feature value.
            Each must include a `"frequency"` field (float), `"flag"`
            and `"value"` keys.
        num_to_generate (int):
            The total number of instances to allocate across feature values.
        feature_nm (str):
            The name of the feature (used for error reporting).

    Returns:
        list[FeatureValuesCountsMap]:
            A list of dictionaries, one per feature value, with integer `"count"`s
            that sum to `num_to_generate`. Each dictionary mirrors the structure of
            the input items and includes:
                - `"flag"`: identifier or flag of the feature value
                - `"value"`: the feature value itself
                - `"count"`: allocated integer count

    Raises:
        ValueError:
            If the provided frequencies are invalid (e.g., negative or not summing to 1).

    Example:
        >>> feature_values_metadata = [
             {"flag": "A", "value": "a", "frequency": 0.45},
             {"flag": "B", "value": "b", "frequency": 0.35},
             {"flag": "C", "value": "c", "frequency": 0.20},
         ]
        >>> stratified_single_feature_value_counts(feature_values_metadata, 10, "example")
        [
            {'flag': 'A', 'value': 'a', 'count': 5},
            {'flag': 'B', 'value': 'b', 'count': 3},
            {'flag': 'C', 'value': 'c', 'count': 2}
        ]
    """
    freq_lst = [
        feature_value_metadata["frequency"]
        for feature_value_metadata in feature_values_metadata
    ]
    try:
        frequencies_valid(freq_lst)
    except Exception as e:
        raise ValueError(
            f"Invalid frequencies for '{feature_nm}': {e}."
        ) from e
    # Use allocate_counts to get integer counts
    combo_records = [{"freq_product": f} for f in freq_lst]
    counts_int = allocate_counts(combo_records, num_to_generate)
    # Wrap counts with flags & values
    feature_value_counts_lst: list[FeatureValuesCountsMap] = []
    for fv, count in zip(feature_values_metadata, counts_int):
        feature_value_count_map = {
            "flag": fv["flag"],
            "value": fv["value"],
            "count": count,
        }
        feature_value_counts_lst.append(feature_value_count_map)

    return feature_value_counts_lst


def stratified_multi_feature_value_counts(
    dataset_features_metadata: DatasetFeaturesMetadata, num_to_generate: int
) -> list[FeatureCountsMap]:
    """
    Compute stratified counts for valid rule & var multi feature combinations.

    This function takes `DatasetFeaturesMetadata`, a nested dictionary describing
    features of different types (`'rule'` & `'var'`), each with values and
    frequencies. It:
    1. Builds all valid cross-type feature combinations with nonzero joint frequency.
    2. Calculates proportional frequencies based on the product of their components.
    3. Allocates integer counts that sum to `num_to_generate`, distributing any
       rounding remainder fairly among the top fractional remainders.


    Args:
        dataset_features_metadata (DatasetFeaturesMetadata):
            Dictionary mapping feature types ('rule', 'var') to lists of feature
            metadata dictionaries, where each maps a feature name to a list of
            fla, value,, and frequency metadata.
        num_to_generate (int):
            Total number of instances to distribute across all combinations.

    Returns:
        list[FeatureCountsMap]:
            A list of mappings, one per valid cross-type feature combination.
            Each mapping includes:
              - `"count"`: integer number of instances allocated to the combination
              - `"dataset_features_by_type"`: feature values grouped by type
              - `"features_name_flag_map"`: mapping of feature names to their flags

    Example:
        >>> input = {
             "rule": [
                 {"color": [
                     {"flag": "R", "value": "red", "frequency": 0.6},
                     {"flag": "B", "value": "blue", "frequency": 0.4}
                 ]}
             ],
             "var": [
                 {"size": [
                     {"flag": "S", "value": "small", "frequency": 0.7},
                     {"flag": "L", "value": "large", "frequency": 0.3}
                 ]}
             ]
         }
        >>> stratified_multi_feature_value_counts(input, 10)
        [
            {
                "count": 4,
                "dataset_features_by_type": {
                    "rule": [{"color": "red"}],
                    "var": [{"size": "small"}]
                },
                "features_name_flag_map": {"color": "R", "size": "S"}
            },
            {
                "count": 2,
                "dataset_features_by_type": {
                    "rule": [{"color": "red"}],
                    "var": [{"size": "large"}]
                },
                "features_name_flag_map": {"color": "R", "size": "L"}
            },
            {
                "count": 3,
                "dataset_features_by_type": {
                    "rule": [{"color": "blue"}],
                    "var": [{"size": "small"}]
                },
                "features_name_flag_map": {"color": "B", "size": "S"}
            },
            {
                "count": 1,
                "dataset_features_by_type": {
                    "rule": [{"color": "blue"}],
                    "var": [{"size": "large"}]
                },
                "features_name_flag_map": {"color": "B", "size": "L"}
            }
        ]
    """
    # Validate frequencies
    validate_feature_frequencies(dataset_features_metadata)
    # drop feature metadata items with frequency 0
    dataset_features_metadata = drop_zero_frequency_items(
        dataset_features_metadata
    )
    # Get all combinations directly from nested input
    combo_records = compute_joint_combinations(dataset_features_metadata)
    # pprint(combo_records, indent=4)
    # Allocate counts proportionally
    counts_int = allocate_counts(combo_records, num_to_generate)

    output = []
    for record, count in zip(combo_records, counts_int):
        if count > 0:
            output.append(
                {
                    "count": count,
                    "dataset_features_by_type": record[
                        "dataset_features_by_type"
                    ],
                    "features_name_flag_map": record["features_name_flag_map"],
                }
            )
    return output


# --------------------------------------------------------

input = {
    "rule": [
        {
            "color": [
                {"flag": "R", "value": "red", "frequency": 0.6},
                {"flag": "B", "value": "blue", "frequency": 0.4},
                {"flag": "C", "value": "c", "frequency": 0},
                {"flag": "D", "value": "d", "frequency": 0},
                {"flag": "E", "value": "e", "frequency": 0},
            ]
        },
        {
            "type": [
                {"flag": "F", "value": "f", "frequency": 0},
                {"flag": "G", "value": "g", "frequency": 0},
            ]
        },
        {"style": [{"flag": "M", "value": "modern", "frequency": 1}]},
    ],
    "var": [
        {
            "size": [
                {"flag": "S", "value": "small", "frequency": 0.7},
                {"flag": "L", "value": "large", "frequency": 0.3},
            ]
        }
    ],
}

out_exp = [
    {
        "count": 4,
        "dataset_features_by_type": {
            "rule": [{"color": "red"}, {"style": "modern"}],
            "var": [{"size": "small"}],
        },
        "features_name_flag_map": {"color": "R", "style": "M", "size": "S"},
    },
    {
        "count": 2,
        "dataset_features_by_type": {
            "rule": [{"color": "red"}, {"style": "modern"}],
            "var": [{"size": "large"}],
        },
        "features_name_flag_map": {"color": "R", "style": "M", "size": "L"},
    },
    {
        "count": 3,
        "dataset_features_by_type": {
            "rule": [{"color": "blue"}, {"style": "modern"}],
            "var": [{"size": "small"}],
        },
        "features_name_flag_map": {"color": "B", "style": "M", "size": "S"},
    },
    {
        "count": 1,
        "dataset_features_by_type": {
            "rule": [{"color": "blue"}, {"style": "modern"}],
            "var": [{"size": "large"}],
        },
        "features_name_flag_map": {"color": "B", "style": "M", "size": "L"},
    },
]

# out = stratified_multi_feature_value_counts(input, 10)ß
# print(out == out_exp)
# pprint(out, indent=4)
