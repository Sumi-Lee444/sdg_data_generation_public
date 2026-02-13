from typing import Literal, TypedDict

# --------------------------------------------------
# Typing modeled after dataset_features_schema.yaml
# --------------------------------------------------
DatasetName = Literal["emp"]
FeatureType = Literal["rule", "var"]
FeatureFlag = str
FeatureValue = str
FeatureFrequency = float


class FeatureName(str):
    """Nonempty feature name."""

    def __new__(cls, name: str):
        if not name:
            raise ValueError("FeatureName cannot be empty.")
        return str.__new__(cls, name)


class FeatureNameAndType(str):
    """Concatenated FeatureName and FeatureType, separated by '_'."""

    def __new__(cls, feature_name: FeatureName, feature_type: FeatureType):
        feature_and_type = f"{feature_name}_{feature_type}"
        return str.__new__(cls, feature_and_type)


class FeatureValueMetadata(TypedDict):
    """Represents a single value of a Feature, along with metadata."""

    flag: FeatureFlag
    value: FeatureValue
    frequency: FeatureFrequency


FeatureMetadata = dict[FeatureName, list[FeatureValueMetadata]]
DatasetFeaturesMetadata = dict[FeatureType, list[FeatureMetadata]]
DatasetFeaturesCatalog = dict[DatasetName, DatasetFeaturesMetadata]


# --------------------------------------------------------------
# Typing for features to render AttrPrompt Jinja prompt templates
# --------------------------------------------------------------
FeatureNameValueMap = dict[FeatureName, FeatureValue]
FeaturesNameFlagMap = dict[FeatureName, FeatureFlag]
DatasetFeaturesByType = dict[FeatureType, list[FeatureNameValueMap]]
"""Mapping of feature types ('rule', 'var') to name-value maps for rendering."""


class FeatureCountsMap(TypedDict):
    count: int
    dataset_features_by_type: DatasetFeaturesByType
    features_name_flag_map: FeaturesNameFlagMap


FeaturesForPrompt = dict[FeatureNameAndType, FeatureValue]



# --------------------------------------------------------------
# Typing for features for Zero-Shot, Few-Shot, Finetune
# --------------------------------------------------------------
class FeatureValuesCountsMap(TypedDict):
    flag: FeatureFlag
    value: FeatureValue
    count: int

