from pprint import pprint
from typing import Literal
from datetime import datetime

import pandas as pd
from duckdb import DuckDBPyConnection
from jinja2 import Template
from sdg_utils.db import (
    DB_PATH,
    ddb_append_rows_to_table,
    ddb_conn_mn,
    ddb_create_empty_table_from_cols,
)
from sdg_utils.llm_call import LlmCall
from sdg_utils.utility_funcs import get_prompt_template, load_yaml

from sdg_data_generation.feature_type_hinting import (
    DatasetFeaturesByType,
    DatasetFeaturesCatalog,
    DatasetFeaturesMetadata,
    DatasetName,
    FeaturesForPrompt,
    FeatureValuesCountsMap,
)
from sdg_data_generation.stratify_funcs import (
    stratified_multi_feature_value_counts,
    stratified_single_feature_value_counts,
)


def get_dataset_features_metadata(
    *, dataset_features_catalog: DatasetFeaturesCatalog, dataset_nm: str
) -> DatasetFeaturesMetadata:
    """
    Load the DatasetFeatures for a specified dataset from a YAML file.

    Args:
        dataset_features_catalog (DatasetFeaturesCatalog): Features catalog for all datasets.
        dataset_nm (str): Name of the dataset whose
            features metadata should be extracted.

    Returns:
        DatasetFeaturesMetadata: A class consisting of a dictionary
            of features metadata by type, for the specified dataset.

    Raises:
        KeyError: If the specified dataset is not found in the DatasetFeaturesCatalog.
    """
    if dataset_nm not in dataset_features_catalog:
        raise KeyError(
            f"Dataset '{dataset_nm}' not found in DatasetFeaturesCatalog keys '{dataset_features_catalog.keys()}."
        )

    return dataset_features_catalog[dataset_nm]


def format_features_for_jinja(
    dataset_features_by_type: DatasetFeaturesByType,
) -> FeaturesForPrompt:
    """Take a DatasetFeaturesByType and format into FeaturesForPrompt for jinja templates."""
    features_for_prompt = {}
    for f_type, feature_n_v_map_lst in dataset_features_by_type.items():
        for feature_n_v_map in feature_n_v_map_lst:
            for f_name, f_value in feature_n_v_map.items():
                features_for_prompt[f"{f_name}_{f_type}"] = f_value
    return features_for_prompt


def render_prompt(
    *, features_for_prompt: FeaturesForPrompt, prompt_template_str: str
) -> str:
    """
    Render a Jinja2 prompt template using the values from a configuration.

    Args:
        features_for_prompt: Config of feature values for a specific dataset with
            rules and values to populate the prompt.
        prompt_template_str (str): Jinja2 prompt templae string.

    Returns:
        str: Rendered prompt.

    Raises:
        RuntimeError: If template rendering fails for any other reason.
    """
    try:
        prompt_template = Template(prompt_template_str)
        return prompt_template.render(**features_for_prompt)
    except Exception as e:
        raise RuntimeError(f"Error rendering prompt Jinja template: {e}")


def format_table_cols_lst(
    dataset_features_metadata: DatasetFeaturesMetadata, syn_data_col_nm: str
) -> list[str]:
    """Format feature names as column names list to use for creating the
    table that stores the synthetic data."""
    cols_lst = [syn_data_col_nm]
    for _, f_metadata_lst in dataset_features_metadata.items():
        for f_metadata in f_metadata_lst:
            for f_name, __ in f_metadata.items():
                cols_lst.append(f_name)
    return cols_lst


def generate_and_append(
    *,
    num_to_generate: int,
    prompt: str,
    llm_model_nm: str,
    table_nm: str,
    table_cols_lst: list[str],
    features_flag_map: dict,
    sdg_config: dict,
    conn: DuckDBPyConnection,
) -> None:
    """
    Generate synthetic data using an LLM and append it to a DuckDB table in batches.

    Args:
        num_to_generate (int):
            Number of synthetic samples to generate.
        prompt (str):
            Prompt to pass to the LLM.
        llm_model_nm (str):
            LLM model name.
        table_nm (str):
            Target DuckDB table name.
        table_cols_lst (list[str]):
            Columns of the table including synthetic data column.
        features_flag_map (dict):
            Feature flags to include in each row.
        sdg_config (dict):
            Config with 'llm_client', 'batch_size', and 'syn_data_col_nm'.
        conn (DuckDBPyConnection):
            Active DuckDB connection.

    Returns:
        None
    """
    llm_client, batch_size, syn_data_col_nm = (
        sdg_config["llm_client"],
        sdg_config["batch_size"],
        sdg_config["syn_data_col_nm"],
    )
    syn_batch_df = pd.DataFrame(columns=table_cols_lst)
    for i in range(num_to_generate):
        #print(
        #    f"\n-----------------------\nMaking LLM call with prompt: \n{prompt}"
        #)
        syn_data = llm_client.make_call(
            user_prompt=prompt, model_nm=llm_model_nm
        )
        #print(f"\n---------------------\nGenerated Data: \n{syn_data}")
        new_row = {**features_flag_map, syn_data_col_nm: syn_data}
        syn_batch_df = pd.concat(
            [syn_batch_df, pd.DataFrame([new_row])], ignore_index=True
        )
        if syn_batch_df.shape[0] >= batch_size:
            ddb_append_rows_to_table(
                table_nm=table_nm, to_append_df=syn_batch_df, conn=conn
            )
            print(f"Appended batch of {batch_size} at time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
            syn_batch_df = pd.DataFrame(columns=table_cols_lst)
    if syn_batch_df.shape[0] >= 1:
        ddb_append_rows_to_table(
            table_nm=table_nm, to_append_df=syn_batch_df, conn=conn
        )


def get_label_values_metadata(
    label_nm: str, dataset_features_metadata: dict
) -> list[FeatureValuesCountsMap]:
    """Get a list of metadata of all the label values.

    From a list of rules, return the list of dicts containing the
    LabelValueCountsMap for the rule with 'label_nm'.

    Raises:
        ValueError: If a rule dict format is invalid or the label is not found.

    """
    rules_lst = dataset_features_metadata["rule"]
    for rule_dict in rules_lst:
        if len(rule_dict) > 1:
            raise ValueError(
                f"Rule structure invalid, \
                multiple keys in a rule feature dict: '{rule_dict.keys()}"
            )
        if label_nm in rule_dict:
            return rule_dict[label_nm]
    # If label not found
    raise ValueError(f"Feature '{label_nm}' not found.")


def run_sdg(
    total_num_to_generate: int,
    sdg_method: str,
    llm_model_nm: str,
    table_nm: str,
    prompt_template_str: str,
    dataset_features_metadata: dict,
    sdg_config: dict,
    conn: DuckDBPyConnection,
) -> None:
    """
    Generate synthetic data using only the main label feature.

    Generates data using zero-shot, few-shot, or finetune method.
    Appends the generated data directly to the specified database table.

    Args:
        total_num_to_generate (int): Number of synthetic records to generate.
        sdg_method (str): SDG method to use ('zero_shot', 'few_shot', 'finetune').
        llm_model_nm (str): Name of the LLM model to use.
        table_nm (str): Target table name in the database.
        prompt_template_str (str): Jinja2 template string for prompt generation.
        dataset_features_metadata (dict): Metadata describing dataset features.
        sdg_config (dict): Configuration for SDG generation, including column names and batch size.
        conn (DuckDBPyConnection): Database connection.

    Returns:
        None
    """
    syn_data_col_nm, label_feature_nm = (
        sdg_config["syn_data_col_nm"],
        sdg_config["label_feature_nm"],
    )
    table_cols_lst = [label_feature_nm, syn_data_col_nm]
    ddb_create_empty_table_from_cols(
        table_nm=table_nm,
        table_cols_metadata=table_cols_lst,
        add_uuid_col = True,
        conn=conn
        )
    label_values_metadata = get_label_values_metadata(
        label_nm=label_feature_nm,
        dataset_features_metadata=dataset_features_metadata,
    )
    label_values_counts_map_lst = stratified_single_feature_value_counts(
        label_values_metadata, total_num_to_generate, label_feature_nm
    )

    for label_values_counts_map in label_values_counts_map_lst:
        num_to_generate = label_values_counts_map["count"]
        label_flag = label_values_counts_map["flag"]
        label_value = label_values_counts_map["value"]
        features_flag_map = {f"{label_feature_nm}": label_flag}
        features_for_prompt = {f"{label_feature_nm}_rule": label_value}
        prompt = render_prompt(
            features_for_prompt=features_for_prompt,
            prompt_template_str=prompt_template_str,
        )

        generate_and_append(
            num_to_generate=num_to_generate,
            prompt=prompt,
            llm_model_nm=llm_model_nm,
            table_nm=table_nm,
            table_cols_lst=table_cols_lst,
            features_flag_map=features_flag_map,
            sdg_config=sdg_config,
            conn=conn,
        )


def run_sdg_attrprompt(
    total_num_to_generate: int,
    llm_model_nm: str,
    table_nm: str,
    prompt_template_str: str,
    dataset_features_metadata: dict,
    sdg_config: dict,
    conn: DuckDBPyConnection,
) -> None:
    """
    Generate synthetic data using the Attrprompt SDG method.

    Appends the generated data directly to the specified database table.

    Args:
        total_num_to_generate (int): Total number of synthetic records to generate.
        llm_model_nm (str): LLM model to use for generation.
        table_nm (str): Database table name.
        prompt_template_str (str): Jinja2 template string for prompts.
        dataset_features_metadata (dict): Feature metadata for the dataset.
        sdg_config (dict): Configuration for SDG generation.
        conn (DuckDBPyConnection): Database connection.

    Returns:
        None
    """
    syn_data_col_nm = sdg_config["syn_data_col_nm"]
    table_cols_lst = format_table_cols_lst(
        dataset_features_metadata, syn_data_col_nm
    )
    ddb_create_empty_table_from_cols(
        table_nm=table_nm,
        table_cols_metadata=table_cols_lst,
        add_uuid_col = True,
        conn=conn
    )
    feature_counts_map_lst = stratified_multi_feature_value_counts(
        dataset_features_metadata, total_num_to_generate
    )
    print("feature_counts_map_lst \n")
    pprint(feature_counts_map_lst, indent=4)
    for feature_counts_map in feature_counts_map_lst:
        num_to_generate = feature_counts_map["count"]
        features_flag_map = feature_counts_map["features_name_flag_map"]
        features_by_type = feature_counts_map["dataset_features_by_type"]
        features_for_prompt = format_features_for_jinja(features_by_type)
        prompt = render_prompt(
            features_for_prompt=features_for_prompt,
            prompt_template_str=prompt_template_str,
        )
        print(f"\n------------------\nfeatures_flag_map")
        pprint(features_flag_map)
        print(f"\n------------------\nnum to generate: {num_to_generate}\n")
        generate_and_append(
            num_to_generate=num_to_generate,
            prompt=prompt,
            llm_model_nm=llm_model_nm,
            table_nm=table_nm,
            table_cols_lst=table_cols_lst,
            features_flag_map=features_flag_map,
            sdg_config=sdg_config,
            conn=conn,
        )


def generate_syn_data(
    dataset_nm: DatasetName,
    total_num_to_generate: int,
    sdg_method: Literal["zero_shot", "few_shot", "finetune", "attr_prompt"],
    llm_model_nm: str,
    table_nm: str,
    prompt_template_path: str,
    dataset_features_catalog: DatasetFeaturesCatalog,
    sdg_config: dict,
) -> None:
    """
    Generate synthetic data for a specified dataset & write it to a DuckDB table.

    This function generates synthetic data using the appropriate SDG 
    (synthetic data generation) method, based on `sdg_method` parameter:
    - 'zero_shot', 'few_shot', 'finetune': uses label-only SDG via 
        `run_sdg` function.
    - 'attr_prompt': uses multi-feature SDG via `run_sdg_attrprompt` function.

    A DuckDB connection is opened internally using a context manager
    (`with ddb_conn_mn(DB_PATH) as conn:`), ensuring the connection is
    closed automatically after the synthetic data is written to DuckDB.

    Args:
        dataset_nm (DatasetName): Name of the dataset to generate data for.
        total_num_to_generate (int): Total number of synthetic records to create.
        sdg_method (Literal["zero_shot", "few_shot", "finetune", "attr_prompt"]):
            SDG generation method.
        llm_model_nm (str): Name of the LLM model to use.
        table_nm (str): Target table name in the database.
        prompt_template_path (str): Path to the Jinja2 prompt template file.
        dataset_features_catalog (DatasetFeaturesCatalog):
            Catalog containing dataset feature metadata.
        sdg_config (dict): Configuration for SDG generation, including column
            names and batch size.

    Raises:
        KeyError: If `sdg_method` is not a valid option.

    Returns:
        None
    """
    prompt_template_str = get_prompt_template(prompt_template_path)
    dataset_features_metadata = get_dataset_features_metadata(
        dataset_features_catalog=dataset_features_catalog,
        dataset_nm=dataset_nm,
    )
    with ddb_conn_mn(DB_PATH) as conn:
        if sdg_method in ["zero_shot", "few_shot", "finetune"]:
            run_sdg(
                total_num_to_generate=total_num_to_generate,
                sdg_method=sdg_method,
                llm_model_nm=llm_model_nm,
                table_nm=table_nm,
                prompt_template_str=prompt_template_str,
                dataset_features_metadata=dataset_features_metadata,
                sdg_config=sdg_config,
                conn=conn,
            )
        elif sdg_method == "attr_prompt":
            run_sdg_attrprompt(
                total_num_to_generate=total_num_to_generate,
                llm_model_nm=llm_model_nm,
                table_nm=table_nm,
                prompt_template_str=prompt_template_str,
                dataset_features_metadata=dataset_features_metadata,
                sdg_config=sdg_config,
                conn=conn,
            )
        else:
            raise KeyError(f"Invalid sdg_method: {sdg_method}")


# --------------------------------------------------------------
# Test Funcs
# --------------------------------------------------------------

# -------- general settings --------
features_catalog_yaml_path = "config/dataset_features_catalog.yaml"
dataset_features_catalog = load_yaml(features_catalog_yaml_path)
llm_client = LlmCall()
batch_size = 50
syn_data_col_nm = "obs"
label_feature_nm = "label"
# config
sdg_config = {
    "syn_data_col_nm": syn_data_col_nm,
    "label_feature_nm": label_feature_nm,
    "batch_size": batch_size,
    "llm_client": llm_client,
}
# -----------------------------------
# dataset vars
dataset_nm = "process_safety_obs"
sdg_method = "attr_prompt"
table_nm = "attrfs_label_1"
llm_model_nm = "qwen3-vl:30b" 
prompt_template_path = "prompts/attr_prompt.txt"
# num
total_num_to_generate = 100
# -----------------------------------
generate_syn_data(
    dataset_nm=dataset_nm,
    total_num_to_generate=total_num_to_generate,
    sdg_method=sdg_method,
    llm_model_nm=llm_model_nm,
    table_nm=table_nm,
    prompt_template_path=prompt_template_path,
    dataset_features_catalog=dataset_features_catalog,
    sdg_config=sdg_config,
)