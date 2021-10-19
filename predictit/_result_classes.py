"""This module define classes (types) that main functions will return so it's possible to use
static type analysis, intellisense etc."""

from __future__ import annotations

import pandas as pd
import numpy as np

from .configuration import Config


class Misc:
    def __init__(
        self,
        evaluated_matrix: np.ndarray,
    ):
        self.evaluated_matrix = evaluated_matrix


class Tables:
    def __init__(
        self,
        simple: str,
        detailed: str,
        simple_table_df: pd.DataFrame,
        detailed_table_df: pd.DataFrame,
        time: str = None,
    ):
        self.simple = simple
        self.detailed = detailed
        self.time = time
        self.simple_table_df = simple_table_df
        self.detailed_table_df = detailed_table_df


class Optimization:
    def __init__(
        self,
        optimized_variable: str,
        optimized_options: str,
        best_value: str,
        values_results_df: pd.DataFrame,
        best_values_for_models: dict,
        all_models_results_df: pd.DataFrame = None,
        all_models_predictions_df: pd.DataFrame = None,
    ):
        self.optimized_variable = optimized_variable
        self.optimized_options = optimized_options
        self.best_value = best_value
        self.values_results = values_results_df
        self.best_values_for_models = best_values_for_models
        self.all_models_results_df = all_models_results_df
        self.all_models_predictions_df = all_models_predictions_df


class ComparisonStandardized:
    def __init__(
        self,
        best_model_name_standardized: str,
        best_optimized_value_standardized: str,
        optimization_standardized: Optimization | None = None,
    ):
        self.best_model_name_standardized = best_model_name_standardized
        self.best_optimized_value_standardized = best_optimized_value_standardized
        self.optimization_standardized = optimization_standardized


class Comparison:
    def __init__(
        self,
        results_df: pd.DataFrame,
        best_model_name: str,
        best_optimized_value: str,
        tables: Tables,
        all_models_results: dict,
        standardized_results: ComparisonStandardized,
        optimization: Optimization | None = None,
    ):
        self.results_df = results_df
        self.best_model_name = best_model_name
        self.best_optimized_value = best_optimized_value
        self.tables = tables
        self.all_models_results = all_models_results
        self.standardized_results = standardized_results
        self.optimization = optimization


class Result:
    def __init__(
        self,
        best_prediction: pd.Series,
        best_model_name: str,
        predictions: pd.DataFrame,
        results_df: pd.DataFrame,
        results: dict,
        with_history: pd.DataFrame,
        tables: Tables,
        config: Config,
        misc: Misc,
        optimization: Optimization | None = None,
        hyperparameter_optimization_kwargs: dict | None = None,
    ):
        self.best_prediction = best_prediction
        self.predictions = predictions
        self.best_model_name = best_model_name
        self.results_df = results_df
        self.results = results
        self.with_history = with_history
        self.tables = tables
        self.config = (config,)
        self.misc = misc
        self.optimization = optimization
        self.hyperparameter_optimization_kwargs = hyperparameter_optimization_kwargs


class Multiple:
    def __init__(
        self,
        best_predictions_dataframes: dict[str, pd.DataFrame],
        results: dict[str, Result],
    ):
        self.best_predictions_dataframes = best_predictions_dataframes
        self.results = results


class BestInput:
    def __init__(
        self,
        best_data_dict: dict[str, str],
        tables: dict[str, str],
        results: dict[str, Result],
    ):
        self.best_data_dict = best_data_dict
        self.tables = tables
        self.results = results
