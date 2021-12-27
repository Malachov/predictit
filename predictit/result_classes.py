"""This module define classes (types) that main functions will return so it's possible to use it in type hints,
,intellisense etc."""

from __future__ import annotations

from typing import Callable, Any
from dataclasses import dataclass, field
import logging

from typing_extensions import Literal
import pandas as pd
import numpy as np

from .configuration import Config
from .models import models_keys


@dataclass
class Tables:
    # TODO check if store original table objects also is possible and whether correct type hints
    """Class with tables and dataframes for creating tables."""
    simple: str
    detailed: str
    time: str
    simple_table_df: pd.DataFrame
    detailed_table_df: pd.DataFrame
    time_df: pd.DataFrame


@dataclass
class ConfigOptimizationSingle:

    variable: str
    values: list[Any]
    best_value: str
    best_values_dict: dict[models_keys, np.ndarray]
    values_results: pd.DataFrame
    best_results_df: pd.DataFrame
    all_results_df: pd.DataFrame
    best_predictions: pd.DataFrame
    all_predictions: pd.DataFrame
    best_model_name: str
    all_results: dict[str, Result]


@dataclass
class ConfigOptimization:
    best_values: dict[str, Any]
    results_dict: dict[str, ConfigOptimizationSingle]
    global_results: pd.DataFrame
    global_predictions: pd.DataFrame


# @dataclass
# class ComparisonStandardized:

#     best_model_name_standardized: str
#     best_optimized_value_standardized: str
#     config_optimization_standardized: ConfigOptimization | None = None


# @dataclass
# class Comparison:
#     results_df: pd.DataFrame
#     best_model_name: str
#     best_optimized_value: str
#     tables: Tables
#     all_models_results: dict[str, Result]
#     standardized_results: ComparisonStandardized
#     config_optimization: ConfigOptimization | None = None


@dataclass
class ResultDetails:
    prediction_index: pd.core.indexes.datetimes.DatetimeIndex | pd.core.indexes.numeric.Int64Index
    history: pd.DataFrame
    last_value: float
    test: pd.Series


@dataclass
class Result:
    best_prediction: pd.Series
    best_model_name: str
    predictions: pd.DataFrame
    results_df: pd.DataFrame
    details_df: pd.DataFrame
    results: dict[str, Result]
    with_history: pd.DataFrame
    tables: Tables
    config: Config
    details: ResultDetails
    hyperparameter_optimization: dict[str, HyperparameterOptimization] | None = None
    internal_result: None | dict[str, Any] = None


@dataclass
class Model:
    model: str = ""
    model_error: float = np.inf
    warnings_list: list[dict[str, Any]] = field(default_factory=list)
    logs_list: list[logging.LogRecord] = field(default_factory=list)
    model_time: float = np.nan
    unstandardized_model_error: float = np.inf
    prediction: np.ndarray | None = None
    test_errors: np.ndarray | None = None
    hyperparameter_optimization: HyperparameterOptimization | None = None
    trained_model: Callable | None = None
    memory_peak_MB: str = "Not configured"


@dataclass
class Multiple:
    best_predictions_dataframes: dict[str, pd.DataFrame]
    results: dict[str, Result]


@dataclass
class BestInput:
    best_data_dict: dict[str, str]
    tables: dict[str, str]
    results: dict[str, Result]


@dataclass
class AnalysisResult:
    """Optimization result with best parameters and some extra values for analysis."""

    error_criterion: str
    boosted: Literal[0, 1, 2]
    worst_params: None | dict = field(default_factory=dict)
    default_params: dict = field(default_factory=dict)
    best_result: float = np.inf
    default_result: None | float = None
    worst_result: None | float = -np.inf
    all_boosted_results: dict = field(default_factory=dict)
    boosted_result_description: None | pd.DataFrame = None
    time: float = np.nan


@dataclass
class HyperparameterOptimization:
    """Optimization result with best parameters and some extra values for analysis."""

    analysis: AnalysisResult
    best_params: dict = field(default_factory=dict)
