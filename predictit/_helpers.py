"""Internal module for some helping functions across library."""

from __future__ import annotations

import mylogging

import pandas as pd

from .configuration import Config


def logger_init_from_config(logger_config: Config.Output.LoggerSubconfig) -> None:
    mylogging.outer_warnings_filter(logger_config.ignored_warnings, logger_config.ignored_warnings_class_type)
    mylogging.config.BLACKLIST = logger_config.ignored_warnings
    mylogging.config.OUTPUT = logger_config.logger_output
    mylogging.config.LEVEL = logger_config.logger_level
    mylogging.config.FILTER = logger_config.logger_filter
    mylogging.config.COLORIZE = logger_config.logger_color
    mylogging._misc.filter_warnings()


def parse_config(config: Config | dict | None, config_default: Config, kwargs) -> Config:

    if config is None:
        config = config_default.copy()

    if isinstance(config, dict):
        update_config = config
        config = config_default.copy()
        config.update(update_config)

    elif isinstance(config, Config):
        config = config.copy()

    if config.general.use_config_preset and config.general.use_config_preset != "none":
        updated_config = config.presets[config.general.use_config_preset]
        config.update(updated_config)

    config.update(kwargs)

    return config


def sort_df_index(df: pd.DataFrame, by: str) -> None:
    if by == "model":
        df["for_sort"] = df.index.get_level_values(0).str.lower()
        df.sort_values("for_sort", inplace=True)
        df.drop(["for_sort"], axis=1, inplace=True)

    elif by == "error":
        df.sort_values("Model error", inplace=True)
