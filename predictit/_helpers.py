# import warnings

import mylogging

from .configuration import Config


def logger_init_from_config(logger_config: Config.Output.LoggerSubconfig):
    mylogging.outer_warnings_filter(
        logger_config.ignored_warnings, logger_config.ignored_warnings_class_type
    )
    mylogging.config.BLACKLIST = logger_config.ignored_warnings
    mylogging.config.OUTPUT = logger_config.logger_output
    mylogging.config.LEVEL = logger_config.logger_level
    mylogging.config.FILTER = logger_config.logger_filter
    mylogging.config.COLORIZE = logger_config.logger_color
    mylogging._misc.filter_warnings()
    pass
