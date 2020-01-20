#!/usr/bin/python
#%%
""" This is main module for making predictions. After setup config.py (data source), run predict function.
There is also predict_multiple_columns function for predicting more columns. It's not necessary to run all models everytime,
so there is also function compare models, that return results of models on the test data. Then you can config again only good models.

There are working examples in main readme and also in test_it module.
"""
import sys
from pathlib import Path
import numpy as np
from prettytable import PrettyTable
import time
import os
import plotly as pl
#import cufflinks as cf
import pandas as pd
import warnings
import traceback
import argparse
import inspect

this_path = Path(__file__).resolve().parents[1]
this_path_string = str(this_path)

jupyter = 0

# If used not as a library but as standalone framework, add path to be able to import predictit if not opened in folder
sys.path.insert(0, this_path_string)

if __name__ == "__main__":
    # Just for develop reasons. Reload all libraries in jupyter everytime
    try:
        __IPYTHON__
        from IPython import get_ipython
        ipython = get_ipython()
        magic_load_ex = '%load_ext autoreload'
        magic_autoreload = '%autoreload 2'

        ipython.magic(magic_load_ex)
        ipython.magic(magic_autoreload)
        jupyter = 1

    except Exception:
        pass

import predictit
from predictit import config
import predictit.data_prep as dp

if __name__ == "__main__":
    # All the config is in config.py - rest only for people that know what are they doing
    # Add settings from command line if used
    parser = argparse.ArgumentParser(description='Prediction framework setting via command line parser!')
    parser.add_argument("--used_function", type=str, choices=['predict', 'predict_multiple_columns', 'compare_models'], help="Which function in main.py use. One predict one column, ohter more at once, the last compare models on test data")
    parser.add_argument("--data_source", type=str, choices=['csv', 'test'], help="What source of data to use")
    parser.add_argument("--csv_full_path", type=str, help="Full CSV path with suffix")
    parser.add_argument("--predicts", type=int, help="Number of predicted values - 7 by default")
    parser.add_argument("--predicted_column", type=int, help="Name of predicted column or it's index - string or int")
    parser.add_argument("--predicted_columns", type=list, help="For predict_multiple_columns function only! List of names of predicted column or it's indexes")
    parser.add_argument("--freq", type=str, help="Interval for predictions 'M' - months, 'D' - Days, 'H' - Hours")
    parser.add_argument("--freqs", type=list, help="For predict_multiple_columns function only! List of intervals of predictions 'M' - months, 'D' - Days, 'H' - Hours")
    parser.add_argument("--plot", type=bool, help="If 1, plot interactive graph")
    parser.add_argument("--date_index", type=int, help="Index of dataframe or it's name. Can be empty, then index 0 or new if no date column.")
    parser.add_argument("--return_all", type=bool, help="If 1, return more models (config.compareit) results sorted by how efficient they are. If 0, return only the best one")
    parser.add_argument("--datalength", type=int, help="The length of the data used for prediction")
    parser.add_argument("--debug", type=bool, help="Debug - print all results and all the errors on the way")
    parser.add_argument("--analyzeit", type=bool, help="Analyze input data - Statistical distribution, autocorrelation, seasonal decomposition etc.")
    parser.add_argument("--optimizeit", type=bool, help="Find optimal parameters of models")
    parser.add_argument("--repeatit", type=int, help="How many times is computation repeated")
    parser.add_argument("--other_columns", type=bool, help="If 0, only predicted column will be used for making predictions.")
    parser.add_argument("--lengths", type=bool, help="Compute on various length of data (1, 1/2, 1/4...). Automatically choose the best length. If 0, use only full length.")
    parser.add_argument("--remove_outliers", type=bool, help="Remove extraordinary values. Value is threshold for ignored values. Value means how many times standard deviation from the average threshold is far")
    parser.add_argument("--standardize", type=str, choices=['standardize', '-11', '01', 'robust'], help="Data standardization, so all columns have similiar scopes")
    parser.add_argument("--criterion", type=str, choices=['mape', 'rmse'], help="Error criterion used for model")
    parser.add_argument("--compareit", type=int, help="How many models will be displayed in final plot. 0 if only the best one.")

    # Non empty command line args
    parser_args_dict = {k: v for k, v in parser.parse_known_args()[0].__dict__.items() if v is not None}

    # Edit config.py default values with command line arguments values if exist
    for i, j in parser_args_dict.items():
        setup_query = f"if hasattr(config, {i!r}): config.{i} = {j!r}"
        exec(setup_query)

if config.debug:
    warnings.filterwarnings('once')
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

else:
    warnings.filterwarnings('ignore')

def predict(data=None, predicts=None, predicted_column=None, freq=None, return_model_criterion=None, models_parameters=None, data_source=None,
            csv_full_path=None, plot=None, used_models=None, date_index=None, return_all=None, datalength=None, data_transform=None, debug=None, analyzeit=None,
            optimizeit=None, repeatit=None, other_columns=None, lengths=None, remove_outliers=None, standardize=None, criterion=None, compareit=None, n_steps_in=None, output_shape=None):

    """Make predictions mostly on time-series data. Data input can be set up in config.py (E.G. csv type, path and name),
    data can be used as input arguments to function (this has higher priority) or it can be setup as command line arguments (highest priority).
    Values defaults are None, because are in config.py. If 

    Args:
        data (np.ndarray, pd.DataFrame): Time series. Can be 2-D - more columns.
            !!! In Numpy array use data series as rows, but in dataframe use cols !!!. If you use CSV, leave it empty. Defaults to [].
        predicts (int, optional): Number of predicted values. Defaults to None.
        predicted_column (int, str, optional): Index of predicted column or it's name (dataframe).
            If list with more values only the first one will be evaluated (use predict_multiple_columns function if you need that. Defaults to None.
        freq (str. 'H' or 'D' or 'M', optional): If date index available, resample data and predict in defined time frequency. Defaults to None.
        return_model_criterion (bool, optional): Mostly for compare_models function, return error criterions instead of predictions
            (MAPE or RMSE based on config. Defaults to 0.

    Returns:
        np.ndarray: Evaluated predicted values.
        If in setup - return all models results {np.ndarray}.
        If in setup - return interactive plot of results.

    """

    # Parse all functions parameters and it's values to edit config.py later
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)

    # Edit config.py default values with arguments values if exist
    for i in args:
        setup_query = f"if values[{i!r}] is not None and hasattr(config, {i!r}): config.{i} = values[{i!r}]"
        exec(setup_query)

    # Do not repeat actually mean evaluate once
    if not config.repeatit:
        config.repeatit = 1

    ##########################################
    ################## DATA ########### ANCHOR Data
    ##########################################

    if config.data is None:

        ############# Load CSV data #############
        if config.data_source == 'csv':
            if config.csv_test_data_relative_path:
                try:
                    data_location = this_path / 'predictit' / 'test_data'
                    csv_path = data_location / config.csv_test_data_relative_path
                    config.csv_full_path = Path(csv_path).as_posix()
                except Exception:
                    print(f"\n ERROR - Test data load failed - Setup CSV adress and column name in config \n\n")
                    raise
            try:
                config.data = pd.read_csv(config.csv_full_path, header=0).iloc[-config.datalength:, :]
            except Exception:
                print("\n ERROR - Data load failed - Setup CSV adress and column name in config \n\n")
                raise

        ############# Load SQL data #############
        elif config.data_source == 'sql':
            try:
                config.data = predictit.database.database_load(server=config.server, database=config.database, freq=config.freq, data_limit=config.datalength, last=config.last_row)
            except Exception:
                print("\n ERROR - Data load from SQL server failed - Setup server, database and predicted column name in config \n\n")
                raise

        elif config.data_source == 'test':
            config.data = predictit.test_data.generate_test_data.gen_random(config.datalength)

    if isinstance(config.data, pd.Series):
        predicted_column_index = 0
        predicted_column_name = 'Predicted Column'
        data_for_predictions_df = pd.DataFrame(data[-config.datalength:])

        if config.remove_outliers:
            data_for_predictions_df = dp.remove_outliers(data_for_predictions_df, predicted_column_index=predicted_column_index, threshold=config.remove_outliers)

        data_for_predictions = data_for_predictions_df.values

    elif isinstance(config.data, pd.DataFrame):
        data_for_predictions_df = config.data.iloc[-config.datalength:, ]

        if isinstance(config.predicted_column, str):

            predicted_column_name = config.predicted_column
            predicted_column_index = data_for_predictions_df.columns.get_loc(predicted_column_name)
        else:
            predicted_column_index = config.predicted_column
            predicted_column_name = data_for_predictions_df.columns[predicted_column_index]

        if config.date_index:

            if isinstance(config.date_index, str):
                data_for_predictions_df.set_index(config.date_index, drop=True, inplace=True)
            else:
                data_for_predictions_df.set_index(data_for_predictions_df.columns[config.date_index], drop=True, inplace=True)

            data_for_predictions_df.index = pd.to_datetime(data_for_predictions_df.index)

            if config.freq:
                data_for_predictions_df.sort_index(inplace=True)
                data_for_predictions_df.resample(config.freq).sum()

            else:
                config.freq = data_for_predictions_df.index.freq

                if config.freq is None:
                    data_for_predictions_df.reset_index(inplace=True)

        # Make predicted column index 0
        data_for_predictions_df.insert(0, predicted_column_name, data_for_predictions_df.pop(predicted_column_name))
        predicted_column_index = 0

        if not config.other_columns:

            data_for_predictions_df = dp.remove_nan_columns(data_for_predictions_df)

            if config.remove_outliers:
                data_for_predictions_df = dp.remove_outliers(data_for_predictions_df, predicted_column_index=predicted_column_index, threshold=config.remove_outliers)

            data_for_predictions_df = dp.keep_corelated_data(data_for_predictions_df)

            data_for_predictions = data_for_predictions_df.values.T

        else:
            if config.remove_outliers:
                data_for_predictions_df = dp.remove_outliers(data_for_predictions_df, predicted_column_index=predicted_column_index, threshold=config.remove_outliers)

            data_for_predictions = data_for_predictions_df[predicted_column_name].to_frame().values.T

    elif isinstance(config.data, np.ndarray):
        data_for_predictions = config.data
        predicted_column_name = 'Predicted column'

        if len(np.shape(config.data)) > 1 and np.shape(config.data)[0] != 1:
            if np.shape(config.data)[1] > np.shape(config.data)[0]:
                data_for_predictions = data_for_predictions.T
            data_for_predictions = data_for_predictions[:, -config.datalength:]

            if config.other_columns:
                # Make predicted column on index 0
                data_for_predictions[[0, config.predicted_column], :] = data_for_predictions[[config.predicted_column, 0], :]
                data_for_predictions = dp.remove_nan_columns(data_for_predictions)

                predicted_column_index = 0
                if config.remove_outliers:
                    data_for_predictions = dp.remove_outliers(data_for_predictions, predicted_column_index=predicted_column_index, threshold=config.remove_outliers)

                data_for_predictions = dp.keep_corelated_data(data_for_predictions)
            else:
                data_for_predictions = data_for_predictions[predicted_column_index]
                predicted_column_index = 0
                if config.remove_outliers:
                    data_for_predictions = dp.remove_outliers(data_for_predictions, threshold=config.remove_outliers)

            data_for_predictions_df = pd.DataFrame(data_for_predictions, columns=[predicted_column_name])

        else:
            data_for_predictions = data_for_predictions[-config.datalength:].reshape(1, -1)

            predicted_column_index = 0
            if config.remove_outliers:
                data_for_predictions = dp.remove_outliers(data_for_predictions, predicted_column_index=predicted_column_index, threshold=config.remove_outliers)

            data_for_predictions_df = pd.DataFrame(data_for_predictions.reshape(-1), columns=[predicted_column_name])

    data_shape = data_for_predictions.shape

    column_for_prediction_dataframe = data_for_predictions_df[data_for_predictions_df.columns[0]].to_frame()

    if data_for_predictions.ndim == 1:
        column_for_prediction = data_for_predictions
    else:
        column_for_prediction = data_for_predictions[predicted_column_index]

    data_abs_max = max(abs(column_for_prediction.min()), abs(column_for_prediction.max()))

    try:
        number_check = int(column_for_prediction[1])

    except Exception:
        print(f"\n ERROR - Predicting not a number datatype. Maybe bad config.predicted_columns setup.\n Predicted datatype is {type(column_for_prediction[1])} \n\n")
        raise

    if config.data_transform == 'difference':
        last_undiff_value = column_for_prediction[-1]

        for i in range(len(data_for_predictions)):
            data_for_predictions[i, 1:] = dp.do_difference(data_for_predictions[i])

        data_for_predictions = np.delete(data_for_predictions, 0, axis=1)

    if config.standardize == '01':
        data_for_predictions, final_scaler = dp.standardize(data_for_predictions, standardizer=config.standardize)
    if config.standardize == '-11':
        data_for_predictions, final_scaler = dp.standardize(data_for_predictions, standardizer=config.standardize)
    if config.standardize == 'standardize':
        data_for_predictions, final_scaler = dp.standardize(data_for_predictions, standardizer=config.standardize)
    if config.standardize == 'robust':
        data_for_predictions, final_scaler = dp.standardize(data_for_predictions, standardizer=config.standardize)

    data_shape = np.shape(data_for_predictions)
    data_length = len(column_for_prediction)

    ############# Analyze #############
    if config.analyzeit:
        predictit.analyze.analyze_data(data_for_predictions.T, window=30)

        # TODO repair decompose
        #predictit.analyze.decompose(column_for_prediction_dataframe, freq=36, model='multiplicative')

    min_data_length = 3 * config.predicts + config.repeatit * config.predicts

    if data_length < min_data_length:
        config.repeatit = 1
        min_data_length = 3 * config.predicts

    assert (min_data_length < data_length), 'To few data - set up less repeat value in settings or add more data'

    if config.lengths:
        data_lengths = [data_length, int(data_length / 2), int(data_length / 4), min_data_length + 50, min_data_length]
        #data_lengths = [k for k in data_lengths if k >= min_data_length]
    else:
        data_lengths = [data_length]

    data_number = len(data_lengths)

    models_number = len(config.used_models)
    models_names = list(config.used_models.keys())


    ##########################################
    ################ Optimize ################ ANCHOR Optimize
    ##########################################

    params_everywhere = {"predicts": config.predicts}

    if len(data_shape) > 1:
        params_everywhere["predicted_column_index"] = predicted_column_index

    for i, j in config.used_models.items():

        # If no parameters or parameters details, add it so no index errors later
        if i not in config.models_parameters:
            config.models_parameters[i] = {}

    if config.optimizeit:
        models_optimizations_time = config.used_models.copy()

        for i, j in config.models_parameters_limits.items():
            if i in config.used_models:
                try:
                    start_optimization = time.time()
                    model_kwargs = {**config.models_parameters[i], **params_everywhere}
                    best_kwargs = predictit.best_params.optimize(config.used_models[i], model_kwargs, j, data_for_predictions, predicts=config.predicts, fragments=config.fragments, iterations=config.iterations, time_limit=config.optimizeit_limit, criterion=config.criterion, name=i, details=config.optimizeit_details)

                    for k, l in best_kwargs.items():

                        config.models_parameters[i][k] = l

                except Exception:
                    if config.debug:
                        warnings.warn(f"\n \t Optimization didn't finished - {traceback.format_exc()} \n")

                finally:
                    stop_optimization = time.time()
                    models_optimizations_time[i] = (stop_optimization - start_optimization)

    # Empty boxes for results definition
    # The final result is - [repeated, model, data, results]
    results_matrix = np.zeros((config.repeatit, models_number, data_number, config.predicts))
    evaluated_matrix = np.zeros((config.repeatit, models_number, data_number))
    results_matrix.fill(np.nan)
    evaluated_matrix.fill(np.nan)

    models_time = {}

    data_end = None

    # Repeat evaluation on shifted data to eliminate randomness

    for r in range(config.repeatit):
        ################################################
        ############# Predict and evaluate ############# ANCHOR Predict
        ################################################

        for p, q in enumerate(data_lengths):

            if data_end:
                data_start = -q - data_end
            else:
                data_start = -q

            train, test = dp.split(data_for_predictions[:, data_start: data_end], predicts=config.predicts, predicted_column_index=predicted_column_index)

            for m, (n, o) in enumerate(config.used_models.items()):

                try:
                    start = time.time()

                    results_matrix[r, m, p] = o(train, **params_everywhere, **config.models_parameters[n])

                except Exception:

                    if config.debug:
                        warnings.warn(f"\n \t Error in compute {n} model on data length {p} : \n\n {traceback.format_exc()} \n")

                finally:
                    end = time.time()

                evaluated_matrix[r, m, p] = predictit.evaluate_predictions.compare_predicted_to_test(results_matrix[r, m, p], test, criterion=config.criterion)

                models_time[n] = (end - start)

        if data_end:
            data_end -= config.predicts
        else:
            data_end = -config.predicts

    # Criterion is the best of average from repetitions
    repeated_average = np.mean(evaluated_matrix, axis=0)

    model_results = np.nanmin(repeated_average, axis=1)

    # Index of the best model
    best_model_index = np.unravel_index(np.nanargmin(model_results), shape=model_results.shape)[0]

    best_model_matrix = repeated_average[best_model_index]
    best_data_index = np.unravel_index(np.nanargmin(best_model_matrix), shape=best_model_matrix.shape)[0]
    best_data_length = data_lengths[best_data_index]

    # Evaluation of the best model
    best_model_name, best_model = list(config.used_models.items())[best_model_index]

    best_model_param = config.models_parameters[best_model_name]
    best_mape = np.nanmin(model_results)

    if config.compareit:

        models_real_number = len(model_results[~np.isnan(model_results)])
        if config.compareit >= models_real_number:
            next_number = models_real_number - 1
        else:
            next_number = config.compareit

        next_models_names = [np.nan] * next_number
        next_models = [np.nan] * next_number
        next_models_data_length = [np.nan] * next_number
        next_models_params = [np.nan] * next_number
        next_models_predicts = [np.nan] * next_number

        nu_best_model_index = best_model_index
        results_copy = repeated_average.copy()
        rest_models = config.used_models.copy()
        rest_params = config.models_parameters.copy()
        next_best_model_name = best_model_name

        for i in range(next_number):

            results_copy = np.delete(results_copy, nu_best_model_index, axis=0)

            del rest_models[next_best_model_name]
            del rest_params[next_best_model_name]

            # Index of the best model
            next_model_index = np.unravel_index(np.nanargmin(results_copy), shape=results_copy.shape)[0]

            next_model_matrix = results_copy[next_model_index]

            next_data_index = np.unravel_index(np.nanargmin(next_model_matrix), shape=next_model_matrix.shape)[0]
            next_models_data_length[i] = data_lengths[next_data_index]

            # Define other good models
            next_models_names[i], next_models[i] = list(rest_models.items())[next_model_index]

            if next_models_names[i] in config.models_parameters:
                next_models_params[i] = config.models_parameters[next_models_names[i]]
            else:
                next_models_params[i] = {}

            nu_best_model_index = next_model_index
            next_best_model_name = next_models_names[i]

            ####### Evaluate next models ####### ANCHOR Evaluate next models
            try:
                next_models_predicts[i] = next_models[i](data_for_predictions[:, -next_models_data_length[i]: ], **params_everywhere, **next_models_params[i])

                if config.standardize:
                    next_models_predicts[i] = final_scaler.inverse_transform(next_models_predicts[i])

                if config.data_transform == 'difference':
                    next_models_predicts[i] = dp.inverse_difference(next_models_predicts[i], last_undiff_value)

                next_models_predicts[i][abs(next_models_predicts[i]) > 10 * data_abs_max] = np.nan

            except Exception:
                if config.debug:
                    warnings.warn(f"\n \t Error in compute {n} model on data {p}: \n\n {traceback.format_exc()} \n")

    ### Optimization of the best model ### ANCHOR Optimize final
    if config.optimizeit_final and best_model_name in config.models_parameters_limits:
        best_kwargs = predictit.best_params.optimize(best_model, best_model_param, config.models_parameters_limits[best_model_name], data_for_predictions[-best_data_length:], predicts=config.predicts, fragments=config.fragments_final, details=0, iterations=config.iterations_final, time_limit=config.optimizeit_final_limit, name=best_model_name)

        for k, l in best_kwargs.items():
            config.models_parameters[best_model_name][k] = l

    ####### Evaluate best Model ####### ANCHOR Evaluate best

    best_model_predicts = np.zeros(config.predicts)
    best_model_predicts.fill(np.nan)

    try:

        best_model_predicts = best_model(data_for_predictions[:, -best_data_length:], **params_everywhere, **best_model_param)

        if config.standardize:
            best_model_predicts = final_scaler.inverse_transform(best_model_predicts)

        if config.data_transform == 'difference':
            best_model_predicts = dp.inverse_difference(best_model_predicts, last_undiff_value)

    except Exception:
        if config.debug:
            warnings.warn(f"\n \t Error in best evaluated model predictions {n} on data {p}: {traceback.format_exc()}")

    ##########################################
    ############# Results ############# ANCHOR Table
    ##########################################

    if config.print_result:
        print(f"\n Best model is {best_model_name} \n\t with results {best_model_predicts} \n\t with model error {config.criterion} {best_mape} \n\t with data length {best_data_length} \n\t with paramters {best_model_param} \n")

    if config.print_table:

        # Definition of the table for results
        models_table = PrettyTable()
        models_table.field_names = ["Model", "Average {} error".format(config.criterion), "Time"]

        # Fill the table
        for i, j in enumerate(models_names):
            models_table.add_row([models_names[i], model_results[i], models_time[models_names[i]]])

        print(f'\n {models_table} \n')

    ### Print detailed resuts ###

    if config.debug:

        for i, j in enumerate(models_names):
            print('\n', models_names[i])

            for k in range(data_number):
                print(f"\t With data length: {data_lengths[k]}  {config.criterion} = {repeated_average[i, k]}")

            if config.optimizeit:
                print(f"\t Time to optimize {models_optimizations_time[j]} \n")
                print("Best models parameters", config.models_parameters[j])


    ###############################
    ######### Plot ######### ANCHOR Results
    ###############################

    if config.plot:

        try:
            lower_bound, upper_bound = predictit.confidence_interval.bounds(column_for_prediction, predicts=config.predicts, confidence=config.confidence)
        except Exception:
            lower_bound = upper_bound = best_model_predicts
            if config.debug:
                warnings.warn(f"\n \t Error in compute confidence interval: \n\n {traceback.format_exc()} \n")

        complete_dataframe = column_for_prediction_dataframe.iloc[-7 * config.predicts:, :]

        global last_date
        last_date = column_for_prediction_dataframe.index[-1]

        if isinstance(last_date, pd._libs.tslibs.timestamps.Timestamp):
            date_index = pd.date_range(start=last_date, periods=config.predicts + 1, freq=config.freq)[1:]
            date_index = pd.to_datetime(date_index)

        else:
            date_index = list(range(last_date + 1, last_date + config.predicts + 1))

        results = pd.DataFrame({'Best prediction': best_model_predicts, 'Lower bound': lower_bound, 'Upper bound': upper_bound}, index=date_index)

        complete_dataframe['Best prediction'] = None
        complete_dataframe['Lower bound'] = None
        complete_dataframe['Upper bound'] = None

        for i in range(len(next_models_predicts)):
            name = next_models_names[i]
            results[name] = next_models_predicts[i]
            complete_dataframe[name] = None

        last_value = complete_dataframe[predicted_column_name].iloc[-1]

        complete_dataframe = pd.concat([complete_dataframe, results])
        complete_dataframe.iloc[-config.predicts - 1] = last_value

        upper_bound = pl.graph_objs.Scatter(
            name = 'Upper Bound',
            x = complete_dataframe.index,
            y = complete_dataframe['Upper bound'],
            mode = 'lines',
            marker = dict(color = "#444"),
            line = dict(width = 0),
            fillcolor = 'rgba(68, 68, 68, 0.3)',
            fill = 'tonexty')

        trace = pl.graph_objs.Scatter(
            name = '1. {}'.format(best_model_name),
            x = complete_dataframe.index,
            y = complete_dataframe['Best prediction'],
            mode = 'lines',
            line = dict(color='rgb(51, 19, 10)', width=4),
            fillcolor = 'rgba(68, 68, 68, 0.3)',
            fill = 'tonexty')

        lower_bound = pl.graph_objs.Scatter(
            name ='Lower Bound',
            x = complete_dataframe.index,
            y = complete_dataframe['Lower bound'],
            marker = dict(color="#444"),
            line = dict(width=0),
            mode = 'lines')

        history = pl.graph_objs.Scatter(
            name = column_for_prediction_dataframe.columns[0],
            x = complete_dataframe.index,
            y = complete_dataframe[predicted_column_name],
            mode = 'lines',
            line = dict(color='rgb(31, 119, 180)', width=3),
            fillcolor = 'rgba(68, 68, 68, 0.3)',
            fill = None)

        layout = pl.graph_objs.Layout(
            yaxis = dict(title='Values'),
            title = {   'text': config.plot_name,
                        'y': 0.9 if jupyter else 0.95,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
            titlefont = {"size": 28},
            showlegend = False)

        graph_data = [lower_bound, trace, upper_bound, history]

        fig = pl.graph_objs.Figure(data=graph_data, layout=layout)

        for i in range(next_number):
            fig.add_trace(pl.graph_objs.Scatter(
                    x = complete_dataframe.index,
                    y = complete_dataframe[next_models_names[i]],
                    mode = 'lines',
                    name = '{}. {}'.format(i + 2, next_models_names[i])))

        if config.save_plot:
            if not config.save_plot_path:
                config.save_plot_path = os.path.normpath(os.path.expanduser("~/Desktop") + '/plot.html')
            pl.offline.plot(fig, filename=config.save_plot_path)
        else:
            try:
                __IPYTHON__
                pl.offline.iplot(fig)
            except Exception:
                fig.show()


    if return_model_criterion:
        return repeated_average

    if config.return_all:
        all_results = np.array(next_models_predicts.insert(0, best_model_predicts))
        return all_results

    return best_model_predicts


def predict_multiple_columns(data=None, predicted_columns=None, freqs=None, database_deploy=None, predicts=None, models_parameters=None, data_source=None,
            csv_full_path=None, plot=None, used_models=None, date_index=None, return_all=None, datalength=None, data_transform=None, debug=None, analyzeit=None,
            optimizeit=None, repeatit=None, other_columns=None, lengths=None, remove_outliers=None, standardize=None, criterion=None, compareit=None, n_steps_in=None, output_shape=None):
    """Predict multiple colums and multiple frequencions at once. Use predict function.

    Args:
        data (np.ndarray, pd.DataFrame): Time series. Can be 2-D - more columns.
            !!! In Numpy array use data series as rows, but in dataframe use cols !!!. Defaults to [].
        predicted_columns (list, optional): List of indexes of predicted columns or it's names (dataframe). Defaults to None.
        freqs (str. 'H' or 'D' or 'M', optional): If date index available, resample data and predict in defined time frequency. Defaults to [].
        database_deploy (bool, optional): Whether deploy results to database !!!
            For every database it's necessary to adjust the database function. Defaults to 0.

    Returns:
        np.ndarray: All the predicted results.
    """

    # Parse all functions parameters and it's values to edit config.py later
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)

    # Edit config.py default values with arguments values if exist
    for i in args:
        setup_query = f"if values[{i!r}] is not None and hasattr(config, {i!r}): config.{i} = values[{i!r}]"
        exec(setup_query)

    predictions_full = [np.nan] * len(config.freqs)

    for j in range(len(config.freqs)):

        predictions = [np.nan] * len(config.predicted_columns)

        for i in range(len(config.predicted_columns)):

            try:
                predictions[i] = predict(predicted_column=config.predicted_columns[i], freq=config.freqs[j])

            except Exception:
                warnings.warn(f"\n Error in making predictions on column {config.predicted_columns[i]} and freq {config.freqs[j]} - {traceback.format_exc()} \n")

        predictions_full[j] = predictions

        if config.database_deploy:
            try:
                # TODO last_day resolve
                predictit.database.database_deploy(last_date, predictions[0], predictions[1], freq=config.freqs[j])
            except Exception:
                warnings.warn(f"\n Error in database deploying on freq {j} - {traceback.format_exc()} \n")

    return predictions_full


def compare_models(data_all=None, predicts=None, predicted_column=None, freq=None, models_parameters=None, data_source=None,
            csv_full_path=None, plot=None, used_models=None, date_index=None, return_all=None, datalength=None, data_transform=None, debug=None, analyzeit=None,
            optimizeit=None, repeatit=None, other_columns=None, lengths=None, remove_outliers=None, standardize=None, criterion=None, compareit=None, n_steps_in=None, output_shape=None):
    """Function that helps to choose apropriate models. It evaluate it on test data and then return results.
    After you know what models are the best, you can use only them in functions predict() or predict_multiple_columns.
    You can define your own test data and find best modules for your process. You can pickle data if you are use it
    more often to faster loading.

    Args:
        data_all (dict): Dictionary of data names and data values (np.array). You can use data from test_data module, generate_test_data script (e.g. gen_sin()).
        data_length (int, optional): If data are pickled, length that will be used. Defaults to 1000.
        **kwargs (dict): Data specific parameters. Mostly for predicted_column value.

    """

    # Parse all functions parameters and it's values to edit config.py later
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)

    # Edit config.py default values with arguments values if exist
    for i in args:
        setup_query = f"if values[{i!r}] is not None and hasattr(config, {i!r}): config.{i} = values[{i!r}]"
        exec(setup_query)

    # If no data_all inserted, default will be used
    if config.data_all is None:
        config.data_all = {'sin': predictit.test_data.generate_test_data.gen_sin(config.datalength), 'Sign': predictit.test_data.generate_test_data.gen_sign(config.datalength), 'Random data': predictit.test_data.generate_test_data.gen_random(config.datalength)}

    ### Pickle option was removed, add if you need it...
    # if config.pickleit:
    #     from predictit.test_data.pickle_test_data import pickle_data_all
    #     import pickle
    #     pickle_data_all(config.data_all, datalength=config.datalength)

    # if config.from_pickled:

    #     script_dir = Path(__file__).resolve().parent
    #     data_folder = script_dir / "test_data" / "pickled"

    #     for i, j in config.data_all.items():
    #         file_name = i + '.pickle'
    #         file_path = data_folder / file_name
    #         try:
    #             with open(file_path, "rb") as input_file:
    #                 config.data_all[i] = pickle.load(input_file)
    #         except Exception:
    #             warnings.warn(f"\n Error: {traceback.format_exc()} \n Warning - test data not loaded - First in config.py pickleit = 1, that save the data on disk, then load from pickled. \n")

    results = {}

    for i, j in config.data_all.items():
        config.plot_name = i
        try:
            result = predictit.main.predict(data=j, return_model_criterion=1)

            results[i] = (result - np.nanmin(result)) / (np.nanmax(result) - np.nanmin(result))

        except Exception:
            warnings.warn(f"\n Comparison for data {i} didn't finished - {traceback.format_exc()} \n")
            results[i] = np.nan

    results_array = np.stack((results.values()), axis=0)

    all_data_average = np.mean(results_array, axis=0)

    models_best_results = np.nanmin(all_data_average, axis=1)
    best_compared_model = int(np.nanargmin(models_best_results))
    best_compared_model_name = list(config.used_models.keys())[best_compared_model]

    all_lengths_average = np.nanmean(all_data_average, axis=0)
    best_all_lengths_index = np.nanargmin(all_lengths_average)

    models_names = list(config.used_models.keys())

    models_table = PrettyTable()
    models_table.field_names = ["Model", "Average standardized {} error".format(config.criterion)]

    # Fill the table
    for i, j in enumerate(models_names):
        models_table.add_row([models_names[i], models_best_results[i]])

    print(f'\n {models_table} \n')

    print(f"\n\nBest model is {best_compared_model_name}")
    print(f"\n\nBest data length index is {best_all_lengths_index}")


if __name__ == "__main__" and config.used_function:
    if config.used_function == 'predict':
        predict()

    elif config.used_function == 'predict_multiple_columns':
        predict_multiple_columns()

    elif config.used_function == 'compare_models':
        compare_models()
