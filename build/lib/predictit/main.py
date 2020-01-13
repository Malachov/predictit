#!/usr/bin/python

""" This is main module for making predictions. After setup config.py (data source), run predict function.
There is also predict_multiple function for predicting more columns. It's not necessary to run all models everytime,
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

this_path = Path(__file__).resolve().parents[1]
this_path_string = str(this_path)

# If used not as a library but as standalone framework, add path to be able to import predictit
sys.path.insert(0, this_path_string)

import predictit
from predictit import config
import predictit.data_prep as dp

# All the config is in config.py - rest only for people that know what are they doing


def predict(data=[], predicts=None, predicted_column=None, freq=None, return_model_criterion=0):
    """Make predictions mostly on time-series data. Data input can be set up in config.py (E.G. csv type, path and name)
    or data can be used as input arguments to function (this has higher priority).

    Args:
        data (np.ndarray, pd.DataFrame): Time series. Can be 2-D - more columns.
            !!! In Numpy array use data series as rows, but in dataframe use cols !!!. Defaults to [].
        predicts (int, optional): Number of predicted values. Defaults to None.
        predicted_column (int, str, optional): Index of predicted column or it's name (dataframe).
            If list with more values only the first one will be evaluated (use predict_multiple function if you need that. Defaults to None.
        freq (str. 'H' or 'D' or 'M', optional): If date index available, resample data and predict in defined time frequency. Defaults to None.
        return_model_criterion (bool, optional): Mostly for compare_models function, return error criterions instead of predictions
            (MAPE or RMSE based on config. Defaults to 0.

    Returns:
        np.ndarray: Evaluated predicted values.
        If in setup - return all models results {np.ndarray}.
        If in setup - return interactive plot of results.

    """

    if not predicts:
        predicts = config.predicts

    if not predicted_column:
        if not config.predicted_columns:
            predicted_column = 0
        elif config.predicted_columns and isinstance(config.predicted_columns, list):
            predicted_column = config.predicted_columns[0]
        else:
            predicted_column = config.predicted_columns

    if not freq:
        if not config.freqs:
            freq = None
        else:
            freq = config.freqs
        if config.freqs and isinstance(config.freqs, list):
            freq = config.freqs[0]

    if not config.repeatit:
        config.repeatit = 1

    if config.debug:
        warnings.filterwarnings('once')
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)

    else:
        warnings.filterwarnings('ignore')

    ##########################################
    ################## DATA ########### ANCHOR Data
    ##########################################

    if not len(data):

        ############# Load CSV data #############
        if config.data_source == 'csv':
            if config.csv_from_test_data_name:
                try:
                    data_location = this_path / 'predictit' / 'test_data'
                    csv_adress = data_location / config.csv_from_test_data_name
                    config.csv_adress = Path(csv_adress).as_posix()
                except Exception:
                    print(f"\n ERROR - Data load failed - Setup CSV adress and column name in config \n\n")
                    raise
            try:
                data = pd.read_csv(config.csv_adress, header=0).iloc[-config.datalength:, :]
            except Exception:
                print("\n ERROR - Data load failed - Setup CSV adress and column name in config \n\n")
                raise

        ############# Load SQL data #############
        if config.data_source == 'sql':
            try:
                data = predictit.database.database_load(server=config.server, database=config.database, freq=freq, data_limit=config.datalength, last=config.last_row)
            except Exception:
                print("\n ERROR - Data load from SQL server failed - Setup server, database and predicted column name in config \n\n")
                raise

        if config.data_source == 'test':
            data = predictit.test_data.data_test.gen_random(config.datalength)


    if isinstance(data, pd.Series):
        predicted_column_index = 0
        predicted_column_name = 'Predicted Column'
        data_for_predictions_df = pd.DataFrame(data[-config.datalength:])

        if config.remove_outliers:
            data_for_predictions_df = dp.remove_outliers(data_for_predictions_df, predicted_column_index=predicted_column_index, threshold=config.remove_outliers)

        data_for_predictions = data_for_predictions_df.values

    if isinstance(data, pd.DataFrame):
        data_for_predictions_df = data.iloc[-config.datalength:, ]

        if isinstance(predicted_column, str):
            predicted_column_name = predicted_column
            predicted_column_index = data_for_predictions_df.columns.get_loc(predicted_column_name)
        else:
            predicted_column_index = predicted_column
            predicted_column_name = data_for_predictions_df.columns[predicted_column_index]

        if config.date_index:

            if isinstance(config.date_index, str):
                data_for_predictions_df.set_index(config.date_index, drop=True, inplace=True)
            else:
                data_for_predictions_df.set_index(data_for_predictions_df.columns[config.date_index], drop=True, inplace=True)

            data_for_predictions_df.index = pd.to_datetime(data_for_predictions_df.index)

            if freq:
                data_for_predictions_df.sort_index(inplace=True)
                data_for_predictions_df.resample(freq).sum()

            else:
                freq = data_for_predictions_df.index.freq

                if freq is None:
                    data_for_predictions_df.reset_index(inplace=True)

        # Make predicted column index 1
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

    if isinstance(data, np.ndarray):
        data_for_predictions = data
        predicted_column_name = 'Predicted column'

        if len(np.shape(data)) > 1 and np.shape(data)[0] != 1:
            if np.shape(data)[1] > np.shape(data)[0]:
                data_for_predictions = data_for_predictions.T
            data_for_predictions = data_for_predictions[:, -config.datalength:]

            if config.other_columns:
                # Make predicted column on index 0
                data_for_predictions[[0, predicted_column], :] = data_for_predictions[[predicted_column, 0], :]
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

    min_data_length = 3 * predicts + config.repeatit * predicts

    if data_length < min_data_length:
        config.repeatit = 1
        min_data_length = 3 * predicts

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

    params_everywhere = {"predicts": predicts}

    if len(data_shape) > 1:
        params_everywhere["predicted_column_index"] = predicted_column_index

    for i, j in config.used_models.items():

        # If no parameters or parameters details, add it so no index errors later
        if i not in config.models_parameters:
            config.models_parameters[i] = {}
        if i not in config.models_parameters_limits:
            config.models_parameters_limits[i] = {}

    if config.optimizeit:
        models_optimized_parameters = {}
        models_optimizations_time = config.used_models.copy()

        for i, j in config.used_models.items():

            try:
                start_optimization = time.time()
                model_kwargs = {**config.models_parameters[i], **params_everywhere}
                best_kwargs = predictit.best_params.optimize(j, model_kwargs, config.models_parameters_limits[i], data_for_predictions, fragments=config.fragments, iterations=config.iterations, time_limit=config.optimizeit_limit, criterion=config.criterion, name=i, details=config.optimizeit_details)
                models_optimized_parameters[i] = best_kwargs

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
    results_matrix = np.zeros((config.repeatit, models_number, data_number, predicts))
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

            train, test = dp.split(data_for_predictions[:, data_start: data_end], predicts=predicts, predicted_column_index=predicted_column_index)

            for m, (n, o) in enumerate(config.used_models.items()):

                try:
                    start = time.time()

                    if n in config.models_parameters:
                        results_matrix[r, m, p] = o(train, **params_everywhere, **config.models_parameters[n])

                    else:
                        results_matrix[r, m, p] = o(train, **params_everywhere)

                except Exception:

                    if config.debug:
                        warnings.warn(f"\n \t Error in compute {n} model on data length {p} : \n\n {traceback.format_exc()} \n")

                finally:
                    end = time.time()

                evaluated_matrix[r, m, p] = predictit.evaluate_predictions.compare_predicted_to_test(results_matrix[r, m, p], test, criterion=config.criterion)

                models_time[n] = (end - start)

        if data_end:
            data_end -= predicts
        else:
            data_end = -predicts

    # Criterion is the best of average from repetitions
    repeated_average = np.nanmean(evaluated_matrix, axis=0)

    model_results = np.nanmin(repeated_average, axis=1)

    # Index of the best model
    best_model_index = np.unravel_index(np.nanargmin(model_results), shape=model_results.shape)[0]

    best_model_matrix = repeated_average[best_model_index]
    best_data_index = np.unravel_index(np.nanargmin(best_model_matrix), shape=best_model_matrix.shape)[0]
    best_data_length = data_lengths[best_data_index]

    # Evaluation of the best model
    best_model_name, best_model = list(config.used_models.items())[best_model_index]

    best_model_param = config.models_parameters[best_model_name]
    best_model_param_limits = config.models_parameters_limits[best_model_name]
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

            # Evaluate models for comparison

            try:
                next_models_predicts[i] = next_models[i](data_for_predictions[:, -next_models_data_length[i]: ], **params_everywhere, **next_models_params[i])

                if config.standardize:
                    next_models_predicts[i] = final_scaler.inverse_transform(next_models_predicts[i])

                if config.data_transform == 'difference':
                    next_models_predicts[i] = dp.inverse_difference(next_models_predicts[i], last_undiff_value)

            except Exception:
                if config.debug:
                    warnings.warn(f"\n \t Error in compute {n} model on data {p}: \n\n {traceback.format_exc()} \n")
    # Final optimalisation of the best model
    if config.optimizeit_final and best_model_name in config.models_parameters_limits:
        best_kwargs = predictit.best_params.optimize(best_model, best_model_param, best_model_param_limits, data_for_predictions[-best_data_length:], fragments=config.fragments_final, iterations=config.iterations_final, time_limit=config.optimizeit_final_limit, name=best_model_name)

        for k, l in best_kwargs.items():
            config.models_parameters[best_model_name][k] = l

    ####### Evaluate best Model ####### ANCHOR Evaluate best

    best_model_predicts = np.zeros(predicts)
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
    ############# Results ############# ANCHOR Results
    ##########################################

    # Definition of the table for results
    models_table = PrettyTable()
    models_table.field_names = ["Model", "Average {} error".format(config.criterion), "Time"]

    # Fill the table
    for i, j in enumerate(models_names):
        models_table.add_row([models_names[i], model_results[i], models_time[models_names[i]]])

        if config.debug:

            print('\n', models_names[i])

            for k in range(data_number):

                print(f"\t With data length: {data_lengths[k]}  {config.criterion} = {repeated_average[i, k]}")

            if config.optimizeit:
                print(f"\t Time to optimize {models_optimizations_time[j]} \n")
                print("Best models parameters", models_optimized_parameters[j])

    print(f'\n {models_table} \n')

    print(f"\n Best model is {best_model_name} \n\t with result {config.criterion} {best_mape} \n\t with data length {best_data_length} \n\t with paramters {best_model_param} \n")

    ########################
    ######### Plot #########
    ########################

    if config.plot:

        try:
            lower_bound, upper_bound = predictit.confidence_interval.bounds(column_for_prediction, predicts=predicts, confidence=config.confidence)
        except Exception:
            lower_bound = upper_bound = best_model_predicts
            if config.debug:
                warnings.warn(f"\n \t Error in compute confidence interval: \n\n {traceback.format_exc()} \n")

        complete_dataframe = column_for_prediction_dataframe.iloc[-7 * predicts:, :]

        global last_date
        last_date = column_for_prediction_dataframe.index[-1]

        if isinstance(last_date, pd._libs.tslibs.timestamps.Timestamp):
            date_index = pd.date_range(start=last_date, periods=predicts + 1, freq=freq)[1:]
            date_index = pd.to_datetime(date_index)

        else:
            date_index = list(range(last_date + 1, last_date + predicts + 1))

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
        complete_dataframe.iloc[-predicts - 1] = last_value

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
                        'y':0.9,
                        'x':0.5,
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
            if not config.save_plot_adress:
                config.save_plot_adress = os.path.normpath(os.path.expanduser("~/Desktop") + '/plot.html')
            pl.offline.plot(fig, filename=config.save_plot_adress)
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


def predict_multiple_columns(data=[], predicted_columns=[], freqs=[], database_deploy=0):
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

    if not predicted_columns:
        predicted_columns = []
        predicted_columns.extend(config.predicted_columns)
    if not freqs:
        freqs = []
        freqs.extend(config.freqs)

    predictions_full = [np.nan] * len(freqs)

    for j in range(len(freqs)):

        predictions = [np.nan] * len(predicted_columns)

        for i in range(len(predicted_columns)):

            try:
                predictions[i] = predict(data=data, predicted_column=predicted_columns[i], freq=freqs[j])

            except Exception:
                warnings.warn(f"\n Error in making predictions on column {i} and freq {j} - {traceback.format_exc()} \n")

        predictions_full[j] = predictions

        if config.database_deploy:
            try:
                # TODO last_day resolve
                predictit.database.database_deploy(last_date, predictions[0], predictions[1], freq=config.freqs[j])
            except Exception:
                warnings.warn(f"\n Error in database deploying on freq {j} - {traceback.format_exc()} \n")

    return predictions_full


def compare_models(data_all, data_length=1000):
    """Function that helps to choose apropriate models. It evaluate it on test data and then return results.
    After you know what models are the best, you can use only them in functions predict() or predict_multiple.
    You can define your own test data and find best modules for your process. You can pickle data if you are use it 
    more often to faster loading.

    Args:
        data_all (dict): Dictionary of data names and data values (np.array). You can use data from test_data module, data_test script (e.g. gen_sin()).
        data_length (int, optional): If data are pickled, length that will be used. Defaults to 1000.

    """

    if config.piclkeit:
        from predictit.test_data.pickle_test_data import pickle_data_all
        import pickle
        pickle_data_all(data_all, datalength=data_length)

    if config.from_pickled:

        script_dir = Path(__file__).resolve().parent
        data_folder = script_dir / "test_data" / "pickled"

        for i, j in data_all.items():
            file_name = i + '.pickle'
            file_adress = data_folder / file_name
            try:
                with open(file_adress, "rb") as input_file:
                    data = pickle.load(input_file)
                data_all[i] = data
            except Exception:
                warnings.warn(f"\n Error: {traceback.format_exc()} \n Warning - test data not loaded - First in config.py pickleit = 1, that save the data on disk, then load from pickled. \n")

    results = {}

    for i, j in data_all.items():
        config.plot_name = i
        try:
            result = predictit.main.predict(data=j, return_model_criterion=1)
            results[i] = (result - np.nanmin(result)) / (np.nanmax(result) - np.nanmin(result))

        except Exception:
            results[i] = np.nan

    results_array = np.stack((results.values()), axis=0)

    all_data_average = np.nanmean(results_array, axis=0)

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
