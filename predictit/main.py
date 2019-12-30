#!/usr/bin/python

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
sys.path.insert(0, this_path_string)

import predictit
from predictit import config
import predictit.data_prep as dp

# All the config is in config.py - rest only for people that know what are they doing


def predict(data=None, predicts=None, predicted_column=None, freq=None):

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
        if isinstance(config.freqs, list):
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

    if not data:

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

        data_for_predictions_df = dp.remove_nan_columns(data_for_predictions_df)

        if config.remove_outliers:
            data_for_predictions_df = dp.remove_outliers(data_for_predictions_df, predicted_column_index=predicted_column_index, threshold=config.remove_outliers)

        data_for_predictions_df = dp.keep_corelated_data(data_for_predictions_df)

        data_for_predictions = data_for_predictions_df.values.T

    if isinstance(data, np.ndarray):
        data_for_predictions = data

        if len(np.shape(data)) > 1:
            if np.shape(data)[1] > np.shape(data)[0]:
                data_for_predictions = data_for_predictions.T
            data_for_predictions = data_for_predictions[:, -config.datalength:]

            # Make predicted column on index 0
            data_for_predictions[[0, predicted_column], :] = data_for_predictions[[predicted_column, 0], :]
            data_for_predictions = dp.remove_nan_columns(data_for_predictions)

            predicted_column_index = 0
            if config.remove_outliers:
                data_for_predictions = dp.remove_outliers(data_for_predictions, predicted_column_index=predicted_column_index, threshold=config.remove_outliers)

            data_for_predictions = dp.keep_corelated_data(data_for_predictions)

        else:
            data_for_predictions = data_for_predictions[-config.datalength:]

            predicted_column_index = 0
            if config.remove_outliers:
                data_for_predictions = dp.remove_outliers(data_for_predictions, predicted_column_index=predicted_column_index, threshold=config.remove_outliers)

        predicted_column_name = 'Predicted column'

        data_for_predictions_df = pd.DataFrame(data_for_predictions, columns=[predicted_column_name])

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

    if data_for_predictions.ndim == 1:

        if config.data_transform == 'difference':
            last_undiff_value = data_for_predictions[-1]
            data_for_predictions = dp.do_difference(data_for_predictions)

    if data_for_predictions.ndim == 2:

        if config.data_transform == 'difference':
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

    data_for_trimming = data_for_predictions

    min_data_length = 3 * predicts + config.repeatit * predicts

    if data_length < min_data_length:
        config.repeatit = 1
        min_data_length = 3 * predicts

    assert (min_data_length < data_length), 'To few data - set up less repeat value in settings or add more data'

    if config.lengths:
        data_lengths = [data_length, int(data_length / 2), int(data_length / 4), min_data_length + 50, min_data_length]
        data_lengths = [k for k in data_lengths if k >= min_data_length]
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
        best_model_parameters = {}
        models_optimizations_time = config.used_models.copy()

        for i, j in config.used_models.items():

            try:
                start_optimization = time.time()
                model_kwargs = {**config.models_parameters[i], **params_everywhere}
                best_kwargs = predictit.best_params.optimize(j, model_kwargs, config.models_parameters_limits[i], data_for_predictions, fragments=config.fragments, iterations=config.iterations, time_limit=config.optimizeit_limit, criterion=config.criterion, name=i, details=config.optimizeit_details)
                best_model_parameters[i] = best_kwargs

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

    # Repeat evaluation on shifted data to eliminate randomness
    for r in range(config.repeatit):

        data_all = {}

        if len(data_shape) == 1:
            for i in range(len(data_lengths)):
                data_all['Trimmed data ' + str(data_lengths[i])] = data_for_trimming[-data_lengths[i]:]

        else:
            for i in range(len(data_lengths)):
                for j in range(data_shape[0]):
                    data_all['Trimmed data ' + str(data_lengths[i])] = data_for_trimming[:, -data_lengths[i]:]
        data_names = list(data_all.keys())

        train_all = []
        test_all = []

        ### TRAIN TEST
        for i in range(data_number):
            train_all.append([0] * len(list(data_all.values())[i]))
            test_all.append([0] * len(list(data_all.values())[i]))
            train_all[i], test_all[i] = dp.split(list(data_all.values())[i], predicts=predicts, predicted_column_index=predicted_column_index)

        ################################################
        ############# Predict and evaluate ############# ANCHOR Predict
        ################################################

        for m, (n, o) in enumerate(config.used_models.items()):
            for p, q in enumerate(train_all):

                try:
                    start = time.time()
                    results_matrix[r, m, p] = o(q, **params_everywhere, **config.models_parameters[n])

                except Exception:
                    try:
                        results_matrix[r, m, p] = o(q, **params_everywhere)

                    except Exception:

                        if config.debug:
                            warnings.warn(f"\n \t Error in compute {n} model on data {p} : \n \t \t {traceback.format_exc()}")

                finally:
                    end = time.time()

                evaluated_matrix[r, m, p] = predictit.evaluate_predictions.compare_predicted_to_test(results_matrix[r, m, p], test_all[p], criterion=config.criterion)

            models_time[n] = (end - start)

        else:
            if len(data_shape) == 1:
                data_for_trimming = data_for_trimming[:-predicts]
            else:
                data_for_trimming = data_for_trimming[:, :-predicts]

    # Criterion is the best of average from repetitions
    repeated_average = np.nanmean(evaluated_matrix, axis=0)

    model_results = np.nanmin(repeated_average, axis=1)

    # Index of the best model
    best_model_index = np.unravel_index(np.nanargmin(model_results), shape=model_results.shape)[0]

    best_model_matrix = repeated_average[best_model_index]
    best_data_index = np.unravel_index(np.nanargmin(best_model_matrix), shape=best_model_matrix.shape)[0]

    # Evaluation of the best model
    best_model_name, best_model = list(config.used_models.items())[best_model_index]
    best_data_name, best_data = list(data_all.items())[best_data_index]

    best_data_len = len(best_data.reshape(-1))
    best_model_param = config.models_parameters[best_model_name]
    best_model_param_limits = config.models_parameters_limits[best_model_name]
    best_mape = np.nanmin(model_results)

    if len(data_shape) == 1:
        best_data_len = len(best_data.reshape(-1))
    else:
        best_data_len = len(best_data[predicted_column_index].reshape(-1))

    if config.compareit:

        models_real_number = len(model_results[~np.isnan(model_results)])
        if config.compareit >= models_real_number:
            next_number = models_real_number - 1
        else:
            next_number = config.compareit

        next_models_names = [np.nan] * next_number
        next_models = [np.nan] * next_number
        next_models_data_names = [np.nan] * next_number
        next_models_data = [np.nan] * next_number
        next_models_data_len = [np.nan] * next_number
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

            # Define other good models
            next_models_names[i], next_models[i] = list(rest_models.items())[next_model_index]
            next_models_data_names[i], next_models_data[i] = list(data_all.items())[next_data_index]

            if next_models_names[i] in config.models_parameters:
                next_models_params[i] = config.models_parameters[next_models_names[i]]
            else:
                next_models_params[i] = {}

            if len(data_shape) == 1 or not config.other_columns:
                next_models_data_len[i] = len(next_models_data[i])
            else:
                next_models_data_len[i] = len(next_models_data[i][0])

            nu_best_model_index = next_model_index
            next_best_model_name = next_models_names[i]

            # Evaluate models for comparison

            if len(data_shape) == 1 or not config.other_columns:
                trimmed_prediction_data = data_for_predictions[-next_models_data_len[i]:]

            else:
                trimmed_prediction_data = data_for_predictions[:, -next_models_data_len[i]:]

            try:
                next_models_predicts[i] = next_models[i](trimmed_prediction_data, **params_everywhere, **next_models_params[i])

                if config.standardize:
                    next_models_predicts[i] = final_scaler.inverse_transform(next_models_predicts[i])

                if config.data_transform == 'difference':
                    next_models_predicts[i] = dp.inverse_difference(next_models_predicts[i], last_undiff_value)

            except Exception:
                if config.debug:
                    warnings.warn(f"\n \t Error in compute {n} model on data {p}: {traceback.format_exc()}")

    # Final optimalisation of the best model
    if config.optimizeit_final and best_model_name in config.models_parameters_limits:

        try:
            best_model_kwargs = predictit.best_params.optimize(best_model, best_model_param, best_model_param_limits, best_data, fragments=config.fragments_final, iterations=config.iterations_final, time_limit=config.optimizeit_final_limit, name=best_model_name)

            for k, l in best_model_kwargs.items():
                config.models_parameters[best_model_name][k] = l
        except Exception:
            pass

    ####### Evaluate best Model ####### ANCHOR Evaluate best
    if len(data_shape) == 1 or not config.other_columns:
        trimmed_prediction_data = data_for_predictions[-best_data_len:]

    else:
        trimmed_prediction_data = data_for_predictions[:, -best_data_len:]

    best_model_predicts = np.zeros(predicts)
    best_model_predicts.fill(np.nan)
    try:

        best_model_predicts = best_model(trimmed_prediction_data, **params_everywhere, **best_model_param)

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
            '''
            for j, k in mae.items():
                print('MAE in {} = {}'.format(j, k))
            for j, k in mape.items():
                print('MAPE in {} = {}'.format(j, k))
            for j, k in mape.items():
                print('RMSE in {} = {}'.format(j, k))
            '''
            print('\n', models_names[i])

            for k in range(data_number):
                print(f"\t With data: {data_names[k]}  {config.criterion} = {repeated_average[i, k]}")

            if config.optimizeit:
                print(f"\t Time to optimize {models_optimizations_time[j]} \n")

            if config.optimizeit:
                print("Best models parameters", best_model_parameters[j])

    print(f'\n {models_table} \n')

    print("\n Best model is {} \n\t with result MAPE {} \n\t with data {} \n\t with paramters {} \n".format(best_model_name, best_mape, best_data_name, best_model_param))

    ########################
    ######### Plot #########
    ########################

    if config.plot:

        try:
            lower_bound, upper_bound = predictit.confidence_interval.bounds(column_for_prediction, predicts=predicts, confidence=config.confidence)
        except Exception:
            lower_bound = upper_bound = best_model_predicts
            if config.debug:
                warnings.warn(f"\n \t Error in compute confidence interval: {traceback.format_exc()}")

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
            yaxis = dict(title='Datum'),
            title = 'Predikce',
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

    if config.return_all:
        all_results = np.array(next_models_predicts.insert(0, best_model_predicts))
        return all_results

    return best_model_predicts


def predict_multiple_columns(data=None, predicted_column_names=[], freqs=[], database_deploy=0):

    if not predicted_column_names:
        predicted_column_names = []
        predicted_column_names.extend(config.predicted_columns)
    if not freqs:
        freqs = []
        freqs.extend(config.freqs)

    predictions_full = [np.nan] * len(freqs)

    for j in range(len(freqs)):

        predictions = [np.nan] * len(predicted_column_names)

        for i in range(len(predicted_column_names)):

            try:
                predictions[i] = predict(data=data, predicted_column=predicted_column_names[i], freq=freqs[j])

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


"""
def compare_models():
    
    # Data all from picle
    if config.piclkeit:
        from predictit.pickle_test_data import pickle_data_all
        import pickle
        pickle_data_all()

    if config.from_pickled:
        data_all = {}

        script_dir = Path(__file__).resolve().parent
        data_folder = script_dir / "test_data"

        for i, j in config.data_all_pickle.items():
            file_name = i + '.pickle'
            file_adress = data_folder / file_name
            try:
                with open(file_adress, "rb") as input_file:
                    data = pickle.load(input_file)
                data_all[i] = data
            except Exception as ex:
                print(f"\n Error: {ex} \n Warning - First in config.py pickleit = 1, that save the data on disk \n")
    else:
        data_all = config.data_all_pickle

    predicted_column_index = 0

    try:
        data_for_predictions = data_all[config.data_name_for_predicts]
        data_shape = np.array(data_for_predictions).shape
    except IndexError:
        print("\n \t Data for prediction are commented or not defined \n")

    column_for_prediction = data_for_predictions.copy()
    column_for_prediction_dataframe = pd.DataFrame()
    column_for_prediction_dataframe['History'] = column_for_prediction
    data_for_predictions_full = column_for_prediction_dataframe
    predicted_column_name = data_for_predictions_full.columns[0]
    data_names = list(data_all.keys())

    #TODO WHY?
    data_number = 1




    for s, t in data_all.items():
        if len(data_shape) == 1:
            data_all[s] = t[:-predicts]
        else:
            data_all[s] = t[:, :-predicts]


    trimmed_prediction_data = data_for_predictions


    model_results = np.nanmean(repeated_average, axis=1)


    results = pd.DataFrame()


    complete_dataframe = pd.concat([complete_dataframe, results], ignore_index=True)
    complete_dataframe.iloc[-predicts - 1] = last_value
"""
    