#!/usr/bin/python
#%%
"""This is copy of main.py for developing reasons."""
import sys
from pathlib import Path
import numpy as np
from prettytable import PrettyTable
import time
import pandas as pd
import argparse
import warnings
import inspect
import os
# import multiprocessing_on_dill as multiprocessing
import multiprocessing

# multiprocessing.set_start_method("fork")

# import ctypes
# from multiprocessing import sharedctypes

this_path = Path(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)).parents[1]
this_path_string = this_path.as_posix()

sys.path.insert(0, this_path_string)

import predictit

from predictit.config import config, presets
from predictit.misc import traceback_warning, user_warning, colorize

# import moo

try:
    import gui_start  # Not included in predictit if used as python library
except Exception:
    pass


def update_gui(content, id):
    try:
        gui_start.edit_gui_py(content, id)
    except Exception:
        pass


# This is core function... It should be sequentionally in the middle of main script in predict function, but it has to be 1st level function to be able to use in multiprocessing.
# To understand content, see code below function
def train_and_predict(
        iterated_model_train, iterated_model_predict, iterated_model_name, iterated_model_index, data_length_index,
        data_length_iteration, data_lengths, model_train_input, model_predict_input, model_test_inputs,
        data_abs_max, data_mean, data_std, models_test_outputs, last_undiff_value=None, final_scaler=None, pipe=None):

    model_results = {}

    if config['optimizeit'] and data_length_index == 0:
        if iterated_model_name in config['models_parameters_limits']:

            try:
                start_optimization = time.time()

                model_results['best_kwargs'] = predictit.best_params.optimize(
                    iterated_model_train, iterated_model_predict, config['models_parameters'][iterated_model_name], config['models_parameters_limits'][iterated_model_name],
                    model_train_input=model_train_input, model_test_inputs=model_test_inputs, models_test_outputs=models_test_outputs,
                    fragments=config['fragments'], iterations=config['iterations'], time_limit=config['optimizeit_limit'],
                    error_criterion=config['error_criterion'], name=iterated_model_name, details=config['optimizeit_details'])

                for k, l in model_results['best_kwargs'].items():

                    config['models_parameters'][iterated_model_name][k] = l

            except Exception:
                traceback_warning("Optimization didn't finished")

            finally:
                stop_optimization = time.time()

                model_results['optimization_time'] = stop_optimization - start_optimization

    try:
        start = time.time()

        model_results['name'] = iterated_model_name
        model_results['index'] = (iterated_model_index, data_length_index)
        model_results['data_length_index'] = data_length_index
        model_results['data_length_iteration'] = data_length_iteration

        # If no parameters or parameters details, add it so no index errors later
        if iterated_model_name not in config['models_parameters']:
            config['models_parameters'][iterated_model_name] = {}

        # Train all models
        model_results['trained_model'] = iterated_model_train(model_train_input, **config['models_parameters'][iterated_model_name])

        # Create predictions - out of sample
        one_reality_result = iterated_model_predict(model_predict_input, model_results['trained_model'], config['predicts'])

        if np.isnan(np.sum(one_reality_result)) or one_reality_result is None:
            return

        # Remove wrong values out of scope to not be plotted
        one_reality_result[abs(one_reality_result) > 3 * data_abs_max] = np.nan

        # Do inverse data preprocessing
        if config['power_transformed'] == 1:
            one_reality_result = predictit.data_preprocessing.fitted_power_transform(one_reality_result, data_std, data_mean)

        if config['standardize']:
            one_reality_result = final_scaler.inverse_transform(one_reality_result.reshape(-1, 1)).ravel()

        if config['data_transform'] == 'difference':
            one_reality_result = predictit.data_preprocessing.inverse_difference(one_reality_result, last_undiff_value)

        model_results['results'] = one_reality_result

        process_repeated_matrix = np.zeros((config['repeatit'], config['predicts']))
        process_evaluated_matrix = np.zeros(config['repeatit'])

        # Predict many values in test inputs to evaluate which models are best - do not inverse data preprocessing, because test data are processed
        for repeat_iteration in range(config['repeatit']):

            # Create in-sample predictions to evaluate if model is good or not

            process_repeated_matrix[repeat_iteration] = iterated_model_predict(model_test_inputs[repeat_iteration], model_results['trained_model'], predicts=config['predicts'])

            if config['power_transformed'] == 2:
                process_repeated_matrix[repeat_iteration] = predictit.data_preprocessing.fitted_power_transform(process_repeated_matrix[repeat_iteration], data_std, data_mean)

            process_evaluated_matrix[repeat_iteration] = predictit.evaluate_predictions.compare_predicted_to_test(
                process_repeated_matrix[repeat_iteration], models_test_outputs[repeat_iteration], error_criterion=config['error_criterion'])

        model_results['repeated_matrix'] = process_repeated_matrix
        model_results['evaluated_matrix'] = process_evaluated_matrix
        model_results['model_error'] = process_evaluated_matrix.mean()

    except Exception:

        traceback_warning(f"Error in {iterated_model_name} model on data length {data_length_iteration}")

    finally:
        model_results['model_time'] = time.time() - start

        if config['multiprocessing'] == 'process':
            pipe.send(model_results)
            pipe.close()
        else:
            return model_results


if __name__ == "__main__":





    config.update({

        'multiprocessing': 'pool',
        'data_source': 'csv',
        'csv_test_data_relative_path': '5000 Sales Records.csv',
        'datetime_index': 5,
        'datalength': 100,
        'freq': 'D',
        'predicted_column': '',
        'print_number_of_models': 10,
        'last_row': 0,
        'correlation_threshold': 0.2,
        'optimizeit': 0,
        'standardize': 0,
        'lengths': 0,
        'data': np.array(range(1000)),
        'repeatit': 3,

    'used_models': {

        # **{model_name: predictit.models.statsmodels_autoregressive for model_name in [
        #     'AR (Autoregression)', 'ARMA', 'ARIMA (Autoregression integrated moving average)']},  # 'SARIMAX (Seasonal ARIMA)'

        # **{model_name: predictit.models.autoreg_LNU for model_name in [
        #     'Autoregressive Linear neural unit', 'Autoregressive Linear neural unit normalized']},  # , 'Linear neural unit with weigths predict'

        # 'Conjugate gradient': predictit.models.conjugate_gradient,

        # 'tensorflow_lstm': predictit.models.tensorflow,
        # 'tensorflow_mlp': predictit.models.tensorflow,

        # **{model_name: predictit.models.sklearn_regression for model_name in [  # 'Extra trees regression', 'Random forest regression', 'Bagging regression', 'Hubber regression', 'Stochastic gradient regression', 'Extreme learning machine', 'Gen Extreme learning machine'
        #     'Sklearn regression', 'Bayes ridge regression', 'Decision tree regression', 'KNeighbors regression',
        #     'Passive aggressive regression', 'Gradient boosting']},

        'Compare with average': predictit.models.compare_with_average
    }
    })






    results = []

    if config['multiprocessing'] == 'process':
        processes = []
        pipes = []
        # queues_dict = config['used_models'].copy()
        # for i in queues_dict:
        #     queues_dict[i] = multiprocessing.Pipe()  # TODO duplex=False

    if config['multiprocessing'] == 'pool':
        pool = multiprocessing.Pool()

        # It is not possible easy share data in multiprocessing, so results are resulted via callback function
        def return_result(result):
            results.append(result)











    if config['debug'] == 1:
        warnings.filterwarnings('once')
    elif config['debug'] == 2:
        warnings.filterwarnings('error')
    else:
        warnings.filterwarnings('ignore')

    for i in config['ignored_warnings']:
        warnings.filterwarnings('ignore', message=fr"[\s\S]*{i}*")

    _GUI = predictit.misc._GUI
    # Add everything printed + warnings to variable to be able to print in GUI
    if _GUI or config['debug'] == -1:
        import io

        stdout = sys.stdout
        sys.stdout = io.StringIO()

    if config["use_config_preset"] and config["use_config_preset"] != 'none':
        config.update(presets[config["use_config_preset"]])

    # Parse all functions parameters and it's values
    args, _, _, values = inspect.getargvalues(inspect.currentframe())

    # Edit config.py default values with arguments values if exist
    config.update({key: value for key, value in values.items() if key in args & values.keys() and value is not None})

    # Do not repeat actually mean evaluate once
    if not config['repeatit']:
        config['repeatit'] = 1

    # Some config values are derived from other values. If it has been changed, it has to be updated.
    predictit.config.update_references_1()
    predictit.config.update_references_2()

    # Find difference between original config and set values and if there are any differences, raise error
    config_errors = set(config.keys()) - predictit.config.config_check_set
    if config_errors:
        raise KeyError(colorize(f"Some config values: {config_errors} was named incorrectly. Check config.py for more informations"))

    # Definition of the table for spent time on code parts
    time_parts_table = PrettyTable()
    time_parts_table.field_names = ["Part", "Time"]

    def update_time_table(time_last):
        time_parts_table.add_row([progress_phase, round((time.time() - time_last), 3)])
        return time.time()
    time_point = time_begin = time.time()

    #######################################
    ############## LOAD DATA ####### ANCHOR Data
    #######################################

    progress_phase = "Data loading and preprocessing"
    update_gui(progress_phase, 'progress_phase')

    if config['data'] is None:


        ############# Load CSV data #############
        if config['data_source'] == 'csv':
            if config['csv_test_data_relative_path']:
                try:
                    config['csv_full_path'] = (this_path / 'predictit' / 'test_data' / config['csv_test_data_relative_path']).as_posix()
                except Exception:
                    raise FileNotFoundError(colorize(("\n ERROR - Test data load failed - Setup CSV adress and column name in config. "
                                                      "Use relative and save to test_data folder or  \n\n")))
            try:
                config['data'] = pd.read_csv(config['csv_full_path'], header=0).iloc[-config['max_imported_length']:, :]
            except Exception:
                raise FileNotFoundError(colorize(f"\n ERROR - Test data load failed - Setup CSV adress and column name in config \n\n"))

        ############# Load SQL data #############
        elif config['data_source'] == 'sql':
            try:
                config['data'] = predictit.database.database_load(server=config['server'], database=config['database'], freq=config['freq'],
                                                                  data_limit=config['max_imported_length'], last=config['last_row'])
            except Exception:
                raise RuntimeError(colorize(f"ERROR - Data load from SQL server failed - Setup server, database and predicted column name in config"))

        elif config['data_source'] == 'test':
            config['data'] = predictit.test_data.generate_test_data.gen_random()
            user_warning(("Test data was used. Setup config.py 'data_source'. Check official readme or do >>> predictit.config.print_config() "
                          "to see all possible options with comments. Data can be insert as function parameters, with editing config.py or with CLI."))

    ##############################################
    ############ DATA PREPROCESSING ###### ANCHOR Preprocessing
    #############################################

    if not config['predicted_column']:
        config['predicted_column'] = 0

    if config['data'] is None:
        raise TypeError(colorize("Data not loaded. Check config.py and use 'data_source' - csv and path or assign data to 'data'"))

    data_for_predictions, data_for_predictions_df, predicted_column_name = predictit.data_preprocessing.data_consolidation(config['data'])

    if config['mode'] == 'validate':
        data_for_predictions, test = predictit.data_preprocessing.split(data_for_predictions, predicts=config['predicts'])
        data_for_predictions_df, _ = predictit.data_preprocessing.split(data_for_predictions_df, predicts=config['predicts'])

    # In data consolidation predicted column was replaced on index 0 as first column
    predicted_column_index = 0

    multicolumn = 0 if data_for_predictions.shape[1] == 1 else 1

    column_for_predictions_series = data_for_predictions_df.iloc[:, 0]

    ########################################
    ############# Data analyze ###### ANCHOR Analyze
    ########################################data_abs_max

    if config['analyzeit'] == 1 or config['analyzeit'] == 3:
        print("Analyze of unprocessed data")
        try:
            predictit.analyze.analyze_data(data_for_predictions[:, 0], column_for_predictions_series, window=30)
            predictit.analyze.analyze_correlation(data_for_predictions_df)
            predictit.analyze.decompose(data_for_predictions[:, 0], **config['analyze_seasonal_decompose'])
        except Exception:
            traceback_warning("Analyze failed")

    if config['remove_outliers']:
        data_for_predictions = predictit.data_preprocessing.remove_outliers(data_for_predictions, predicted_column_index=predicted_column_index,
                                                                            threshold=config['remove_outliers'])

    if config['smooth']:
        data_for_predictions = predictit.data_preprocessing.smooth(data_for_predictions, config['smooth'][0], config['smooth'][1])

    if config['correlation_threshold'] and multicolumn:
        data_for_predictions = predictit.data_preprocessing.keep_corelated_data(data_for_predictions, threshold=config['correlation_threshold'])

    if config['data_transform'] == 'difference':
        last_undiff_value = data_for_predictions[-1, 0]
        for i in range(data_for_predictions.shape[1]):
            data_for_predictions[1:, i] = predictit.data_preprocessing.do_difference(data_for_predictions[:, i])
    else:
        last_undiff_value = None

    if config['standardize']:
        data_for_predictions, final_scaler = predictit.data_preprocessing.standardize(data_for_predictions, used_scaler=config['standardize'])
    else:
        final_scaler = None

    column_for_predictions = data_for_predictions[:, predicted_column_index]

    ############# Processed data analyze ############

    data_shape = np.shape(data_for_predictions)
    data_length = len(column_for_predictions)

    data_std = np.std(column_for_predictions[-30:])
    data_mean = np.mean(column_for_predictions[-30:])
    data_abs_max = max(abs(column_for_predictions.min()), abs(column_for_predictions.max()))

    multicolumn = 0 if data_shape[1] == 1 else 1

    if config['analyzeit'] == 2 or config['analyzeit'] == 3:
        print("\n\n Analyze of preprocessed data \n")
        try:
            predictit.analyze.analyze_data(column_for_predictions, pd.Series(column_for_predictions), window=30)
        except Exception:
            traceback_warning("Analyze failed")

    min_data_length = 3 * config['predicts'] + config['default_n_steps_in']

    if data_length < min_data_length:
        config['repeatit'] = 1
        min_data_length = 3 * config['predicts'] + config['repeatit'] * config['predicts'] + config['default_n_steps_in']

    assert (min_data_length < data_length), 'To few data - set up less repeat value in settings or add more data'

    if config['lengths']:
        data_lengths = [data_length, int(data_length / 2), int(data_length / 4), min_data_length + 50, min_data_length]
        data_lengths = [k for k in data_lengths if k >= min_data_length]
    else:
        data_lengths = [data_length]

    data_number = len(data_lengths)

    models_names = list(config['used_models'].keys())
    models_number = len(models_names)

    # Empty boxes for results definition
    # The final result is - [repeated, model, data, results]
    test_results_matrix = np.zeros((config['repeatit'], models_number, data_number, config['predicts']))
    evaluated_matrix = np.zeros((config['repeatit'], models_number, data_number))
    reality_results_matrix = np.zeros((models_number, data_number, config['predicts']))
    test_results_matrix.fill(np.nan)
    evaluated_matrix.fill(np.nan)
    reality_results_matrix.fill(np.nan)

    time_point = update_time_table(time_point)
    progress_phase = "Predict"
    update_gui(progress_phase, 'progress_phase')

    models_test_outputs = np.zeros((config['repeatit'], config['predicts']))

    for i in range(config['repeatit']):

        models_test_outputs[i] = column_for_predictions[-config['predicts'] - i: - i] if i > 0 else column_for_predictions[-config['predicts'] - i:]

    models_test_outputs = models_test_outputs[::-1]

    data_end = -config['predicts'] - config['repeatit'] - config['validation_gap'] if config['mode'] == 'validate' else None

    used_input_types = []
    for i in models_names:
        used_input_types.append(config['models_input'][i])
    used_input_types = set(used_input_types)

    #######################################
    ############# Main loop ######## ANCHOR Main loop
    #######################################

    for data_length_index, data_length_iteration in enumerate(data_lengths):

        for input_type in used_input_types:
            try:
                used_sequention = predictit.define_inputs.create_inputs(input_type, data_for_predictions, predicted_column_index=predicted_column_index)
            except Exception:
                traceback_warning(f"Error in creating sequentions on input type: {input_type} model on data length: {data_length_iteration}")
                continue

            for iterated_model_index, (iterated_model_name, iterated_model) in enumerate(config['used_models'].items()):
                if config['models_input'][iterated_model_name] == input_type:

                    if config['models_input'][iterated_model_name] in ['one_step', 'one_step_constant']:
                        if multicolumn and config['predicts'] > 1:

                            user_warning(f"""Warning in model {iterated_model_name} \n\nOne-step prediction on multivariate data (more columns).
                                             Use batch (y lengt equals to predict) or do use some one column data input in config models_input or predict just one value.""")
                            continue

                    if isinstance(used_sequention, tuple):
                        model_train_input = (used_sequention[0][data_length - data_length_iteration: data_end, :], used_sequention[1][data_length - data_length_iteration: data_end, :])
                        model_predict_input = used_sequention[2]
                        model_test_inputs = used_sequention[3]

                    elif used_sequention.ndim == 1:
                        model_train_input = used_sequention[data_length - data_length_iteration: data_end]
                        model_predict_input = used_sequention[data_length - data_length_iteration:]
                        model_test_inputs = []
                        for i in range(config['repeatit']):
                            model_test_inputs.append(used_sequention[data_length - data_length_iteration: - config['predicts'] - config['repeatit'] + i + 1])

                    else:
                        model_train_input = used_sequention[:, data_length - data_length_iteration: data_end]
                        model_predict_input = used_sequention[:, data_length - data_length_iteration:]
                        model_test_inputs = []
                        for i in range(config['repeatit']):
                            model_test_inputs.append(used_sequention[:, data_length - data_length_iteration: - config['predicts'] - config['repeatit'] + i + 1])

                    predict_parameters = {
                        'iterated_model_train': iterated_model.train, 'iterated_model_predict': iterated_model.predict, 'iterated_model_name': iterated_model_name,
                        'iterated_model_index': iterated_model_index, 'data_length_index': data_length_index, 'data_length_iteration': data_length_iteration,
                        'data_lengths': data_lengths, 'model_train_input': model_train_input, 'model_predict_input': model_predict_input,
                        'model_test_inputs': model_test_inputs, 'data_abs_max': data_abs_max, 'data_mean': data_mean, 'data_std': data_std,
                        'last_undiff_value': last_undiff_value, 'models_test_outputs': models_test_outputs, 'final_scaler': final_scaler
                    }

                    if config['multiprocessing'] == 'process':
                        pipes.append(multiprocessing.Pipe())
                        p = multiprocessing.Process(target=train_and_predict, kwargs={**predict_parameters, **{'pipe': pipes[-1][1]}})

                        processes.append(p)
                        p.start()

                    elif config['multiprocessing'] == 'pool':

                        pool.apply_async(train_and_predict, (), predict_parameters, callback=return_result)

                    else:
                        results.append(train_and_predict(**predict_parameters))

    if config['multiprocessing']:
        if config['multiprocessing'] == 'process':
            for i in pipes:
                try:
                    results.append(i[0].recv())
                except Exception:
                    pass

        if config['multiprocessing'] == 'pool':
            pool.close()
            pool.join()

    for i in results:
        try:
            if 'results' and 'evaluated_matrix' in i:
                reality_results_matrix[i['index'][0], i['index'][1], :] = i['results']
                evaluated_matrix[:, i['index'][0], i['index'][1]] = i['evaluated_matrix']
        except Exception:
            pass

    # TODO Do repeate average and more from evaluate in multiprocessing loop and then use sorting as
    # a = {k: v for k, v in sorted(x.items(), key=lambda item: item[1]['a'])} then use just slicing for plot and print results

    #############################################
    ############# Evaluate models ######## ANCHOR Evaluate
    #############################################

    # Criterion is the best of average from repetitions
    time_point = update_time_table(time_point)
    progress_phase = "Evaluation"
    update_gui(progress_phase, 'progress_phase')

    repeated_average = np.mean(evaluated_matrix, axis=0)

    model_results = []

    for i in repeated_average:
        model_results.append(np.nan if np.isnan(i).all() else np.nanmin(i))

    sorted_results = np.argsort(model_results)

    predicted_models_for_table = {}
    predicted_models_for_plot = {}

    for i, j in enumerate(sorted_results):
        this_model = list(config['used_models'].keys())[j]

        if i == 0:
            best_model_name = this_model

        if not config['print_number_of_models'] or i < config['print_number_of_models']:
            predicted_models_for_table[this_model] = {
                'order': i + 1, 'error_criterion': model_results[j], 'predictions': reality_results_matrix[j, np.argmin(repeated_average[j])],
                'data_length': np.argmin(repeated_average[j])}

        if not config['plot_number_of_models'] is None or i < config['plot_number_of_models']:
            predicted_models_for_plot[this_model] = {
                'order': i + 1, 'error_criterion': model_results[j], 'predictions': reality_results_matrix[j, np.argmin(repeated_average[j])],
                'data_length': np.argmin(repeated_average[j])}

    best_model_predicts = predicted_models_for_table[best_model_name]['predictions'] if best_model_name in predicted_models_for_table else "Best model had an error"

    complete_dataframe = column_for_predictions_series[-config['plot_history_length']:].to_frame()

    if config['confidence_interval']:
        try:
            lower_bound, upper_bound = predictit.misc.confidence_interval(column_for_predictions_series, predicts=config['predicts'], confidence=config['confidence_interval'])
            complete_dataframe['Lower bound'] = complete_dataframe['Upper bound'] = None
            bounds = True
        except Exception:
            bounds = False
            traceback_warning("Error in compute confidence interval")

    else:
        bounds = False

    last_date = column_for_predictions_series.index[-1]

    if isinstance(last_date, (pd.core.indexes.datetimes.DatetimeIndex, pd._libs.tslibs.timestamps.Timestamp)):
        date_index = pd.date_range(start=last_date, periods=config['predicts'] + 1, freq=column_for_predictions_series.index.freq)[1:]
        date_index = pd.to_datetime(date_index)

    else:
        date_index = list(range(last_date + 1, last_date + config['predicts'] + 1))

    results_dataframe = pd.DataFrame(data={'Lower bound': lower_bound, 'Upper bound': upper_bound}, index=date_index) if bounds else pd.DataFrame(index=date_index)

    for i, j in predicted_models_for_plot.items():
        if 'predictions' in j:
            results_dataframe[f"{j['order']} - {i}"] = j['predictions']
            complete_dataframe[f"{j['order']} - {i}"] = None
            if j['order'] == 1:
                best_model_name_plot = f"{j['order']} - {i}"

    if config['mode'] == 'validate':
        best_model_name_plot = 'Test'
        results_dataframe['Test'] = test

    last_value = float(column_for_predictions_series.iloc[-1])

    complete_dataframe = pd.concat([complete_dataframe, results_dataframe], sort=False)
    complete_dataframe.iloc[-config['predicts'] - 1] = last_value

    #######################################
    ############# Plot ############# ANCHOR Plot
    #######################################

    time_point = update_time_table(time_point)
    progress_phase = "plot"
    update_gui(progress_phase, 'progress_phase')

    if config['plot']:

        plot_return = 'div' if _GUI else ''
        div = predictit.plot.plotit(complete_dataframe, plot_type=config['plot_type'], show=config['show_plot'], save=config['save_plot'],
                                    save_path=config['save_plot_path'], plot_return=plot_return, best_model_name=best_model_name_plot,
                                    predicted_column_name=predicted_column_name)

    time_point = update_time_table(time_point)
    progress_phase = "Completed"
    update_gui(progress_phase, 'progress_phase')

    ####################################
    ############# Results ####### ANCHOR Results
    ####################################

    if config['print']:
        if config['print_result']:
            print((f"\n Best model is {best_model_name} \n\t with results {best_model_predicts} \n\t with model error {config['error_criterion']} = "
                   f"{predicted_models_for_table[best_model_name]['error_criterion']}"))

            # TODO
            # print((f"\n\t with data length {data_lengths[predicted_models_for_table[best_model_name]['data_length']]} \n\t with paramters "
            #        f"{config['models_parameters'][best_model_name]} \n"))


        # Table of results
        models_table = PrettyTable()
        models_table.field_names = ['Model', f"Average {config['error_criterion']} error"]
        # Fill the table
        for i, j in predicted_models_for_table.items():
            models_table.add_row([i, round(j['error_criterion'], 3)])

        if config['print_table']:
            print(f'\n {models_table} \n')

        time_parts_table.add_row(['Complete time', round((time.time() - time_begin), 3)])

        if config['print_time_table']:
            print(f'\n {time_parts_table} \n')

        ### Print detailed resuts ###
        if config['print_detailed_result']:

            detailed_table = PrettyTable()
            detailed_table.field_names = ['Name', f"Average {config['error_criterion']} error", 'Time', 'Iteration', 'Iteration value']
            for i in results:
                try:
                    detailed_table.add_row([i['name'], round(i['model_error'], 3), round(i['model_time']), round(i['data_length_iteration'], 3), i['data_length_index']])
                except Exception:
                    pass

                # if config['optimizeit']:
                #     print(f"\t Time to optimize {i['optimization_time']} \n")
                #     print("Best models parameters", config['models_parameters'][j])

            print(detailed_table)

    time_point = update_time_table(time_point)
    progress_phase = 'finished'
    update_gui(progress_phase, 'progress_phase')

    # Return stdout and stop collect warnings and printed output
    if _GUI:
        output = sys.stdout.getvalue()
        sys.stdout = stdout

        print(output)
