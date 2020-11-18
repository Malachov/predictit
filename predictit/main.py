#!/usr/bin/python

#%%
""" This is main module for making predictions.

It contain functions - predict() - More return types - Depends on Config
                     - predict_multiple() - Predict multiple columns at once
                     - compare_models() - Test on data that was not in test set and compare models errors

    Examples:

        >>> import predictit
        >>> import numpy as np

        >>> predictions = predictit.main.predict(np.random.randn(1, 100), predicts=3, plotit=1)

Do not edit this file if you are user, it's not necassary! Only call function from here. The only file to edit is configuration.py. If you are developer, edit as you need.

There are working examples in main readme and also in test_it module. Particular modules functionality is vissible in visual.py in tests.
"""
import sys
import multiprocessing
from pathlib import Path, PurePath
import inspect
import os
import warnings
import time
import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate

import mydatapreprocessing as mdp
from mylogging import traceback_warning, user_message, set_warnings, user_warning

# Get module path and insert in sys path for working even if opened from other cwd (current working directory)
this_path = Path(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)).parents[1]
this_path_string = this_path.as_posix()

sys.path.insert(0, this_path_string)

import predictit
from predictit import configuration
from predictit.configuration import Config

Config.this_path = this_path


if __name__ == "__main__":

    # All the Config is in configuration.py" - rest only for people that know what are they doing
    # Add settings from command line if used
    parser = argparse.ArgumentParser(description='Prediction framework setting via command line parser!')
    parser.add_argument("--use_config_preset", type=str, choices=[None, 'fast'], help="Edit some selected Config, other remains the same, check config_presets.py file.")
    parser.add_argument("--used_function", type=str, choices=['predict', 'predict_multiple_columns', 'compare_models', 'validate_predictions'],
                        help=("Which function in main.py use. Predict predict just one defined column, predict_multiple_columns predict more columns at once, "
                              "compare_models compare models on defined test data, validate_predictions test models on data not used for training"))
    parser.add_argument("--data", type=str, help="Path (local or web) to data. Supported formats - [.csv, .txt, .xlsx, .parquet]. Numpy Array and dataframe "
                                                 "allowed, but i don't know how to pass via CLI...")
    parser.add_argument("--predicts", type=int, help="Number of predicted values - 7 by default")
    parser.add_argument("--predicted_column", type=eval, help="""Name of predicted column or it's index - int "1" or string as "'Name'" """)
    parser.add_argument("--predicted_columns", type=list, help="For predict_multiple_columns function only! List of names of predicted column or it's indexes")
    parser.add_argument("--freq", type=str, help="Interval for predictions 'M' - months, 'D' - Days, 'H' - Hours")
    parser.add_argument("--freqs", type=list, help="For predict_multiple_columns function only! List of intervals of predictions 'M' - months, 'D' - Days, 'H' - Hours")
    parser.add_argument("--plotit", type=bool, help="If 1, plot interactive graph")
    parser.add_argument("--datetime_column", type=int, help="Index of dataframe or it's name. Can be empty, then index 0 or new if no date column.")
    parser.add_argument("--return_type", type=str, choices=['best', 'all_dataframe', 'detailed_dictionary', 'models_error_criterion', 'all_dataframe', 'detailed_dictionary'],
                        help=("'best', 'all_dataframe', 'detailed_dictionary', or 'results'."
                              "'best' return array of predictions, 'all_dataframe' return results and models names in columns. 'detailed_dictionary' is used for GUI"
                              "and return results as best result,dataframe for plot, string div with plot and more. If 'results', then all models data are returned including trained models etc..."))
    parser.add_argument("--datalength", type=int, help="The length of the data used for prediction")
    parser.add_argument("--debug", type=bool, help="Debug - print all results and all the errors on the way")
    parser.add_argument("--analyzeit", type=bool, help="Analyze input data - Statistical distribution, autocorrelation, seasonal decomposition etc.")
    parser.add_argument("--optimizeit", type=bool, help="Find optimal parameters of models")
    parser.add_argument("--repeatit", type=int, help="How many times is computation repeated")
    parser.add_argument("--other_columns", type=bool, help="If 0, only predicted column will be used for making predictions.")
    parser.add_argument("--default_other_columns_length", type=bool, help="Length of other columns in input vectors")
    parser.add_argument("--lengths", type=bool, help="Compute on various length of data (1, 1/2, 1/4...). Automatically choose the best length. If 0, use only full length.")
    parser.add_argument("--remove_outliers", type=bool, help=("Remove extraordinary values. Value is threshold for ignored values. Value means how many times standard "
                                                              "deviation from the average threshold is far"))
    parser.add_argument("--standardizeit", type=str, choices=[None, 'standardize', '-11', '01', 'robust'], help="Data standardization, so all columns have similiar scopes")
    parser.add_argument("--error_criterion", type=str, choices=['mape', 'rmse'], help="Error criterion used for model")
    parser.add_argument("--print_number_of_models", type=int, help="How many models will be displayed in final plot. 0 if only the best one.")

    # Non empty command line args
    parser_args_dict = {k: v for k, v in parser.parse_known_args()[0].__dict__.items() if v is not None}

    # Edit Config default values with command line arguments values if exist
    Config.update(parser_args_dict)


def predict(positional_data=None, positional_predicted_column=None, **function_kwargs):

    """Make predictions mostly on time-series data. Data input and other Config options can be set up in configuration.py or overwritenn on the fly. Setup can be also done
    as function input arguments or as command line arguments (it will overwrite Config values).

    For all posible arguments run `predictit.configuration.print_config()`

    There are working examples in main readme and also in test_it module.

    Args example:
        data (np.ndarray, pd.DataFrame): Time series. Can be 2-D - more columns.
            !!! In Numpy array use data series as rows, but in dataframe use cols !!!. If you use CSV, leave it empty. Defaults to [].
        predicted_column (int, str, optional): Index of predicted column or it's name (dataframe).
            If list with more values only the first one will be evaluated (use predict_multiple_columns function if you need that. Defaults to None.
        predicts (int, optional): Number of predicted values. Defaults to None.

        **kwargs (dict): There is much more parameters of predict function. Check configuration.py or run predictit.configuration.print_config() for parameters details.

    Returns:
        Depend on 'return_type' Config value - return best prediction {np.ndarray}, all models results {np.ndarray}, detailed results{dict}
            or interactive plot or print tables of results

    """

    py_version = sys.version_info
    if py_version.major < 3 or py_version.minor < 6:
        raise RuntimeError(user_message("Python version >=3.6 necessary. Python 2 not supported."))

    # Some global Config variables will be updated. But after finishing function, original global Config have to be returned
    config_freezed = Config.freeze()

    _GUI = predictit.misc._GUI

    # Add everything printed + warnings to variable to be able to print in GUI
    if _GUI or Config.debug == -1:
        import io

        stdout = sys.stdout
        sys.stdout = io.StringIO()

    # Dont want to define in gui condition, so if not gui, do nothing
    if _GUI:
        def update_gui(content, id):
            try:
                predictit.gui_start.edit_gui_py(content, id)
            except Exception:
                pass
    else:
        def update_gui(content, id):
            pass

    # Edit configuration.py default values with arguments values if exist
    if positional_data is not None:
        function_kwargs['data'] = positional_data

    if positional_predicted_column is not None:
        function_kwargs['predicted_column'] = positional_predicted_column

    Config.update(function_kwargs)

    # Define whether to print warnings or not or stop on warnings as on error
    set_warnings(Config.debug, Config.ignored_warnings, Config.ignored_warnings_class_type)

    if Config.use_config_preset and Config.use_config_preset != 'none':
        updated_config = Config.presets[Config.use_config_preset]
        Config.update(updated_config)

    # Do not repeat actually mean evaluate once
    if not Config.repeatit:
        Config.repeatit = 1

    # Find difference between original Config and set values and if there are any differences, raise error
    config_errors = configuration.all_variables_set - set(predictit.configuration.orig_config)
    if config_errors:
        raise KeyError(user_message(f"Some Config values: {config_errors} was named incorrectly. Check configuration.py for more informations"))

    # Definition of the table for spent time on code parts
    time_df = []

    def update_time_table(time_last):
        time_df.append([progress_phase, round((time.time() - time_last), 3)])
        return time.time()
    time_point = time_begin = time.time()

    #######################################.iloc[-Config.max_imported_length:, :]
    ############## LOAD DATA ####### ANCHOR Data
    #######################################

    progress_phase = "Data loading and preprocessing"
    update_gui(progress_phase, 'progress_phase')

    if isinstance(Config.data, (str, PurePath)):
        Config.data = mdp.preprocessing.load_data(
            Config.data, header=Config.header, csv_style=Config.csv_style, predicted_table=Config.predicted_table,
            max_imported_length=Config.max_imported_length, request_datatype_suffix=Config.request_datatype_suffix,
            data_orientation=Config.data_orientation)

    #############################################
    ############ DATA consolidation ###### ANCHOR Data consolidation
    #############################################

    if not Config.predicted_column:
        Config.predicted_column = 0

    data_for_predictions_df = mdp.preprocessing.data_consolidation(
        Config.data, predicted_column=Config.predicted_column, other_columns=Config.other_columns, datalength=Config.datalength,
        datetime_column=Config.datetime_column, unique_threshlold=Config.unique_threshlold,
        embedding=Config.embedding, freq=Config.freq, resample_function=Config.resample_function, remove_nans_threshold=Config.remove_nans_threshold,
        remove_nans_or_replace=Config.remove_nans_or_replace, dtype=Config.dtype)

    if Config.mode == 'validate':
        Config.repeatit = 1
        data_for_predictions_df, test = mdp.preprocessing.split(data_for_predictions_df, predicts=Config.predicts)
        data_for_predictions_df, _ = mdp.preprocessing.split(data_for_predictions_df, predicts=Config.predicts)

    # In data consolidation predicted column was replaced on index 0 as first column
    predicted_column_index = 0
    predicted_column_name = data_for_predictions_df.columns[0]

    ########################################
    ############# Data analyze ###### Analyze original data
    ########################################

    column_for_predictions_df = data_for_predictions_df.iloc[:, 0:1]
    used_models_assigned = {i: j for (i, j) in predictit.models.models_assignment.items() if i in Config.used_models}

    results = {}
    used_input_types = []

    for i in Config.used_models:
        used_input_types.append(Config.models_input[i])
    used_input_types = set(used_input_types)

    if Config.analyzeit == 1 or Config.analyzeit == 3:
        print("Analyze of unprocessed data")
        try:
            predictit.analyze.analyze_column(data_for_predictions_df.values[:, 0], window=30)
            predictit.analyze.analyze_data(data_for_predictions_df)
            predictit.analyze.decompose(data_for_predictions_df.values[:, 0], **Config.analyze_seasonal_decompose)
        except Exception:
            traceback_warning("Analyze failed")

    semaphor = None

    if Config.multiprocessing:

        if not Config.processes_limit:
            Config.processes_limit = multiprocessing.cpu_count() - 1

        if Config.multiprocessing == 'process':
            pipes = []
            semaphor = multiprocessing.Semaphore(Config.processes_limit)

        elif Config.multiprocessing == 'pool':
            pool = multiprocessing.Pool(Config.processes_limit)

            # It is not possible easy share data in multiprocessing, so results are resulted via callback function
            def return_result(result):
                for i, j in result.items():
                    results[i] = j

    ### Optimization loop

    if Config.optimization:
        option_optimization_list = Config.optimization_values
    else:
        option_optimization_list = ['Not optimized']
        Config.optimization_variable = None

    time_point = update_time_table(time_point)
    progress_phase = "Predict"
    update_gui(progress_phase, 'progress_phase')

    #######################################
    ############# Main loop ######## ANCHOR Main loop
    #######################################

    for optimization_index, optimization_value in enumerate(option_optimization_list):

        # Some Config values are derived from other values. If it has been changed, it has to be updated.
        if not Config.input_types:
            Config.update_references_input_types()
        if Config.optimizeit and not Config.models_parameters_limits:
            Config.update_references_optimize()


        ##########################################
        ############# Data extension #############
        ##########################################

        if Config.add_fft_columns:
            data_for_predictions_df = mdp.preprocessing.add_frequency_columns(data_for_predictions_df, window=Config.fft_window)

        if Config.do_data_extension:
            data_for_predictions_df = mdp.preprocessing.add_derived_columns(
                data_for_predictions_df, differences=Config.add_differences, second_differences=Config.add_second_differences,
                multiplications=Config.add_multiplications, rolling_means=Config.add_rolling_means,
                rolling_stds=Config.add_rolling_stds, mean_distances=Config.add_mean_distances, window=Config.rolling_data_window)

        ##############################################
        ############# Feature extraction #############
        ##############################################

        # data_for_predictions_df TODO

        #############################################
        ############ DATA preprocessing ###### ANCHOR Data preprocessing
        #############################################

        data_for_predictions, last_undiff_value, final_scaler = mdp.preprocessing.preprocess_data(
            data_for_predictions_df.values, remove_outliers=Config.remove_outliers, smoothit=Config.smoothit,
            correlation_threshold=Config.correlation_threshold, data_transform=Config.data_transform,
            standardizeit=Config.standardizeit)

        column_for_predictions_processed = data_for_predictions[:, predicted_column_index]

        data_shape = np.shape(data_for_predictions)
        data_length = len(column_for_predictions_processed)

        data_std = np.std(column_for_predictions_processed[-30:])
        data_mean = np.mean(column_for_predictions_processed[-30:])
        data_abs_max = max(abs(column_for_predictions_processed.min()), abs(column_for_predictions_processed.max()))

        multicolumn = 0 if data_shape[1] == 1 else 1

        if (Config.analyzeit == 2 or Config.analyzeit == 3) and optimization_index == len(option_optimization_list) - 1:

            print("\n\n Analyze of preprocessed data \n")
            try:
                predictit.analyze.analyze_column(column_for_predictions_processed, window=30)
            except Exception:
                traceback_warning("Analyze failed")

        min_data_length = 3 * Config.predicts + Config.default_n_steps_in

        if data_length < min_data_length or data_length < Config.repeatit + Config.default_n_steps_in + Config.predicts:
            Config.repeatit = 1
            min_data_length = 3 * Config.predicts + Config.default_n_steps_in

        assert (min_data_length < data_length), user_message('Set up less predicted values in settings or add more data', caption="To few data")

        if Config.mode == 'validate':
            models_test_outputs = [test]

        else:
            if Config.evaluate_type == 'original':
                models_test_outputs = mdp.inputs.create_tests_outputs(column_for_predictions_df.values.ravel(), predicts=Config.predicts, repeatit=Config.repeatit)

            elif Config.evaluate_type == 'preprocessed':
                models_test_outputs = mdp.inputs.create_tests_outputs(data_for_predictions[:, 0], predicts=Config.predicts, repeatit=Config.repeatit)

        for input_type_name in used_input_types:
            try:
                model_train_input, model_predict_input, model_test_inputs = mdp.inputs.create_inputs(
                    data_for_predictions, input_type_name=input_type_name, input_type_params=Config.input_types[input_type_name], mode=Config.mode,
                    predicts=Config.predicts, repeatit=Config.repeatit, predicted_column_index=predicted_column_index
                )

            except Exception:
                traceback_warning(f"Error in creating input type: {input_type_name} with option optimization: {optimization_value}")
                continue

            config_multiprocessed = Config.freeze()
            del config_multiprocessed['data']

            for iterated_model_index, (iterated_model_name, iterated_model) in enumerate(used_models_assigned.items()):
                if Config.models_input[iterated_model_name] == input_type_name:

                    predict_parameters = {
                        'Config': config_multiprocessed, 'iterated_model_train': iterated_model.train, 'iterated_model_predict': iterated_model.predict, 'iterated_model_name': iterated_model_name,
                        'iterated_model_index': iterated_model_index, 'optimization_index': optimization_index, 'optimization_value': optimization_value,
                        'option_optimization_list': option_optimization_list, 'model_train_input': model_train_input, 'model_predict_input': model_predict_input,
                        'model_test_inputs': model_test_inputs, 'data_abs_max': data_abs_max, 'data_mean': data_mean, 'data_std': data_std,
                        'last_undiff_value': last_undiff_value, 'models_test_outputs': models_test_outputs, 'final_scaler': final_scaler, 'semaphor': semaphor
                    }

                    if Config.models_input[iterated_model_name] in ['one_step', 'one_step_constant']:
                        if multicolumn and Config.predicts > 1:

                            user_warning(f"""Warning in model {iterated_model_name} \n\nOne-step prediction on multivariate data (more columns).
                                             Use batch (y lengt equals to predict) or do use some one column data input in Config models_input or predict just one value.""")
                            continue

                    if Config.multiprocessing == 'process':

                        # TODO duplex=False
                        pipes.append(multiprocessing.Pipe())
                        p = multiprocessing.Process(target=predictit.main_loop.train_and_predict, kwargs={**predict_parameters, **{'pipe': pipes[-1][1]}})
                        p.start()

                    elif Config.multiprocessing == 'pool':

                        pool.apply_async(predictit.main_loop.train_and_predict, (), predict_parameters, callback=return_result)

                    else:
                        results = {**results, **predictit.main_loop.train_and_predict(**predict_parameters)}

    if Config.multiprocessing:
        if Config.multiprocessing == 'process':

            for i in pipes:
                try:
                    results = {**results, **i[0].recv()}
                except Exception:
                    pass

        if Config.multiprocessing == 'pool':
            pool.close()
            pool.join()

    # Create confidence intervals
    if Config.confidence_interval:
        try:
            lower_bound, upper_bound = predictit.misc.confidence_interval(column_for_predictions_df.values, predicts=Config.predicts, confidence=Config.confidence_interval)

            bounds = True
        except Exception:
            bounds = False
            traceback_warning("Error in compute confidence interval")

    else:
        bounds = False

    ################################################
    ############# Results processing ######## ANCHOR Results processing
    ################################################

    # Criterion is the best of average from repetitions
    time_point = update_time_table(time_point)
    progress_phase = "Evaluation"
    update_gui(progress_phase, 'progress_phase')

    # Two kind of results we will create. Both as dataframe
    #   - First are all the details around prediction. Model errors, time, memory peak etc.
    #   - Second we have predicted values

    # Results such as trained model etc. that cannot be displayed in dataframe are in original results dict.

    # Convert results from dictionary to dataframe - exclude objects like trained model
    results_df = pd.DataFrame.from_dict(results, orient='index').drop(['Index'], axis=1)

    results_df.sort_values("Model error", inplace=True)

    # Generate date indexes for result predictions
    last_date = column_for_predictions_df.index[-1]

    if isinstance(last_date, (pd.core.indexes.datetimes.DatetimeIndex, pd._libs.tslibs.timestamps.Timestamp)):
        date_index = pd.date_range(start=last_date, periods=Config.predicts + 1, freq=column_for_predictions_df.index.freq)[1:]
        date_index = pd.to_datetime(date_index)

    else:
        date_index = list(range(last_date + 1, last_date + Config.predicts + 1))

    predictions = pd.DataFrame(index=date_index)

    for i in results_df['Results'].index:
        predictions[i] = results_df['Results'][i]

    best_model_name = results_df.index[0]
    best_model_predicts = predictions[best_model_name]

    if Config.mode == 'validate':
        best_model_name_plot = 'Test'
        predictions['Test'] = test

    else:
        best_model_name_plot = best_model_name

    if best_model_predicts is None or np.isnan(np.min(best_model_predicts)):
        raise RuntimeError(user_message("None of models finished predictions. Setup Config to one to see the warnings.",
                                        caption="All models failed for some reason"))

    predictions_for_plot = predictions
    predictions_for_plot.columns = [f"{i + 1} - {j}" for i, j in enumerate(predictions_for_plot.columns)]

    bounds_df = pd.DataFrame(index=date_index)

    if bounds:
        bounds_df['Upper bound'] = upper_bound
        bounds_df['Lower bound'] = lower_bound

    last_value = float(column_for_predictions_df.iloc[-1])

    predictions_for_plot = pd.concat([predictions_for_plot.iloc[:, :Config.plot_number_of_models], bounds_df], axis=1)
    predictions_with_history = pd.concat([column_for_predictions_df[-Config.plot_history_length:], predictions_for_plot], sort=False)
    predictions_with_history.iloc[-Config.predicts - 1, :] = last_value

    if Config.sort_results_by.lower() == 'name':
        results_df.sort_index(key=lambda x: x.str.lower(), inplace=True)
        predictions.sort_index(key=lambda x: x.str.lower(), inplace=True)

    time_df.append(['Complete time', round((time.time() - time_begin), 3)])
    time_df = pd.DataFrame(time_df, columns=["Part", "Time"])

    if Config.print_table == 1:
        models_table = tabulate(
            results_df[['Name', 'Model error']].iloc[:Config.print_number_of_models, :].values,
            headers=['Model', f'Average {Config.error_criterion} error'],
            tablefmt='grid', floatfmt='.3f', numalign="center", stralign="center")

    else:
        used_columns = set(('Name', f'Model error', 'Optimization value', 'Model time', 'Memory Peak\n[MB]')) & set(results_df.columns)

        models_table = tabulate(
            results_df[used_columns].set_index('Name', drop=True, inplace=False),
            headers=used_columns,
            tablefmt='grid', floatfmt=".2f", numalign="center", stralign="center")

    ### ANCHOR Print

    if Config.printit:
        if Config.print_best_model_result:
            print((f"\n Best model is {best_model_name} with results \n\n\t{best_model_predicts.values} \n\n\t with model error {Config.error_criterion} = "
                   f"{results_df.loc[best_model_name, 'Model error']}"))

        if Config.print_table:
            print(f"\n {models_table} \n")

        if Config.print_time_table:
            print(f'\n {tabulate(time_df.values, headers=time_df.columns, tablefmt="grid", floatfmt=".3f", numalign="center", stralign="center")} \n')

    #######################################
    ############# Plot ############# ANCHOR Plot
    #######################################

    time_point = update_time_table(time_point)
    progress_phase = "plot"
    update_gui(progress_phase, 'progress_phase')

    if Config.plotit:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)

            plot_return = 'div' if _GUI else ''
            div = predictit.plots.plot(
                predictions_with_history, plot_type=Config.plot_type, show=Config.show_plot, save=Config.save_plot,
                save_path=Config.save_plot_path, plot_return=plot_return, best_model_name=best_model_name_plot,
                legend=Config.plot_legend, predicted_column_name=predicted_column_name)

            if Config.plot_all_optimized_models:
                predictit.plots.plot(
                    predictions, plot_type=Config.plot_type, show=Config.show_plot, save=Config.save_plot,
                    legend=Config.plot_legend, save_path=Config.save_plot_path, best_model_name=best_model_name_plot)

    update_time_table(time_point)
    progress_phase = "Completed"
    update_gui(progress_phase, 'progress_phase')

    ################################
    ########### Return ###### ANCHOR Return
    ################################

    # Return stdout and stop collect warnings and printed output
    if _GUI:
        output = sys.stdout.getvalue()
        sys.stdout = stdout

        print(output)

    return_type = Config.return_type

    # Return original Config values before predict function
    Config.update(config_freezed)


    if return_type == 'best':
        return best_model_predicts

    elif return_type == 'all_dataframe':
        return predictions

    elif return_type == 'detailed_dictionary':
        detailed_dictionary = {
            'best': best_model_predicts,
            'all_predictions': predictions,
            'results': results_df,
            'predictions_with_history': predictions_with_history
        }

        if _GUI:
            detailed_dictionary.update({
                'plot': div,
                'output': output,

                'time_table': str(tabulate(time_df.values, headers=time_df.columns, tablefmt="html")),
                'models_table': str(tabulate(
                    results_df[['Name', 'Model_error']].iloc[:Config.print_number_of_models, :].values,
                    headers=['Model', f'Average {Config.error_criterion} error'],
                    tablefmt='html', floatfmt='.3f'))
            })

        else:
            detailed_dictionary.update({
                'time_table': time_df,
                'models_table': models_table,
            })

        return detailed_dictionary

    elif return_type == 'visual_check':
        return {'data_for_predictions (X, y)': data_for_predictions, 'model_train_input': model_train_input,
                'model_predict_input': model_predict_input, 'model_test_inputs': model_test_inputs, 'models_test_outputs': models_test_outputs}

    else:
        return results


def predict_multiple_columns(positional_data=None, positional_predicted_columns=None, **function_kwargs):
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

    if positional_data is not None:
        function_kwargs['data'] = positional_data

    if positional_predicted_columns is not None:
        function_kwargs['predicted_column'] = positional_predicted_columns

    config_freezed = Config.freeze()

    Config.update(function_kwargs)

    freqs = Config.freqs if Config.freqs else [Config.freq]
    predicted_columns = Config.predicted_columns if Config.predicted_columns else [Config.predicted_column]


    if predicted_columns in ['*', ['*']]:

        if isinstance(Config.data, str):
            Config.data = mdp.preprocessing.load_data(
                Config.data, header=Config.header, csv_style=Config.csv_style, predicted_table=Config.predicted_table,
                max_imported_length=Config.max_imported_length, request_datatype_suffix=Config.request_datatype_suffix,
                data_orientation=Config.data_orientation)

        predicted_columns = mdp.preprocessing.data_consolidation(Config.data).columns

    predictions = {}

    for fi, f in enumerate(freqs):

        for ci, c in enumerate(predicted_columns):

            try:
                predictions[f"Column: {c} - Freq: {f}"] = predict(predicted_column=c, freq=f)

            except Exception:
                traceback_warning(f"Error in making predictions on column {c} and freq {f}")


        # if Config.database_deploy:
        #     try:
        #         predictit.database.database_deploy(Config.server, Config.database, last_date, predictions[0], predictions[1], freq=f)
        #     except Exception:
        #         traceback_warning(f"Error in database deploying on freq {f}")

    # Return original Config values before predict function
    Config.update(config_freezed)

    return predictions


def compare_models(positional_data_all=None, positional_predicted_column=None, **function_kwargs):
    """Function that helps to choose apropriate models. It evaluate it on test data and then return results.
    After you know what models are the best, you can use only them in functions predict() or predict_multiple_columns.
    You can define your own test data and find best modules for your process.

    Args:
        data_all (dict): Dictionary of data names and data values (np.array).
            Examples:
              {data_1 = (my_dataframe, 'column_name_or_index')}
              (my_data[-2000:], my_data[-1000:])  # and 'predicted_column' as usually in Config.
        **kwargs (dict): All parameters of predict function. Check Config.py or run predictit.configuration.print_config() for parameters details.
    """

    if positional_data_all is not None:
        function_kwargs['data'] = positional_data_all

    if positional_predicted_column is not None:
        function_kwargs['predicted_column'] = positional_predicted_column

    config_freezed = Config.freeze()

    # Edit Config.py default values with arguments values if exist
    Config.update(function_kwargs)

    # Edit Config.py default values with arguments values if exist
    Config.update({'return_type': 'results', 'mode': 'validate', 'confidence_interval': 0, 'optimizeit': 0})

    # If no data_all inserted, default will be used
    if not Config.data_all:
        Config.data_all = {'sin': (mdp.generatedata.gen_sin(), 0),
                           'Sign': (mdp.generatedata.gen_sign(), 0),
                           'Random data': (mdp.generatedata.gen_random(), 0)}
        user_warning("Test data was used. Setup 'data_all' in Config...")

    results = {}
    unstardized_results = {}

    data_dict = Config.data_all
    same_data = False

    if isinstance(data_dict, (list, tuple, np.ndarray)):
        same_data = True
        data_dict = {f"Data {i}": (j, Config.predicted_column) for i, j in enumerate(data_dict)}

    optimization_number = len(Config.optimization_values) if Config.optimization_values else 1

    for i, j in data_dict.items():

        Config.data = j[0]
        if not same_data:
            Config.predicted_column = j[1]

        Config.plot_name = i

        try:
            result = predict()

            evaluated_matrix = np.zeros((Config.repeatit, len(Config.used_models), optimization_number))
            evaluated_matrix.fill(np.nan)

            for k in result.values():
                try:
                    if 'Results' and 'Test errors' in k:
                        evaluated_matrix[:, k['Index'][0], k['Index'][1]] = k['Model error']
                except Exception:
                    pass

            repeated_average = np.mean(evaluated_matrix, axis=0)

            results[i] = (repeated_average - np.nanmin(repeated_average)) / (np.nanmax(repeated_average) - np.nanmin(repeated_average))
            unstardized_results[i] = repeated_average

        except Exception:
            traceback_warning(f"Comparison for data {i} didn't finished.")
            results[i] = np.nan

    results_array = np.stack(list(results.values()), axis=0)
    unstardized_results_array = np.stack(list(unstardized_results.values()), axis=0)

    # results_array[np.isnan(results_array)] = np.inf

    all_data_average = np.nanmean(results_array, axis=0)
    unstardized_all_data_average = np.nanmean(unstardized_results_array, axis=0)

    models_best_results = []
    unstardized_models_best_results = []

    for i in all_data_average:
        models_best_results.append(np.nan if np.isnan(i).all() else np.nanmin(i))
    models_best_results = np.array(models_best_results)

    for i in unstardized_all_data_average:
        unstardized_models_best_results.append(np.nan if np.isnan(i).all() else np.nanmin(i))
    unstardized_models_best_results = np.array(unstardized_models_best_results)

    best_compared_model = int(np.nanargmin(models_best_results))
    best_compared_model_name = list(Config.used_models)[best_compared_model]

    print("\n\nTable of complete results. Percentual standardized error is between 0 and 1. If 0, model was the best on all defined data, 1 means it was the worst.")
    models_table = []

    # Fill the table
    for i, j in enumerate(Config.used_models):
        models_table.append([j, models_best_results[i], unstardized_models_best_results[i]])

    models_table = pd.DataFrame(models_table, columns=['Model', 'Percentual standardized error', 'Error average'])

    print(f"\n {tabulate(models_table.values, headers=models_table.columns, tablefmt='grid', floatfmt=('.3f'))} \n")

    print(f"\n\nBest model is {best_compared_model_name}")

    compared_result = {'Models table': models_table, 'Best model': best_compared_model_name}

    # If Config value optimization
    if all_data_average.shape[1] == 1:
        print("No Donfig variable optimization was applied")
        compared_result['Best optimized value'] = 'Not optimized'
    else:
        all_lengths_average = np.nanmean(all_data_average, axis=0)
        best_all_lengths_index = np.nanargmin(all_lengths_average)
        print(f"\n\nBest optimized value is {best_all_lengths_index}")
        compared_result['Best optimized value'] = best_all_lengths_index

    # Return original Config values before predict function
    Config.update(config_freezed)

    return compared_result


if __name__ == "__main__" and Config.used_function:
    if Config.used_function == 'predict':
        prediction_results = predict()

    elif Config.used_function == 'predict_multiple':
        prediction_results = predict_multiple_columns()

    elif Config.used_function == 'compare_models':
        compare_models()
