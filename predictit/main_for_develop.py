#%%

""" Just copy of inner content of main function. Developing is much faster
having all variables in variable explorer and all plots in jupyter cell because code is not in fuction but declarative.
After finishing developing, just copy result back in main.

"""

if __name__ == "__main__":

    from pathlib import Path
    import sys

    jupyter = 0
    this_path = Path(__file__).resolve().parents[1]
    this_path_string = str(this_path)

    from pathlib import Path
    import numpy as np
    #import matplotlib.pyplot as plt
    from prettytable import PrettyTable
    import time
    import pickle
    import os
    import plotly as pl
    import cufflinks as cfe
    import sklearn
    import pandas as pd
    import warnings
    from sklearn import preprocessing
    import traceback
    import inspect


    this_path = Path(__file__).resolve().parents[1]
    this_path_string = str(this_path)

    if 'predictit' not in sys.modules:

        # If used not as a library but as standalone framework, add path to be able to import predictit if not opened in folder
        sys.path.insert(0, this_path_string)

        import predictit

    from predictit.config import config, presets
    import predictit.data_prep as dp
    from misc import traceback_warning, _GUI

    def update_gui(content, id):
        try:
            gui_start.edit_gui_py(content, id)
        except Exception:
            pass

    gui = 0
    
    warnings.filterwarnings('once')
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)








    # Add everything printed + warnings to variable to be able to print in GUI
    if _GUI:
        import io

        stdout = sys.stdout
        sys.stdout = io.StringIO()

    if config["use_config_preset"] and config["use_config_preset"] != 'none':
        config.update(presets[config["use_config_preset"]])

    # Parse all functions parameters and it's values to edit config.py later
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)

    # # Edit config.py default values with arguments values if exist
    for i in args:

        if values[i] is not None:
            if i in config:
                config[i] = values[i]

            else:
                warnings.warn(f"\n \t Inserted option with function argument --{i} not found in config.py.\n")

    # Do not repeat actually mean evaluate once
    if not config['repeatit']:
        config['repeatit'] = 1

    # Definition of the table for spent time on code parts
    time_parts_table = PrettyTable()
    time_parts_table.field_names = ["Part", "Time"]

    def update_time_table(time_last):
        time_parts_table.add_row([progress_phase, time.time() - time_last])
        return time.time()
    time_point = time_begin = time.time()

    ##########################################
    ################## DATA ########### ANCHOR Data
    ##########################################

    progress_phase = "Data loading and preprocessing"
    update_gui(progress_phase, 'progress_phase')

    if config['data'] is None:

        ############# Load CSV data #############
        if config['data_source'] == 'csv':
            if config['csv_test_data_relative_path']:
                try:
                    data_location = this_path / 'predictit' / 'test_data'
                    csv_path = data_location / config['csv_test_data_relative_path']
                    config['csv_full_path'] = Path(csv_path).as_posix()
                except Exception:
                    print(f"\n ERROR - Test data load failed - Setup CSV adress and column name in config \n\n")
                    raise
            try:
                config['data'] = pd.read_csv(config['csv_full_path'], header=0).iloc[-config['datalength']:, :]
            except Exception:
                print("\n ERROR - Data load failed - Setup CSV adress and column name in config \n\n")
                raise

        ############# Load SQL data #############
        elif config['data_source'] == 'sql':
            try:
                config['data'] = predictit.database.database_load(server=config['server'], database=config['database'], freq=config['freq'], data_limit=config['datalength'], last=config['last_row'])
            except Exception:
                print("\n ERROR - Data load from SQL server failed - Setup server, database and predicted column name in config \n\n")
                raise

        elif config['data_source'] == 'test':
            config['data'] = predictit.test_data.generate_test_data.gen_random(config['datalength'])

    ### pd.Series ###

    if isinstance(config['data'], pd.Series):

        predicted_column_index = 0
        predicted_column_name = 'Predicted Column'

        data_for_predictions_df = pd.DataFrame(data[-config['datalength']:])

        if config['remove_outliers']:
            data_for_predictions_df = dp.remove_outliers(data_for_predictions_df, predicted_column_index=predicted_column_index, threshold=config['remove_outliers'])

        data_for_predictions = data_for_predictions_df.values

    ### pd.DataFrame ###

    elif isinstance(config['data'], pd.DataFrame):
        data_for_predictions_df = config['data'].iloc[-config['datalength']:, ]

        if isinstance(config['predicted_column'], str):

            predicted_column_name = config['predicted_column']
            predicted_column_index = data_for_predictions_df.columns.get_loc(predicted_column_name)
        else:
            predicted_column_index = config['predicted_column']
            predicted_column_name = data_for_predictions_df.columns[predicted_column_index]

        if config['date_index']:

            if isinstance(config['date_index'], str):
                data_for_predictions_df.set_index(config['date_index'], drop=True, inplace=True)
            else:
                data_for_predictions_df.set_index(data_for_predictions_df.columns[config['date_index']], drop=True, inplace=True)

            data_for_predictions_df.index = pd.to_datetime(data_for_predictions_df.index)

            if config['freq']:
                data_for_predictions_df.sort_index(inplace=True)
                data_for_predictions_df.resample(config['freq']).sum()

            else:
                config['freq'] = data_for_predictions_df.index.freq

                if config['freq'] is None:
                    data_for_predictions_df.reset_index(inplace=True)

        # Make predicted column index 0
        data_for_predictions_df.insert(0, predicted_column_name, data_for_predictions_df.pop(predicted_column_name))
        predicted_column_index = 0

        if config['other_columns']:

            data_for_predictions_df = dp.remove_nan_columns(data_for_predictions_df)

            if config['remove_outliers']:
                data_for_predictions_df = dp.remove_outliers(data_for_predictions_df, predicted_column_index=predicted_column_index, threshold=config['remove_outliers'])

            data_for_predictions_df = dp.keep_corelated_data(data_for_predictions_df)

            data_for_predictions = data_for_predictions_df.values.T

        else:
            if config['remove_outliers']:
                data_for_predictions_df = dp.remove_outliers(data_for_predictions_df, predicted_column_index=predicted_column_index, threshold=config['remove_outliers'])

            data_for_predictions = data_for_predictions_df[predicted_column_name].to_frame().values.T

    ### np.ndarray ###

    elif isinstance(config['data'], np.ndarray):
        data_for_predictions = config['data']
        predicted_column_name = 'Predicted column'

        if data_for_predictions.ndim == 2 and data_for_predictions.shape[0] != 1:
            if data_for_predictions.shape[0] > data_for_predictions.shape[1]:
                data_for_predictions = data_for_predictions.T
            data_for_predictions = data_for_predictions[:, -config['datalength']:]

            if config['other_columns']:
                # Make predicted column on index 0
                data_for_predictions[[0, config['predicted_column']], :] = data_for_predictions[[config['predicted_column'], 0], :]

                predicted_column_index = 0
                if config['remove_outliers']:
                    data_for_predictions = dp.remove_outliers(data_for_predictions, predicted_column_index=predicted_column_index, threshold=config['remove_outliers'])

                data_for_predictions = dp.keep_corelated_data(data_for_predictions)
            else:
                data_for_predictions = data_for_predictions[predicted_column_index]
                predicted_column_index = 0
                if config['remove_outliers']:
                    data_for_predictions = dp.remove_outliers(data_for_predictions, threshold=config['remove_outliers'])

            data_for_predictions_df = pd.DataFrame(data_for_predictions.T)
            data_for_predictions_df.rename(columns={0: predicted_column_name})

        else:
            data_for_predictions = data_for_predictions[-config['datalength']:].reshape(1, -1)

            predicted_column_index = 0
            if config['remove_outliers']:
                data_for_predictions = dp.remove_outliers(data_for_predictions, predicted_column_index=predicted_column_index, threshold=config['remove_outliers'])

            data_for_predictions_df = pd.DataFrame(data_for_predictions.reshape(-1), columns=[predicted_column_name])

    ### Data preprocessing, common for all datatypes ###

    data_for_predictions = data_for_predictions.astype(config['dtype'], copy=False)

    data_shape = data_for_predictions.shape

    column_for_prediction_dataframe = data_for_predictions_df[data_for_predictions_df.columns[0]].to_frame()
    column_for_plot = column_for_prediction_dataframe.iloc[-7 * config['predicts']:]

    last_value = column_for_plot.iloc[-1]
    try:
        number_check = int(last_value)

    except Exception:
        print(f"\n ERROR - Predicting not a number datatype. Maybe bad config['predicted_columns'] setup.\n Predicted datatype is {type(column_for_prediction[1])} \n\n")
        raise

    if config['data_transform'] == 'difference':

        for i in range(len(data_for_predictions)):
            data_for_predictions[i, 1:] = dp.do_difference(data_for_predictions[i])

        data_for_predictions = np.delete(data_for_predictions, 0, axis=1)

    if config['standardize'] == '01':
        data_for_predictions, final_scaler = dp.standardize(data_for_predictions, standardizer=config['standardize'])
    if config['standardize'] == '-11':
        data_for_predictions, final_scaler = dp.standardize(data_for_predictions, standardizer=config['standardize'])
    if config['standardize'] == 'standardize':
        data_for_predictions, final_scaler = dp.standardize(data_for_predictions, standardizer=config['standardize'])
    if config['standardize'] == 'robust':
        data_for_predictions, final_scaler = dp.standardize(data_for_predictions, standardizer=config['standardize'])

    if data_for_predictions.ndim == 1:
        column_for_prediction = data_for_predictions
    else:
        column_for_prediction = data_for_predictions[predicted_column_index]

    ###################################
    ############# Analyze ###### ANCHOR Analyze
    ###################################

    data_shape = np.shape(data_for_predictions)
    data_length = len(column_for_prediction)

    data_std = np.std(data_for_predictions[0, -30:])
    data_mean = np.mean(data_for_predictions[0, -30:])
    data_abs_max = max(abs(column_for_prediction.min()), abs(column_for_prediction.max()))

    if data_for_predictions.ndim == 1 or (data_for_predictions.ndim == 2 and (data_shape[0] == 1 or data_shape[1] == 1)):
        multicolumn = 0
    else:
        multicolumn = 1

    if config['analyzeit']:
        predictit.analyze.analyze_data(data_for_predictions.T, window=30)

        # TODO repair decompose
        #predictit.analyze.decompose(column_for_prediction_dataframe, freq=36, model='multiplicative')

    min_data_length = 3 * config['predicts'] + config['repeatit'] * config['predicts'] + config['default_n_steps_in']

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

    for i in models_names:

        # If no parameters or parameters details, add it so no index errors later
        if i not in config['models_parameters']:
            config['models_parameters'][i] = {}

    # Empty boxes for results definition
    # The final result is - [repeated, model, data, results]
    test_results_matrix = np.zeros((config['repeatit'], models_number, data_number, config['predicts']))
    evaluated_matrix = np.zeros((config['repeatit'], models_number, data_number))
    reality_results_matrix = np.zeros((models_number, data_number, config['predicts']))
    test_results_matrix.fill(np.nan)
    evaluated_matrix.fill(np.nan)
    reality_results_matrix.fill(np.nan)

    models_time = {}

    data_end = None

    time_point = update_time_table(time_point)
    progress_phase = "Predict"
    update_gui(progress_phase, 'progress_phase')

    trained_models = {}

    test_sequentions = np.zeros((config['repeatit'], config['predicts']))

    for i in range(config['repeatit']):

        test_sequentions[i] = column_for_prediction[-config['predicts'] - i: - i] if i > 0 else column_for_prediction[-config['predicts'] - i: ]

    test_sequentions = test_sequentions[::-1]

    used_input_types = []
    for i in models_names:
        used_input_types.append(config['models_input'][i])
    used_input_types = set(used_input_types)

    ############################################
    ############# Main loop ############# ANCHOR Main loop
    ############################################

    # Repeat evaluation on shifted data to eliminate randomness
    for data_length_index, data_length_iteration in enumerate(data_lengths):

        for input_type in used_input_types:
            used_sequention = predictit.define_inputs.create_inputs(input_type, data_for_predictions, predicted_column_index=predicted_column_index, multicolumn=multicolumn, predicts=config['predicts'], repeatit=config['repeatit'])

            for iterated_model_index, (iterated_model_name, iterated_model) in enumerate(config['used_models'].items()):
                if config['models_input'][iterated_model_name] == input_type:

                    if isinstance(used_sequention, tuple):
                        model_train_input = (used_sequention[0][data_length - data_length_iteration:, :], used_sequention[1][data_length - data_length_iteration: , :])
                        model_predict_input = used_sequention[2]
                        model_test_input = used_sequention[3]

                    elif used_sequention.ndim == 1:
                        model_train_input = model_predict_input = model_test_input = used_sequention[data_length - data_length_iteration:]

                    else:
                        model_train_input = model_predict_input = model_test_input = used_sequention[:, data_length - data_length_iteration:]





                    if config['optimizeit']:
                        if iterated_model_name in config['models_parameters_limits']:

                            try:
                                start_optimization = time.time()
                                model_kwargs = {**config['models_parameters'][iterated_model_name]}

                                best_kwargs = predictit.best_params.optimize(iterated_model, model_kwargs, config['models_parameters_limits'][iterated_model_name], test=test, train_input=used_sequention[config['models_input'][iterated_model_name]], fragments=config['fragments'], iterations=config['iterations'], time_limit=config['optimizeit_limit'], criterion=config['criterion'], name=iterated_model_name, details=config['optimizeit_details'])

                                for k, l in best_kwargs.items():

                                    config['models_parameters'][iterated_model_name][k] = l

                            except Exception:
                                if config['debug']:
                                    traceback_warning("Optimization didn't finished")

                            finally:
                                stop_optimization = time.time()
                                models_optimizations_time[i] = (stop_optimization - start_optimization)





                    try:
                        start = time.time()

                        # Train all models
                        trained_models[iterated_model_name] = iterated_model.train(model_train_input, **config['models_parameters'][iterated_model_name])

                        # Create predictions - out of sample
                        reality_results_matrix[iterated_model_index, data_length_index] = iterated_model.predict(model_predict_input, trained_models[iterated_model_name], predicts=config['predicts'])

                        # Remove wrong values out of scope to not be plotted
                        reality_results_matrix[iterated_model_index, data_length_index][abs(reality_results_matrix[iterated_model_index, data_length_index]) > 10 * data_abs_max] = np.nan

                        if config['power_transformed'] == 1:
                            test_results_matrix[iterated_model_index, data_length_index] = dp.fitted_power_transform(test_results_matrix[iterated_model_index, data_length_index], data_std, data_mean)

                        if config['standardize']:
                            reality_results_matrix[iterated_model_index, data_length_index] = final_scaler.inverse_transform(reality_results_matrix[iterated_model_index, data_length_index])

                        if config['data_transform'] == 'difference':
                            reality_results_matrix[iterated_model_index, data_length_index] = dp.inverse_difference(reality_results_matrix[iterated_model_index, data_length_index], last_value)

                    except Exception:
                        if config['debug']:
                            traceback_warning("Optimization didn't finished")

                    try:
                        for repeat_iteration in range(config['repeatit']):






                            print('model_test_input[repeat_iteration]', model_test_input)
                            print('test_sequentions[repeat_iteration]', test_sequentions[repeat_iteration])




                            # Create in-sample predictions to evaluate if model is good or not
                            if model_test_input.ndim == 1:
                                test_results_matrix[repeat_iteration, iterated_model_index, data_length_index] = iterated_model.predict(model_test_input, trained_models[iterated_model_name], predicts=config['predicts'])
                            else:
                                test_results_matrix[repeat_iteration, iterated_model_index, data_length_index] = iterated_model.predict(model_test_input[repeat_iteration], trained_models[iterated_model_name], predicts=config['predicts'])

                            if config['power_transformed'] == 2:
                                test_results_matrix[repeat_iteration, iterated_model_index, data_length_index] = dp.fitted_power_transform(test_results_matrix[repeat_iteration, iterated_model_index, data_length_index], data_std, data_mean)

                            evaluated_matrix[repeat_iteration, iterated_model_index, data_length_index] = predictit.evaluate_predictions.compare_predicted_to_test(test_results_matrix[repeat_iteration, iterated_model_index, data_length_index], test_sequentions[repeat_iteration], criterion=config['criterion'])

                    except Exception:

                        if config['debug']:
                            traceback_warning(f"Error in {iterated_model_name} model on data length {data_length_iteration}")

                    finally:
                        models_time[iterated_model_name] = (time.time() - start)

    ###########################################
    ############# Evaluate models ############# ANCHOR Table
    ###########################################

    # Criterion is the best of average from repetitions
    time_point = update_time_table(time_point)
    progress_phase = "Evaluation"
    update_gui(progress_phase, 'progress_phase')

    repeated_average = np.mean(evaluated_matrix, axis=0)

    model_results = []

    for i in repeated_average:
        model_results.append(np.nan if np.isnan(i).all() else np.nanmin(i))

    sorted_results = np.argsort(model_results)

    if config['compareit']:
        sorted_results = sorted_results[:config['compareit']]
    else:
        sorted_results = sorted_results[0]

    predicted_models = {}

    for i, j in enumerate(sorted_results):
        this_model = list(config['used_models'].keys())[j]

        if i == 0:
            best_model_name = this_model


        predicted_models[this_model] = {'order': i, 'criterion': model_results[j], 'predictions': reality_results_matrix[j, np.argmin(repeated_average[j])], 'data_length': np.argmin(repeated_average[j])}

    ##########################################
    ############# Results ############# ANCHOR Table
    ##########################################

    best_model_predicts = predicted_models[best_model_name]['predictions']

    if config['print_result']:
        print(f"\n Best model is {best_model_name} \n\t with results {best_model_predicts} \n\t with model error {config['criterion']} = {predicted_models[best_model_name]['criterion']} \n\t with data length {data_lengths[predicted_models[best_model_name]['data_length']]} \n\t with paramters {config['models_parameters'][best_model_name]} \n")


    # Definition of the table for results
    models_table = PrettyTable()
    models_table.field_names = ['Model', f"Average {config['criterion']} error"]

    # Fill the table
    for i, j in predicted_models.items():
        models_table.add_row([i, j['criterion']])


    if config['print_table']:
        print(f'\n {models_table} \n')

    ### Print detailed resuts ###

    if config['debug']:

        for i, j in enumerate(models_names):
            print(models_names[i])

            for k in range(data_number):
                print(f"\t With data length: {data_lengths[k]}  {config['criterion']} = {repeated_average[i, k]} \n")

            if config['optimizeit']:
                print(f"\t Time to optimize {models_optimizations_time[j]} \n")
                print("Best models parameters", config['models_parameters'][j])

    ###############################
    ######### Plot ######### ANCHOR Results
    ###############################
    time_point = update_time_table(time_point)
    progress_phase = "plot"
    update_gui(progress_phase, 'progress_phase')

    if config['plot']:

        plot_return = 'div' if _GUI else ''
        div = predictit.plot.plotit(column_for_plot, predicted_models, plot_type=config['plot_type'], show=config['show_plot'], save=config['save_plot'], save_path=config['save_plot_path'], plot_return=plot_return)

    time_point = update_time_table(time_point)
    progress_phase = 'finished'
    update_gui(progress_phase, 'progress_phase')
    time_parts_table.add_row(['Complete time', time.time() - time_begin])

    if config['print_time_table']:
        print(f'\n {time_parts_table} \n')

    # Return stdout and stop collect warnings and printed output
    if _GUI:
        output = sys.stdout.getvalue()
        sys.stdout = stdout














#%%
import numpy as np

from sklearn import linear_model, multioutput


da = np.array([range(300), range(1000, 1300)])

jj = pd.DataFrame(da)

jj.rename(columns={0: "x"})