#%%
#%%
# jen pro jupyter - smazat

# Dodelat / testpre jenom jedene return

import sys
sys.path.insert(0, r'C:\Users\daniel.malachov\Desktop\Diplomka')
sys.path.insert(0, r'C:\Users\daniel.malachov\Desktop\Diplomka\models')

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import time
import pickle
from pathlib import Path
import os
import plotly.offline as py
#import plotly.offline.graph_objs as go
import plotly.graph_objs as go
import cufflinks as cf
import sklearn
import pandas as pd
import warnings
from sklearn import preprocessing

from database import database_load
import config
import models
from data_prep import split, data_clean, remove_outliers
from test_pre import test_pre 
from analyze import analyze_data, decompose
from best_params import optimize
from confidence_interval import bounds
from database import database_load, database_deploy

'''
%load_ext autoreload
%autoreload 2
%aimport config, models, database
'''

# Veškerá nastavitelná data v config.py - ostatní jen pro zasvěcené
# Pokud nemáte pracovní složku nastavenou do modulu, načte jeho absolutní adresu a nastaví ji jako cwd
my_file = Path("main_function.py")
if not my_file.is_file():
    path = Path(cwd)
    os.chdir(path)

if config.debug:
    warnings.filterwarnings('once')

else:
    warnings.filterwarnings('ignore')

def make_predictions(predicted_column_name, freq='D'):




    ##########################################
    ################## DATA ########### ANCHOR Data
    ##########################################

    data_all_original = {}

    ############# Test data #############

    # Data all from picle
    if config.evaluate_test_data:

        # Testovací data jsou zapiklována kvůli větší rychlosti načítání. Při změne dat, je potřeba přepyklit
        # Funkce pickle_data_all uloží na disk všechna data definovaná v data_test. Ta budou následně automaticky načtena
        if config.piclkeit:
            from pickle_data import pickle_data_all
            pickle_data_all(datalength=config.datalength)

        data_folder = Path("test_data/")

        for i, j in config.data_all_pickle.items():
            file_name = i + '.pickle'
            file_adress = data_folder / file_name
            try:
                with open(file_adress, "rb") as input_file:
                    data = pickle.load(input_file)
            except:
                raise FileNotFoundError("\n \t Error - Najprv dej v config.py pickleit = 1, tím se data uloží na disk \n")

        data_all_original[i] = data
        data_names = list(data_all_original.keys())

        predicted_column_index = 0
        data_all = data_all_original.copy()

        try:
            data_for_predictions = [j for i,j in data_all.items() if i == config.data_name_for_predicts][0]
            data_shape = np.array(data_for_predictions).shape

        except IndexError:
            raise IndexError("\n \t Data for prediction are commented or not defined \n")

        column_for_prediction = data_for_predictions.copy()
        column_for_prediction_dataframe = pd.DataFrame()
        column_for_prediction_dataframe['History'] = column_for_prediction
        data_for_predictions_full = column_for_prediction_dataframe
        predicted_column_name = data_for_predictions_full.columns[0]
        data_number = len(data_all)

    ############# Real data #############

    if config.evaluate_test_data == 0:

        if config.data_source == 'csv':
        ############# Načtení dat z CSV #############
            try:
                data_for_predictions_full = pd.read_csv(config.csv_adress, header=0, index_col=0)
            except Exception as exc:
                raise FileNotFoundError("\n \t ERROR - Data load failed - Setup CSV adress and column name in config \n")

        if config.data_source == 'sql':

        ############# Načtení dat z SQL #############
            try:
                data_for_predictions_full = database_load(server=config.server, database=config.database, freq=freq, index_col=config.index_col, data_limit=config.datalength, last=config.last_row)
            except Exception as exc:
                raise ConnectionError("\n \t ERROR - Data load from SQL server failed - Setup server, database and predicted column name in config \n")

        data_shape = data_for_predictions_full.shape
        if data_shape[1] > data_shape[0]:
            data_for_predictions_full = data_for_predictions_full.T

        data_shape = data_for_predictions_full.shape

        if len(data_shape) == 1:
            data_for_predictions = data_for_predictions_full[-config.datalength:].values
            data_length = len(data_for_predictions_full)
            predicted_column_index = 0
            column_for_prediction = data_for_predictions.copy()
            column_for_prediction_dataframe = pd.DataFrame()
            column_for_prediction_dataframe[predicted_column_name] = data_for_predictions_full[predicted_column_name]

            if config.remove_outliers:
                data_for_predictions = remove_outliers(data_for_predictions, predicted_column_index=predicted_column_index, threshold=config.remove_outliers)

            if config.data_transform == 'difference':
                data_for_predictions = do_difference(data_for_predictions)

        else:
            predicted_column_index = data_for_predictions_full.columns.get_loc(predicted_column_name)
            column_for_prediction_dataframe = pd.DataFrame()
            column_for_prediction_dataframe[predicted_column_name] = data_for_predictions_full[predicted_column_name]
            column_for_prediction_all = np.array(column_for_prediction_dataframe).reshape(-1)
            column_for_prediction = column_for_prediction_all[-config.datalength:]

            if config.other_columns:

                ######## Clean, normalize, only correlated columns ########

                cleaned_data = data_clean(data_for_predictions_full)

                if config.remove_outliers:
                    cleaned_data = remove_outliers(cleaned_data, predicted_column_index=predicted_column_index, threshold=config.remove_outliers)

                if config.data_transform == 'difference':
                    last_undiff_value = cleaned_data[-1]
                    for i in range(len(cleaned_data)):
                        cleaned_data[i] = do_difference(cleaned_data[i])

                # Matice korelací
                cleaned_data = pd.DataFrame(cleaned_data, columns = data_for_predictions_full.columns)
                corr = cleaned_data.corr()
                names_to_keep = corr[corr[predicted_column_name] > abs(config.correlation_threshold)].index
                corelated_data = data_for_predictions_full[names_to_keep]

                data_for_predictions_unnormalized = corelated_data.values.T
                data_for_predictions_normalized = data_for_predictions_unnormalized.copy()

                for i in range(len(corelated_data.columns)):
                    unnormalized_column = data_for_predictions_unnormalized[i, :].reshape(-1, 1)
                    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
                    scaled = scaler.fit_transform(unnormalized_column)
                    data_for_predictions_normalized[i, :] = scaled.reshape(-1)

                unnormalized_predicted_column = data_for_predictions_unnormalized[predicted_column_index, :].reshape(-1, 1)
                final_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
                final_fitted_scaler = final_scaler.fit_transform(unnormalized_predicted_column)

                data_for_predictions = data_for_predictions_normalized[:, -config.datalength:]
                data_length = len(data_for_predictions[predicted_column_index])

            else:
                predicted_column_index = 0
                data_for_predictions = column_for_prediction
                data_length = len(data_for_predictions)
                data_shape = [2]
                if config.remove_outliers:
                    data_for_predictions = remove_outliers(data_for_predictions, predicted_column_index=predicted_column_index, threshold=config.remove_outliers)

        ############# Analyze #############
        if config.analyzeit:
            analyze_data(data_for_predictions.T, predicted_column_index=predicted_column_index, window=30)

            #decompose(column_for_prediction_dataframe, freq=36, model='multiplicative')


        data_for_trimming = data_for_predictions

        min_data_length = 3 * config.predicts + config.repeatit * config.predicts

        if data_length < min_data_length:
            config.repeatit = 1
            min_data_length = 3 * config.predicts

        assert min_data_length < data_length, 'To few data - set up less repeat value in settings or add more data'

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

    params_everywhere = {"predicts": config.predicts}

    if len(data_shape) > 1: 
        if predicted_column_index or predicted_column_index == 0:
            params_everywhere["predicted_column_index"] = predicted_column_index

    if config.optimizeit:
        best_model_parameters = {}
        models_optimizations_time = config.used_models.copy()

        for i, j in config.used_models.items():
            model_kwargs = {**config.models_parameters[i], **params_everywhere}
            start_optimization = time.time()

            try:
                best_kwargs = optimize(j, model_kwargs, config.models_parameters_limits[i], data_for_predictions, fragments=config.fragments, iterations=config.iterations, time_limit=config.optimizeit_limit, criterion=config.criterion, name=i, details=config.optimizeit_details)
                best_model_parameters[i] = best_kwargs

                for k, l in best_kwargs.items():

                    config.models_parameters[i][k] = l

            except Exception as exc:
                warnings.warn("\n \t Optimization didn't finished - {} \n".format(exc))

            finally:
                stop_optimization = time.time()
                models_optimizations_time[i] = (stop_optimization - start_optimization)


    # Definice prázdných schránek pro data
    # Matice pro výsledky má následující rozměr - [počet opakování, model, data, výsledky]
    results_matrix = np.zeros((config.repeatit, models_number, data_number, config.predicts))
    test_matrix = np.zeros((config.repeatit, models_number, data_number, config.predicts))
    evaluated_matrix = np.zeros((config.repeatit, models_number, data_number))
    results_matrix.fill(np.nan)
    test_matrix.fill(np.nan)
    evaluated_matrix.fill(np.nan)

    results_shape = results_matrix.shape

    models_time = {}

    # Opakuje výpočet nad několikrát zkrácenými daty, aby nejlépe hodnocený model neměl pouze štěstí
    for r in range(config.repeatit):

        if config.evaluate_test_data == 0:

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
            train_all[i], test_all[i] = split(list(data_all.values())[i], predicts = config.predicts, predicted_column_index=predicted_column_index)

        ##########################################
        ############# Predict ############# ANCHOR Predict
        ##########################################

        for m, (n, o) in enumerate(config.used_models.items()):
            for p, q in enumerate(train_all):

                try:
                    start = time.time()
                    results_matrix[r, m, p] = o(q, **params_everywhere, **config.models_parameters[n])
                    test_matrix[r, m, p] = test_all[p]

                except Exception as err:
                    warnings.warn("\n \t Error in compute {} model on data {} : \n \t \t {}".format(n, p, err))

                finally:
                    end = time.time()

            models_time[n] = (end - start)


        if config.evaluate_test_data:
            for s, t in data_all.items():
                if len(data_shape) == 1:
                    data_all[s] = t[:-config.predicts]
                else:
                    data_all[s] = t[:, :-config.predicts]
        else: 
            if len(data_shape) == 1:
                data_for_trimming = data_for_trimming[:-config.predicts]
            else:
                data_for_trimming = data_for_trimming[:, :-config.predicts]

        ##########################################
        ############# Evaluate Model ###### ANCHOR Evaluate
        ##########################################

    for i in range(results_shape[0]):
        for j in range(results_shape[1]):
            for k in range(results_shape[2]):

                if config.criterion == 'mape':
                    evaluated_matrix[i, j, k] = test_pre(results_matrix[i, j, k, :], test_matrix[i, j, k, :], criterion='mape', predicts=config.predicts)

                if config.criterion == 'rmse':
                    evaluated_matrix[i, j, k] = test_pre(results_matrix[i, j, k, :], test_matrix[i, j, k, :], criterion='rmse', predicts=config.predicts)

    # Pro testovací data je mape hodnota modelu průměrem dat z průměru úseků
    # Pro reálná data je mape hodnota modelu tou nejlepší hodnotou dat z průměru úseků
    repeated_average = np.nanmean(evaluated_matrix, axis=0)


    if config.evaluate_test_data:
        model_results = np.nanmean(repeated_average, axis=1)

    else:
        model_results = np.nanmin(repeated_average, axis=1)

    # Index nejlepšího modelu ve tvaru [model]
    best_model_index = np.unravel_index(np.nanargmin(model_results), shape=model_results.shape)[0]

    best_model_matrix = repeated_average[best_model_index]
    best_data_index = np.unravel_index(np.nanargmin(best_model_matrix), shape=best_model_matrix.shape)[0]

    # Vyhodnocení nejlepšího modelu
    best_model_name, best_model = list(config.used_models.items())[best_model_index]
    best_data_name, best_data = list(data_all.items())[best_data_index]

    best_data_len = len(best_data.reshape(-1))
    best_model_param = config.models_parameters[best_model_name]
    best_mape = np.nanmin(model_results)

    if len(data_shape) == 1:
        best_data_len = len(best_data.reshape(-1))
    else:
        best_data_len = len(best_data[predicted_column_index].reshape(-1))

    if config.compareit:

        if config.compareit >= models_number:
            next_number = models_number - 1
        else:
            next_number = config.compareit

        next_models_names = [np.nan] * next_number
        next_models = [np.nan] * next_number
        next_models_data_names = [np.nan] * next_number
        next_models_data = [np.nan] * next_number
        next_models_data_len = [np.nan] * next_number
        next_models_params = [np.nan] * next_number
        next_models_predicts = [np.nan] * next_number
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

            # Index nejlepšího modelu ve tvaru [model]
            next_model_index = np.unravel_index(np.nanargmin(results_copy), shape=results_copy.shape)[0]

            next_model_matrix = results_copy[next_model_index]

            next_data_index = np.unravel_index(np.nanargmin(next_model_matrix), shape=next_model_matrix.shape)[0]

            # Define other good models
            next_models_names[i], next_models[i] = list(rest_models.items())[next_model_index]
            next_models_data_names[i], next_models_data[i] = list(data_all.items())[next_data_index]
            next_models_params[i] = config.models_parameters[next_models_names[i]]

            if data_shape == 1 or not config.other_columns:
                next_models_data_len[i] = len(next_models_data[i])
            else:
                next_models_data_len[i] = len(next_models_data[i][0])

            nu_best_model_index = next_model_index
            next_best_model_name = next_models_names[i]

        # Evaluate models for comparison

            if not config.evaluate_test_data:
                if len(data_shape) == 1 or not config.other_columns:
                    trimmed_prediction_data = data_for_predictions[-next_models_data_len[i]:]

                else:
                    trimmed_prediction_data = data_for_predictions[:, -next_models_data_len[i]:]

            else: trimmed_prediction_data = data_for_predictions

            try:
                next_models_predicts[i] = next_models[i](trimmed_prediction_data, **params_everywhere, **next_models_params[i])

                if len(data_shape) > 1 or config.other_columns:
                    next_model_predicts_unnormalized = np.array(next_models_predicts[i]).reshape(-1, 1)
                    next_model_predicts_normalized = final_scaler.inverse_transform(next_model_predicts_unnormalized)

                    if config.data_transform == 'difference':
                        next_models_predicts[i] = inverse_difference(next_model_predicts_normalized, last_undiff_value)

                    next_models_predicts[i] = next_model_predicts_normalized.reshape(-1)

            except Exception as err:
                warnings.warn("\n \t Error in compute {} model on data {}: {}".format(n, p, err))

    # Dooptimalizování nejlepšího modelu
    if config.optimizeit_final:
        help_var = config.models_parameters[best_model_name].copy()
        kwargs = {key: value for (key, value) in help_var.items() if key in config.models_parameters_limits[best_model_name]}
        kwargs_limits = config.models_parameters_limits[best_model_name]

        best_kwargs = optimize(config.models_parameters_limits[best_model_name], kwargs, kwargs_limits, best_data, fragments=config.fragments_final, iterations=config.iterations_final, time_limit=config.optimizeit_limit, name=best_model_name)

        for k, l in best_kwargs.items():
            config.models_parameters[best_model_name][k] = l

    ####### Evaluate best Model ####### ANCHOR Evaluate best
    if not config.evaluate_test_data:
        if len(data_shape) == 1 or not config.other_columns:
            trimmed_prediction_data = data_for_predictions[-best_data_len:]

        else:
            trimmed_prediction_data = data_for_predictions[:, -best_data_len:]

    try:
        if len(data_shape) == 1 or not config.other_columns:
            best_model_predicts = best_model(trimmed_prediction_data, **params_everywhere, **best_model_param)

        else:
            best_model_predicts_unnormalized = best_model(trimmed_prediction_data, **params_everywhere, **best_model_param)
            best_model_predicts_unnormalized = np.array(best_model_predicts_unnormalized).reshape(-1, 1)
            best_model_predicts = final_scaler.inverse_transform(best_model_predicts_unnormalized)

            if config.data_transform == 'difference':
                best_model_predicts = inverse_difference(best_model_predicts, last_undiff_value)

            best_model_predicts = best_model_predicts.reshape(-1)

    except Exception as err:
        warnings.warn("\n \t Error in compute {} model on data {}: {}".format(n, p, err))

    ##########################################
    ############# Results ############# ANCHOR Results
    ##########################################

    # Definice tabulky pro výsledky
    models_table = PrettyTable()
    models_table.field_names = ["Model", "Average {} error".format(config.criterion), "Time"]

    # Vyplnění tabulky
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
                print("\t With data: {}  {} = {}".format(data_names[k], config.criterion, repeated_average[i, k]))

            if config.optimizeit:
                print("\t Time to optimize {} \n".format(models_optimizations_time[j]))
                    
            if config.optimizeit:
                print("Best models parameters", best_model_parameters[j])

    print('\n',models_table)

    print("\n Best model is {} \n\t with result MAPE {} \n\t with data {} \n\t with paramters {} \n".format(best_model_name, best_mape, best_data_name, best_model_param))

    ######### Graph #########
    if config.plot:

        arima_result, confidence, bound = bounds(column_for_prediction, predicts=config.predicts, confidence=config.confidence)
        bound = bound.T
        complete_dataframe = column_for_prediction_dataframe.iloc[-7 * config.predicts:, :].copy()


        last_date = data_for_predictions_full.index[-1]

        if not config.evaluate_test_data:

            date_index = pd.date_range(start=last_date, periods=config.predicts + 1, freq=freq)
            date_index = date_index[1:]

        complete_dataframe['Best prediction'] = None
        complete_dataframe['Lower bound'] = None
        complete_dataframe['Upper bound'] = None

        if not config.evaluate_test_data:
            results = pd.DataFrame(index=date_index)
            results.index = pd.to_datetime(results.index)

        else:
            results = pd.DataFrame()
        results['Best prediction'] = best_model_predicts
        results['Lower bound'] = bound[0]
        results['Upper bound'] = bound[1]

        for i in range(len(next_models_predicts)):
            name = next_models_names[i]
            results[name] = next_models_predicts[i]
            complete_dataframe[name] = None

        last_value = complete_dataframe[predicted_column_name].iloc[-1]

        if config.evaluate_test_data:
            complete_dataframe = pd.concat([complete_dataframe, results], ignore_index=True)
            complete_dataframe.iloc[-config.predicts - 1] = last_value

        else:
            complete_dataframe = pd.concat([complete_dataframe, results])
            complete_dataframe.iloc[-config.predicts - 1] = last_value

            try:
                complete_dataframe.index = pd.to_datetime(complete_dataframe.index)
            except:
                complete_dataframe.reset_index(drop=True, inplace=True)
                pass

        upper_bound = go.Scatter(
            name = 'Upper Bound',
            x = complete_dataframe.index,
            y = complete_dataframe['Upper bound'],
            mode = 'lines',
            marker = dict(color = "#444"),
            line = dict(width = 0),
            fillcolor = 'rgba(68, 68, 68, 0.3)',
            fill = 'tonexty')

        trace = go.Scatter(
            name = '1. {}'.format(best_model_name),
            x = complete_dataframe.index,
            y = complete_dataframe['Best prediction'],
            mode = 'lines',
            line = dict(color='rgb(51, 19, 10)', width=4),
            fillcolor = 'rgba(68, 68, 68, 0.3)',
            fill = 'tonexty')

        lower_bound = go.Scatter(
            name ='Lower Bound',
            x = complete_dataframe.index,
            y = complete_dataframe['Lower bound'],
            marker = dict(color="#444"),
            line = dict(width=0),
            mode = 'lines')

        history = go.Scatter(
            name=column_for_prediction_dataframe.columns[0],
            x = complete_dataframe.index,
            y = complete_dataframe[predicted_column_name],
            mode = 'lines',
            line = dict(color='rgb(31, 119, 180)', width=3),
            fillcolor = 'rgba(68, 68, 68, 0.3)',
            fill = None)

        layout = go.Layout(
            yaxis = dict(title='Datum'),
            title = 'Predikce',
            showlegend = False)

        graph_data = [lower_bound, trace, upper_bound, history]

        fig = go.Figure(data=graph_data, layout=layout)

        for i in range(next_number):
            fig.add_trace(go.Scatter(
                    x = complete_dataframe.index,
                    y = complete_dataframe[next_models_names[i]],
                    mode='lines',
                    name='{}. {}'.format(i + 2, next_models_names[i])))

        py.plot(fig, filename='predictions.html')




    return best_model_predicts, last_date

predictions = ['NAN'] * len(config.predicted_columns_names)
predictions_full = ['NAN'] * len(config.freqs)


for j in range(len(config.freqs)):
    for i in range(len(config.predicted_columns_names)):
        try:
            predictions[i], last_date = make_predictions(config.predicted_columns_names[i], freq=config.freqs[j])
        except Exception as excp:
            print('\n Error in making predictions on column {} and freq {} - {} \n'.format(i, j, excp))
        predictions_full[j] = predictions

    try:
        database_deploy(last_date, predictions[0], predictions[1], freq=config.freqs[j])
    except Exception as exx:
        print('\n Error in database deploying on freq {} - {} \n'.format(j, exx))

