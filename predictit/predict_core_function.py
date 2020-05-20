
import numpy as np
import time



import predictit
# from predictit.config import config
from predictit.misc import traceback_warning


def train_and_predict(
        iterated_model_train, iterated_model_predict, iterated_model_name, iterated_model_index, data_length_index,
        data_length_iteration, data_lengths, model_train_input, model_predict_input, model_test_inputs, model_parameters,
        predicts, optimizeit, data_transform, standardize, power_transformed, repeatit, error_criterion):

    model_results = {}

    # if config['optimizeit'] and data_length_index == 0:
    #     if iterated_model_name in config['models_parameters_limits']:

            # try:
            #     start_optimization = time.time()

            #     model_results['best_kwargs'] = predictit.best_params.optimize(
            #         iterated_model, config['models_parameters'][iterated_model_name], config['models_parameters_limits'][iterated_model_name],
            #         model_train_input=model_train_input, model_test_inputs=model_test_inputs, models_test_outputs=models_test_outputs,
            #         fragments=config['fragments'], iterations=config['iterations'], time_limit=config['optimizeit_limit'],
            #         error_criterion=config['error_criterion'], name=iterated_model_name, details=config['optimizeit_details'])

            #     for k, l in model_results['best_kwargs'].items():

            #         config['models_parameters'][iterated_model_name][k] = l

            # except Exception:
            #     traceback_warning("Optimization didn't finished")

            # finally:
            #     stop_optimization = time.time()

            #     model_results['optimization_time'] = stop_optimization - start_optimization

    try:
        start = time.time()

        model_results['name'] = iterated_model_name
        model_results['index'] = (iterated_model_index, data_length_index)

        # Train all models
        model_results['trained_model'] = iterated_model_train(model_train_input, **model_parameters)

        # Create predictions - out of sample
        one_reality_result = iterated_model_predict(model_predict_input, model_results['trained_model'], predicts)

        if np.isnan(np.sum(one_reality_result)) or one_reality_result is None:
            return

        # Remove wrong values out of scope to not be plotted
        one_reality_result[abs(one_reality_result) > 3 * data_abs_max] = np.nan

        # Do inverse data preprocessing
        if power_transformed == 1:
            one_reality_result = predictit.data_preprocessing.fitted_power_transform(one_reality_result, data_std, data_mean)

        if standardize:
            one_reality_result = final_scaler.inverse_transform(one_reality_result.reshape(-1, 1)).ravel()

        if data_transform == 'difference':
            one_reality_result = predictit.data_preprocessing.inverse_difference(one_reality_result, last_undiff_value)

        model_results['results'] = one_reality_result

        process_repeated_matrix = np.zeros((repeatit, predicts))
        process_evaluated_matrix = np.zeros(repeatit)

        # Predict many values in test inputs to evaluate which models are best - do not inverse data preprocessing, because test data are processed
        for repeat_iteration in range(repeatit):

            # Create in-sample predictions to evaluate if model is good or not

            process_repeated_matrix[repeat_iteration] = iterated_model_predict(model_test_inputs[repeat_iteration], model_results['trained_model'], predicts=predicts)

            if power_transformed == 2:
                process_repeated_matrix[repeat_iteration] = predictit.data_preprocessing.fitted_power_transform(process_repeated_matrix[repeat_iteration], data_std, data_mean)

            process_evaluated_matrix[repeat_iteration] = predictit.evaluate_predictions.compare_predicted_to_test(
                process_repeated_matrix[repeat_iteration], models_test_outputs[repeat_iteration], error_criterion=error_criterion)

        model_results['repeated_matrix'] = process_repeated_matrix
        model_results['evaluated_matrix'] = process_evaluated_matrix

    except Exception:

        traceback_warning(f"Error in {iterated_model_name} model on data length {data_length_iteration}")

    finally:
        model_results['model_time'] = time.time() - start

        # if config['multiprocessing'] == 'process':
        #     pipe.send(model_results)
        #     pipe.close()
        # else:
        #     return model_results
    return model_results
