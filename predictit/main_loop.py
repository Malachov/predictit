import time
import numpy as np

import predictit


# This is core function... It should be sequentionally in the middle of main script in predict function, but it has to be 1st level function to be able to use in multiprocessing.
# To understand content, see code below function
def train_and_predict(
        config, iterated_model_train, iterated_model_predict, iterated_model_name, iterated_model_index, optimization_index,
        optimization_value, option_optimization_list, model_train_input, model_predict_input, model_test_inputs,
        data_abs_max, data_mean, data_std, models_test_outputs, last_undiff_value=None, final_scaler=None, pipe=None):

    model_results = {}

    if (config['optimizeit'] and optimization_index == 0 and iterated_model_name in config['models_parameters_limits']):

        try:
            start_optimization = time.time()

            model_results['best_kwargs'] = predictit.best_params.optimize(
                iterated_model_train, iterated_model_predict, config['models_parameters'][iterated_model_name], config['models_parameters_limits'][iterated_model_name],
                model_train_input=model_train_input, model_test_inputs=model_test_inputs, models_test_outputs=models_test_outputs,
                fragments=config['fragments'], iterations=config['iterations'], time_limit=config['optimizeit_limit'],
                error_criterion=config['error_criterion'], name=iterated_model_name, details=config['optimizeit_details'], plot=config['optimizeit_plot'])

            for k, l in model_results['best_kwargs'].items():

                config['models_parameters'][iterated_model_name][k] = l

        except Exception:
            predictit.misc.traceback_warning("Optimization didn't finished")

        finally:
            stop_optimization = time.time()

            model_results['optimization_time'] = stop_optimization - start_optimization

    try:
        start = time.time()

        model_results['name'] = iterated_model_name
        model_results['index'] = (iterated_model_index, optimization_index)
        model_results['optimization_index'] = optimization_index
        model_results['optimization_value'] = optimization_value

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
        if config['power_transformed']:
            one_reality_result = predictit.data_preprocessing.fitted_power_transform(one_reality_result, data_std, data_mean)

        one_reality_result = predictit.data_preprocessing.preprocess_data_inverse(
            one_reality_result, final_scaler=final_scaler, last_undiff_value=last_undiff_value,
            standardizeit=config['standardizeit'], data_transform=config['data_transform'])

        model_results['results'] = one_reality_result

        process_repeated_matrix = np.zeros((config['repeatit'], config['predicts']))
        process_evaluated_matrix = np.zeros(config['repeatit'])

        # Predict many values in test inputs to evaluate which models are best - do not inverse data preprocessing, because test data are processed
        for repeat_iteration in range(config['repeatit']):

            # Create in-sample predictions to evaluate if model is good or not
            process_repeated_matrix[repeat_iteration] = iterated_model_predict(model_test_inputs[repeat_iteration], model_results['trained_model'], predicts=config['predicts'])

            if config['power_transformed']:
                process_repeated_matrix[repeat_iteration] = predictit.data_preprocessing.fitted_power_transform(process_repeated_matrix[repeat_iteration], data_std, data_mean)

            if config['evaluate_type'] == 'original':
                process_repeated_matrix[repeat_iteration] = predictit.data_preprocessing.preprocess_data_inverse(
                    process_repeated_matrix[repeat_iteration], final_scaler=final_scaler, last_undiff_value=last_undiff_value,
                    standardizeit=config['standardizeit'], data_transform=config['data_transform'])

            process_evaluated_matrix[repeat_iteration] = predictit.evaluate_predictions.compare_predicted_to_test(
                process_repeated_matrix[repeat_iteration], models_test_outputs[repeat_iteration], error_criterion=config['error_criterion'])

        model_results['repeated_matrix'] = process_repeated_matrix
        model_results['evaluated_matrix'] = process_evaluated_matrix
        model_results['model_error'] = process_evaluated_matrix.mean()

    except Exception:

        predictit.misc.traceback_warning(f"Error in {iterated_model_name} model on data length {optimization_value}")

    finally:
        model_results['model_time'] = time.time() - start

        if config['multiprocessing'] == 'process':
            pipe.send(model_results)
            pipe.close()
        else:
            return model_results
