import time
import numpy as np
from mylogging import traceback_warning, set_warnings
from mydatapreprocessing import preprocessing
import predictit


# This is core function... It should be sequentionally in the middle of main script in predict function, but it has to be 1st level function to be able to use in multiprocessing.
def train_and_predict(
        # Config
        trace_processes_memory, optimization, optimizeit, optimizeit_details, optimizeit_plot, optimizeit_limit,
        models_parameters, models_parameters_limits, fragments, iterations, error_criterion, power_transformed,
        standardizeit, data_transform, predicts, repeatit, evaluate_type, optimization_variable, multiprocessing,

        # Other
        set_warnings_params, iterated_model_train, iterated_model_predict, iterated_model_name, iterated_model_index, optimization_index,
        optimization_value, model_train_input, model_predict_input, model_test_inputs,
        data_abs_max, data_mean, data_std, models_test_outputs, last_undiff_value=None, final_scaler=None, pipe=None, semaphor=None,
        _IS_TESTED=False):

    set_warnings(set_warnings_params)

    # Global module variables changed in for example tests module ignored because module reload...
    # therefore reload all necessary variables from misc
    predictit.misc._IS_TESTED = _IS_TESTED

    # eres just dic, so cannot use alueyntax here
    if semaphor:
        semaphor.acquire()

    if trace_processes_memory:
        import tracemalloc

        tracemalloc.start()

    model_results = {}
    model_results['Name'] = iterated_model_name

    result_name = f"{iterated_model_name} - {optimization_value}" if optimization else f"{iterated_model_name}"

    if (optimizeit and optimization_index == 0 and iterated_model_name in models_parameters_limits):

        try:
            start_optimization = time.time()

            model_results['best_kwargs'] = predictit.best_params.optimize(
                iterated_model_train, iterated_model_predict, models_parameters[iterated_model_name], models_parameters_limits[iterated_model_name],
                model_train_input=model_train_input, model_test_inputs=model_test_inputs, models_test_outputs=models_test_outputs,
                fragments=fragments, iterations=iterations, time_limit=optimizeit_limit,
                error_criterion=error_criterion, name=iterated_model_name, details=optimizeit_details, plot=optimizeit_plot)

            for k, l in model_results['best_kwargs'].items():

                models_parameters[iterated_model_name][k] = l

            stop_optimization = time.time()
            model_results['optimization_time'] = stop_optimization - start_optimization

        except Exception:
            traceback_warning(f"Hyperparameters optimization of {iterated_model_name} didn't finished")


    start = time.time()

    try:

        # If no parameters or parameters details, add it so no index errors later
        if iterated_model_name not in models_parameters:
            models_parameters[iterated_model_name] = {}

        # Train all models
        trained_model = iterated_model_train(model_train_input, **models_parameters[iterated_model_name])

        # Create predictions - out of sample
        one_reality_result = iterated_model_predict(model_predict_input, trained_model, predicts)

        if np.isnan(np.sum(one_reality_result)) or one_reality_result is None:
            raise ValueError('NaN predicted from model.')

        # Remove wrong values out of scope to not be plotted
        one_reality_result[abs(one_reality_result) > 3 * data_abs_max] = np.nan

        # Do inverse data preprocessing
        if power_transformed:
            one_reality_result = preprocessing.fitted_power_transform(one_reality_result, data_std, data_mean)

        one_reality_result = preprocessing.preprocess_data_inverse(
            one_reality_result, final_scaler=final_scaler, last_undiff_value=last_undiff_value,
            standardizeit=standardizeit, data_transform=data_transform)

        tests_results = np.zeros((repeatit, predicts))
        test_errors = np.zeros(repeatit)

        # Predict many values in test inputs to evaluate which models are best - do not inverse data preprocessing, because test data are processed
        for repeat_iteration in range(repeatit):

            # Create in-sample predictions to evaluate if model is good or not
            tests_results[repeat_iteration] = iterated_model_predict(model_test_inputs[repeat_iteration], trained_model, predicts=predicts)

            if power_transformed:
                tests_results[repeat_iteration] = preprocessing.fitted_power_transform(tests_results[repeat_iteration], data_std, data_mean)

            if evaluate_type == 'preprocessed':
                test_errors[repeat_iteration] = predictit.evaluate_predictions.compare_predicted_to_test(
                    tests_results[repeat_iteration], models_test_outputs[repeat_iteration], error_criterion=error_criterion)

            tests_results[repeat_iteration] = preprocessing.preprocess_data_inverse(
                tests_results[repeat_iteration], final_scaler=final_scaler, last_undiff_value=last_undiff_value,
                standardizeit=standardizeit, data_transform=data_transform)

            if evaluate_type == 'original':
                test_errors[repeat_iteration] = predictit.evaluate_predictions.compare_predicted_to_test(
                    tests_results[repeat_iteration], models_test_outputs[repeat_iteration], error_criterion=error_criterion)

        model_results['Model error'] = test_errors.mean()
        if optimization_variable:
            model_results['Optimization value'] = optimization_value
        model_results['Results'] = one_reality_result
        model_results['Test errors'] = test_errors
        model_results['Index'] = (iterated_model_index, optimization_index)

        # For example tensorflow is not pickable, so sending model from process would fail. Trained models only if not multiprocessing
        if not multiprocessing:
            model_results['Trained model'] = trained_model

    except Exception:
        model_results['Model error'] = np.inf
        traceback_warning(f"Error in '{result_name}' model" if not optimization else f"Error in {iterated_model_name} model with optimized value: {optimization_value}")

    finally:
        model_results['Model time [s]'] = time.time() - start

        if trace_processes_memory:
            _, memory_peak_MB = tracemalloc.get_traced_memory()
            model_results['Memory Peak\n[MB]'] = memory_peak_MB / 10**6
            tracemalloc.stop()

        if semaphor:
            semaphor.release()

        if multiprocessing == 'process':
            pipe.send({f"{result_name}": model_results})
            pipe.close()

        else:
            return {f"{result_name}": model_results}
