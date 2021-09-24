import time
import numpy as np
import mylogging
import predictit

# Lazy imports
# import tracemalloc


# This is core function... It should be sequentially in the middle of main script in predict function,
# but it has to be 1st level function to be able to use in multiprocessing.
def train_and_predict(
    config,
    # Functions to not import all modules
    preprocess_data_inverse,
    fitted_power_transform,
    # Other
    iterated_model_train,
    iterated_model_predict,
    iterated_model_name,
    iterated_model_index,
    optimization_index,
    optimization_value,
    model_train_input,
    model_predict_input,
    model_test_inputs,
    models_test_outputs,
    models_test_outputs_unstandardized,
    data_abs_max,
    data_mean,
    data_std,
    last_undiff_value=None,
    final_scaler=None,
    pipe=None,
    semaphor=None,
):
    """Inner function, that can run in parallel with multiprocessing.

    Note:
        config is just a dictionary passed as param, so cannot use dot syntax here.

    Args:
        Some values from predictit configuration.

    Returns:
        dict: Return dict of results.
    """

    logs_list = []
    warnings_list = []

    if config["multiprocessing"]:
        mylogging._misc.filter_warnings()
        mylogging.outer_warnings_filter(
            config["ignored_warnings"], config["ignored_warnings_class_type"]
        )
        mylogging.config.BLACKLIST = config["ignored_warnings"]
        mylogging.config.OUTPUT = config["logger_output"]
        mylogging.config.LEVEL = config["logger_level"]
        mylogging.config.FILTER = config["logger_filter"]
        mylogging.config.COLORIZE = config["logger_color"]
        logs_redirect = mylogging.redirect_logs_and_warnings_to_lists(
            logs_list, warnings_list
        )

    if config["is_tested"]:
        import mypythontools

        mypythontools.tests.setup_tests(matplotlib_test_backend=True)

    if semaphor:
        semaphor.acquire()

    if config["trace_processes_memory"]:
        import tracemalloc

        tracemalloc.start()

    model_results = {"Name": iterated_model_name}

    result_name = (
        f"{iterated_model_name} - {optimization_value}"
        if config["optimization"]
        else f"{iterated_model_name}"
    )

    if (
        config["optimizeit"]
        and optimization_index == 0
        and iterated_model_name in config["models_parameters_limits"]
    ):

        start_optimization = time.time()

        try:
            model_results["Best optimized parameters"] = predictit.best_params.optimize(
                iterated_model_train,
                iterated_model_predict,
                config["models_parameters"].get(iterated_model_name),
                config["models_parameters_limits"][iterated_model_name],
                model_train_input=model_train_input,
                model_test_inputs=model_test_inputs,
                models_test_outputs=models_test_outputs,
                time_limit=config["optimizeit_limit"],
                error_criterion=config["error_criterion"],
                name=iterated_model_name,
                iterations=config["iterations"],
                fragments=config["fragments"],
                details=config["optimizeit_details"],
                plot=config["optimizeit_plot"],
            )

        except TimeoutError:
            mylogging.traceback(
                f"Hyperparameters optimization of {iterated_model_name} didn't finished"
            )

        for k, l in model_results["Best optimized parameters"].items():

            if iterated_model_name not in config["models_parameters"]:
                config["models_parameters"][iterated_model_name] = {}
            config["models_parameters"][iterated_model_name][k] = l

        stop_optimization = time.time()
        model_results["Hyperparameter optimization time"] = (
            stop_optimization - start_optimization
        )

    start = time.time()

    try:

        # If no parameters or parameters details, add it so no index errors later
        if iterated_model_name not in config["models_parameters"]:
            config["models_parameters"][iterated_model_name] = {}

        # Train all models
        trained_model = iterated_model_train(
            model_train_input, **config["models_parameters"][iterated_model_name]
        )

        # Create predictions - out of sample
        one_reality_result = iterated_model_predict(
            model_predict_input, trained_model, config["predicts"]
        )

        if np.isnan(np.sum(one_reality_result)) or one_reality_result is None:
            raise ValueError("NaN predicted from model.")

        # Remove wrong values out of scope to not be plotted
        one_reality_result[abs(one_reality_result) > 3 * data_abs_max] = np.nan

        # Do inverse data preprocessing
        if config["power_transformed"]:
            one_reality_result = fitted_power_transform(
                one_reality_result, data_std, data_mean
            )

        one_reality_result = preprocess_data_inverse(
            one_reality_result,
            final_scaler=final_scaler,
            last_undiff_value=last_undiff_value,
            standardizeit=config["standardizeit"],
            data_transform=config["data_transform"],
        )

        tests_results = np.zeros((config["repeatit"], config["predicts"]))
        test_errors_unstandardized = np.zeros((config["repeatit"], config["predicts"]))
        test_errors = np.zeros(config["repeatit"])

        # Predict many values in test inputs to evaluate which models are best - do not inverse data preprocessing,
        # because test data are processed
        for repeat_iteration in range(config["repeatit"]):

            # Create in-sample predictions to evaluate if model is good or not
            tests_results[repeat_iteration] = iterated_model_predict(
                model_test_inputs[repeat_iteration],
                trained_model,
                predicts=config["predicts"],
            )

            if config["power_transformed"]:
                tests_results[repeat_iteration] = fitted_power_transform(
                    tests_results[repeat_iteration], data_std, data_mean
                )

            test_errors[
                repeat_iteration
            ] = predictit.evaluate_predictions.compare_predicted_to_test(
                tests_results[repeat_iteration],
                models_test_outputs[repeat_iteration],
                error_criterion=config["error_criterion"],
            )

            tests_results[repeat_iteration] = preprocess_data_inverse(
                tests_results[repeat_iteration],
                final_scaler=final_scaler,
                last_undiff_value=last_undiff_value,
                standardizeit=config["standardizeit"],
                data_transform=config["data_transform"],
            )

            test_errors_unstandardized[
                repeat_iteration
            ] = predictit.evaluate_predictions.compare_predicted_to_test(
                tests_results[repeat_iteration],
                models_test_outputs_unstandardized[repeat_iteration],
                error_criterion=config["error_criterion"],
            )

        model_results["Model error"] = test_errors.mean()
        model_results["Unstandardized model error"] = test_errors.mean()
        model_results["Results"] = one_reality_result
        model_results["Test errors"] = test_errors

        # For example tensorflow is not pickleable, so sending model from process would fail.
        # Trained models only if not multiprocessing
        if not ["multiprocessing"]:
            model_results["Trained model"] = trained_model

    except Exception:
        results_array = np.zeros(config["predicts"])
        results_array.fill(np.nan)
        test_errors = np.zeros((config["repeatit"], config["predicts"]))
        test_errors.fill(np.nan)

        model_results["Model error"] = np.inf
        model_results["Unstandardized model error"] = np.inf
        model_results["Best optimized parameters"] = None
        model_results["Results"] = results_array
        model_results["Test errors"] = test_errors
        error_message = (
            f"Error in '{result_name}' model"
            if not config["optimization"]
            else f"Error in {iterated_model_name} model with optimized value: {optimization_value}"
        )

        mylogging.traceback(caption=error_message)

    finally:
        model_results["Index"] = (optimization_index, iterated_model_index)
        model_results["warnings_list"] = warnings_list
        model_results["logs_list"] = logs_list
        model_results["Model time [s]"] = time.time() - start

        if config["optimization_variable"]:
            model_results["Optimization value"] = optimization_value

        if config["trace_processes_memory"]:
            _, memory_peak_MB = tracemalloc.get_traced_memory()
            model_results["Memory Peak\n[MB]"] = memory_peak_MB / 10 ** 6
            tracemalloc.stop()

        if config["multiprocessing"]:
            logs_redirect.close_redirect()

        if semaphor:
            semaphor.release()

        if config["multiprocessing"] == "process":
            pipe.send({f"{result_name}": model_results})
            pipe.close()

        else:
            return {f"{result_name}": model_results}
