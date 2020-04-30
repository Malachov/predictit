""" Visual test on various components. Mostly for data preparation and input creating functions.
Just run and manually check results.

"""
if __name__ == "__main__":

    import sys
    import pathlib
    import pandas as pd
    import numpy as np

    script_dir = pathlib.Path(__file__).resolve()
    lib_path_str = script_dir.parents[1].as_posix()
    sys.path.insert(0, lib_path_str)

    import predictit
    import predictit.data_preprocessing as dp
    from predictit.config import config

    # Data must have 2 dimensions. If you have only one column, reshape(-1, 1)!!!
    data = np.array([[1, 3, 5, 2, 3, 4, 5, 66, 3, 2, 4, 5, 6, 0, 0, 0, 0, 7, 3, 4, 55, 3, 2]]).T

    column_for_prediction = pd.DataFrame(data)

    data_multi_col = np.array([[1, 22, 3, 3, 5, 8, 3, 3, 5, 8], [5, 6, 7, 6, 7, 8, 3, 9, 5, 8], [8, 9, 10, 6, 8, 8, 3, 3, 7, 8]]).T

    dataframe_raw = pd.DataFrame(data_multi_col)
    predicted_column_index = 0


    print("""
            ###############
            ### Analyze ###
            ###############\n
    """)

    predictit.analyze.analyze_data(data, column_for_prediction)

    # Define some longer functions of data_preprocessing, that is bad to compute in f strings...
    seqs, Y, x_input, test_inputs = predictit.define_inputs.make_sequences(data, n_steps_in=6, n_steps_out=1, constant=1)
    seqs_2, Y_2, x_input2, test_inputs2 = predictit.define_inputs.make_sequences(data, n_steps_in=4, n_steps_out=2, constant=0)
    seqs_m, Y_m, x_input_m, test_inputs_m = predictit.define_inputs.make_sequences(data_multi_col, n_steps_in=4, n_steps_out=1, default_other_columns_length=None, constant=1)
    seqs_2_m, Y_2_m, x_input2_m, test_inputs2_m = predictit.define_inputs.make_sequences(data_multi_col, n_steps_in=3, n_steps_out=2, default_other_columns_length=1, constant=0)

    normalized, scaler = dp.standardize(data)
    normalized_multi, scaler_multi = dp.standardize(data_multi_col)


    print(f"""

            ##########################
            ### Data_preprocessing ###
            ##########################

        ##################
        ### One column ###
        ##################\n

    ### Remove outliers ###\n

    With outliers: \n {data} \n\nWith no outliers: \n{dp.remove_outliers(data, threshold = 3)} \n

    ### Difference transform ### \n
    Original data: \n {data} \n\nDifferenced data: \n{dp.do_difference(data[:, 0])} \n\n
    Backward difference: \n{dp.inverse_difference(dp.do_difference(data[:, 0]), data[0, 0])}\n

    ### Standardize ### \n
    Original: \n {data} \n\nStandardized: \n{normalized} \n

    Inverse standardization: \n{scaler.inverse_transform(normalized)} \n

    ### Split ### \n
    Original: \n {data} \n\nsplited train: \n{dp.split(data)[0]} \n \n\nsplited test: \n{dp.split(data)[1]} \n

    ### Make sequences - n_steps_in = 6, n_steps_out = 1, constant = 1 ### \n
    Original: \n {data} \n\nsequences: \n{seqs} \n\nY: \n{Y} \n\n\nx_input:{x_input} \n

    ### Make batch sequences - n_steps_in = 4, n_steps_out = 2, constant = 0 ### \n
    Original: \n {data} \n\nsequences: \n{seqs_2} \n\nY: \n{Y_2} \n \nx_input: \n{x_input2} \n

        ####################
        ### More columns ###
        ####################\n

    ### Remove outliers ### \n
    With outliers: \n {data_multi_col} \n\nWith no outliers: \n{dp.remove_outliers(data_multi_col, predicted_column_index=predicted_column_index, threshold = 1)} \n

    ### Standardize ### \n
    Original: \n {data_multi_col} \n\nStandardized: \n{normalized_multi} \n
    Inverse standardization: \n {scaler_multi.inverse_transform(normalized_multi[:, 0])} \n

    ### Split ### \n
    Original: \n {data_multi_col} \n\nsplited train: \n{dp.split(data_multi_col, predicts=2, predicted_column_index=0)[0]} \n \n\nsplited test: \n{dp.split(data_multi_col, predicts=2, predicted_column_index=0)[1]} \n

    ### Make sequences - n_steps_in=4, n_steps_out=1, default_other_columns_length=None, constant=1 ### \n
    Original: \n {data_multi_col} \n\nsequences: \n{seqs_m} \n\nY: \n{Y_m} \nx_input: \n\n{x_input_m} \n

    ### Make batch sequences - n_steps_in=3, n_steps_out=2, default_other_columns_length=1, constant=0 ### \n
    Original: \n {data_multi_col} \n\nsequences: \n{seqs_2_m} \n\nY: \n{Y_2_m} \nx_input: \n\n{x_input2_m} \n

    """)



    print(f"""

            #######################
            ### Defining inputs ###
            #######################\n
    """)


    print(f"""

            ###########################
            ### Data_postprocessing ###
            ###########################\n
    ### Fitt power transform ### \n
    Original: \n {data}, original std = {data.std()}, original mean = {data.mean()} \n\ntransformed: \n{dp.fitted_power_transform(data, 10, 10)} \n\ntransformed std = {dp.fitted_power_transform(data, 10, 10).std()},
    transformed mean = {dp.fitted_power_transform(data, 10, 10).mean()} (shoud be 10 and 10)\n

    """)

    config.update({
        'predicts': 3,
        'default_n_steps_in': 5,
        'used_models': {"Bayes ridge regression": predictit.models.sklearn_regression},
        'standardize': 0,
        'remove_outliers': 0,
    })
    inputs = predictit.main.predict(predictit.test_data.generate_test_data.gen_slope(100))