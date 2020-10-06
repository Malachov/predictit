#%%
""" Visual test on various components. Mostly for data preparation and input creating functions.
Just run and manually check results.

"""
### Config ###

print_analyze = 1
print_preprocessing = 1
print_data_flow = 1  # Show what are data inputing for train, for prediction, for testing etc.
print_postprocessing = 1


def visual_test(print_analyze, print_preprocessing, print_data_flow, print_postprocessing):
    import sys
    import pathlib
    import pandas as pd
    import numpy as np

    script_dir = pathlib.Path(__file__).resolve()
    lib_path_str = script_dir.parents[1].as_posix()
    sys.path.insert(0, lib_path_str)

    import predictit
    from mydatapreprocessing import preprocessing, inputs
    from predictit.configuration import Config

    np.set_printoptions(suppress=True, precision=1)

    # Data must have 2 dimensions. If you have only one column, reshape(-1, 1)!!!
    data = np.array([[1, 3, 5, 2, 3, 4, 5, 66, 3, 2, 4, 5, 6, 0, 0, 0, 0, 7, 3, 4, 55, 3, 2]]).T

    column_for_prediction = pd.DataFrame(data)

    data_multi_col = np.array([[1, 22, 3, 3, 5, 8, 3, 3, 5, 8], [5, 6, 7, 6, 7, 8, 3, 9, 5, 8], [8, 9, 10, 6, 8, 8, 3, 3, 7, 8]]).T

    # Some calculations, that are to long to do in f-strings - Just ignore...

    seqs, Y, x_input, test_inputs = inputs.make_sequences(data, predicts=7, repeatit=3, n_steps_in=6, n_steps_out=1, constant=1)
    seqs_2, Y_2, x_input2, test_inputs2 = inputs.make_sequences(data, predicts=7, repeatit=3, n_steps_in=4, n_steps_out=2, constant=0)
    seqs_m, Y_m, x_input_m, test_inputs_m = inputs.make_sequences(data_multi_col, predicts=7, repeatit=3, n_steps_in=4, n_steps_out=1, default_other_columns_length=None, constant=1)
    seqs_2_m, Y_2_m, x_input2_m, test_inputs2_m = inputs.make_sequences(data_multi_col, predicts=7, repeatit=3, n_steps_in=3, n_steps_out=2, default_other_columns_length=1, constant=0)

    normalized, scaler = preprocessing.standardize(data)
    normalized_multi, scaler_multi = preprocessing.standardize(data_multi_col)

    if print_analyze:

        print("""
                ###############
                ### Analyze ###
                ###############

            ### One predicted column \n""")

        predictit.analyze.analyze_column(column_for_prediction)

        print("### All columns \n")

        predictit.analyze.analyze_data(data_multi_col)



    if print_preprocessing:

        print(f"""

                ##########################
                ### Data_preprocessing ###
                ##########################

            ##################
            ### One column ###
            ################## \n

        ### Remove outliers ### \n

        With outliers: \n {data} \n\nWith no outliers: \n{preprocessing.remove_the_outliers(data, threshold = 3)} \n

        ### Difference transform ### \n
        Original data: \n {data} \n\nDifferenced data: \n{preprocessing.do_difference(data[:, 0])} \n\n
        Backward difference: \n{preprocessing.inverse_difference(preprocessing.do_difference(data[:, 0]), data[0, 0])}\n

        ### Standardize ### \n
        Original: \n {data} \n\nStandardized: \n{normalized} \n

        Inverse standardization: \n{scaler.inverse_transform(normalized)} \n

        ### Split ### \n
        Original: \n {data} \n\nsplited train: \n{preprocessing.split(data)[0]} \n \n\nsplited test: \n{preprocessing.split(data)[1]} \n

        ### Make sequences - n_steps_in = 6, n_steps_out = 1, constant = 1 ### \n
        Original: \n {data} \n\nsequences: \n{seqs} \n\nY: \n{Y} \n\nx_input:{x_input} \n\n Tests inputs:{test_inputs}\n

        ### Make batch sequences - n_steps_in = 4, n_steps_out = 2, constant = 0 ### \n
        Original: \n {data} \n\nsequences: \n{seqs_2} \n\nY: \n{Y_2} \n \nx_input: \n{x_input2} \n\n Tests inputs:{test_inputs2} \n

            ####################
            ### More columns ###
            ####################\n

        ### Remove outliers ### \n
        With outliers: \n {data_multi_col} \n\nWith no outliers: \n{preprocessing.remove_the_outliers(data_multi_col, threshold = 1)} \n

        ### Standardize ### \n
        Original: \n {data_multi_col} \n\nStandardized: \n{normalized_multi} \n
        Inverse standardization: \n {scaler_multi.inverse_transform(normalized_multi[:, 0])} \n

        ### Split ### \n
        Original: \n {data_multi_col} \n\nsplited train: \n{preprocessing.split(data_multi_col, predicts=2)[0]} \n \n\nsplited test: \n{preprocessing.split(data_multi_col, predicts=2)[1]} \n

        ### Make sequences - n_steps_in=4, n_steps_out=1, default_other_columns_length=None, constant=1 ### \n
        Original: \n {data_multi_col} \n\nsequences: \n{seqs_m} \n\nY: \n{Y_m} \nx_input: \n\n{x_input_m} \n\n Tests inputs:{test_inputs_m} \n

        ### Make batch sequences - n_steps_in=3, n_steps_out=2, default_other_columns_length=1, constant=0 ### \n
        Original: \n {data_multi_col} \n\nsequences: \n{seqs_2_m} \n\nY: \n{Y_2_m} \nx_input: \n\n{x_input2_m} \n\n Tests inputs:{test_inputs2_m} \n

        """)


    if print_data_flow:

        Config.update({
            'data': np.array(range(40)),
            'return_type': 'visual_check',
            'predicts': 3,
            'default_n_steps_in': 5,
            'standardizeit': 0,
            'remove_outliers': 0,
            'plotit': 0,
            'printit': 0,
            'repeatit': 3,
            'optimization': 0,
            'mode': 'predict',
            'validation_gap': 2,
            'models_input': {'Bayes ridge regression': 'batch', 'AR (Autoregression)': 'data_one_column'},
        })

        results = {
            'sklearn in predict mode': predictit.main.predict(used_models={"Bayes ridge regression"}, mode='prediction'),
            'sklearn in validate mode': predictit.main.predict(used_models={"Bayes ridge regression"}, mode='validate'),
            'ar in predict mode': predictit.main.predict(used_models={'AR (Autoregression)'}, mode='prediction'),
            'ar in validate mode': predictit.main.predict(used_models={'AR (Autoregression)'}, mode='validate')
        }

        print("""

                #######################
                ### Defining inputs ###
                #######################\n

        Input data = [0, 1, 2, 3... 48, 49, 50]
        Config values: 'predicts': 3, 'default_n_steps_in': 5, 'repeatit': 3, 'standardizeit': 0,
            'remove_outliers': 0

        In function compare_models with compare_mode train_everytime, it is automatically set up 'repeatit': 1, and 'validation_gap': 0\n
        """)

        for i, j in results.items():
            print(f"\n### Used data packets for model {i} ###\n")
            for k, l in j.items():
                print(f"{k}: \n{l}\n")


    if print_postprocessing:
        print(f"""

                ###########################
                ### Data_postprocessing ###
                ###########################\n
        ### Fitt power transform ### \n
        Original: \n {data}, original std = {data.std()}, original mean = {data.mean()} \n\ntransformed: \n{preprocessing.fitted_power_transform(data, 10, 10)} \n\ntransformed std = {preprocessing.fitted_power_transform(data, 10, 10).std()},
        transformed mean = {preprocessing.fitted_power_transform(data, 10, 10).mean()} (shoud be 10 and 10)\n

        """)


if __name__ == "__main__":
    visual_test(print_analyze=print_analyze, print_preprocessing=print_preprocessing, print_data_flow=print_data_flow, print_postprocessing=print_postprocessing)
