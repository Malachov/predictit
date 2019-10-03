from sklearn.multioutput import MultiOutputRegressor
from sklearn_extensions.extreme_learning_machines.elm import GenELMRegressor
import numpy as np

from ..data_prep import make_sequences, make_x_input

def elm_gen(data, n_steps_in=50, predicts=7, predicted_column_index=0, output_shape='one_step', other_columns_lenght=None, constant=None, alpha=0.5):

    data = np.array(data)
    data_shape = np.array(data).shape

    if output_shape == 'one_step':
        X, y = make_sequences(data, n_steps_in=n_steps_in, predicted_column_index=predicted_column_index, other_columns_lenght=other_columns_lenght, constant=constant)
    if output_shape == 'batch':
        X, y = make_sequences(data, n_steps_in=n_steps_in, n_steps_out=predicts, predicted_column_index=predicted_column_index, other_columns_lenght=other_columns_lenght, constant=constant)

    reg = GenELMRegressor()
    multi_regressor = MultiOutputRegressor(reg)
    multi_regressor.fit(X, y)

    ## Data v jednom sloupci
    if len(data_shape) == 1:

        if output_shape == 'one_step':

            predictions = []
            x_input = make_x_input(data, n_steps_in=n_steps_in, constant=constant)

            for i in range(predicts):

                yhat = multi_regressor.predict(x_input)

                x_input = np.insert(x_input, n_steps_in, yhat[0], axis=1)
                x_input = np.delete(x_input, 0, axis=1 )
                predictions.append(yhat[0])

        if output_shape == 'batch':

            x_input = make_x_input(data, n_steps_in=n_steps_in, constant=constant)

            predictions = multi_regressor.predict(x_input)
            predictions = predictions[0]

    else:

    ## Data ve více sloupcích
        if not other_columns_lenght:
            other_columns_lenght = n_steps_in

        if output_shape == 'one_step':

            from models import ar
            
            predictions = []
            nu_data_shape = data.shape

            for i in range(predicts):

                x_input = make_x_input(data, n_steps_in=n_steps_in, predicted_column_index=predicted_column_index, other_columns_lenght=other_columns_lenght, constant=constant)

                nucolumn = []
                for_prediction = data[predicted_column_index]

                yhat = multi_regressor.predict(x_input)
                yhat_flat = yhat[0]
                predictions.append(yhat_flat)
                
                for_prediction = np.append(for_prediction, yhat)

                for j in data:
                    new = ar(j, predicts=1)
                    nucolumn.append(new)

                nucolumn_T = np.array(nucolumn).reshape(nu_data_shape[0], 1)
                data = np.append(data, nucolumn_T, axis=1)
                data[predicted_column_index] = for_prediction

        if output_shape == 'batch':

            x_input = make_x_input(data, n_steps_in=n_steps_in, predicted_column_index=predicted_column_index, other_columns_lenght=other_columns_lenght, constant=constant)

            predictions = multi_regressor.predict(x_input)
            predictions = predictions[0]

    predictions = np.array(predictions).reshape(-1)

    return predictions
