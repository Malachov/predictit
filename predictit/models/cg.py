import numpy as np
from ..data_prep import make_sequences, make_x_input

def cg(data, n_steps_in=50, predicts=7, predicted_column_index=0, epochs=100, constant=1, other_columns_lenght=None):

    data = np.array(data)
    data_shape = data.shape

    X, y = make_sequences(data, n_steps_in=n_steps_in, predicted_column_index=predicted_column_index, other_columns_lenght=other_columns_lenght, constant=constant)

    y = y.reshape(-1)

    w = np.zeros(X.shape[1])

    for i in range(epochs):
        b = np.dot(X.T,y)
        A = np.dot(X.T,X)
        re = b - np.dot(A,w) 
        p = re.copy()

        alpha = np.dot(re.T,re)/(np.dot(np.dot(p.T,A),p))
        w = w + alpha * p
        re_prev = re.copy()
        re = re_prev - alpha * np.dot(A,p)
        beta = np.dot(re.T,re) / np.dot(re_prev.T,re_prev)
        p = re + beta * p

    predictions = []
    x_input = make_x_input(data, n_steps_in=n_steps_in, constant=constant).reshape(-1)

    for i in range(predicts):
        ypre = np.dot(w, x_input)
        predictions.append(ypre)
        x_input = np.insert(x_input, n_steps_in, ypre)
        x_input = np.delete(x_input, 0)

    return predictions
