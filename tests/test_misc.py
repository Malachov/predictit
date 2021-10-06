import numpy as np

import mydatapreprocessing as mdp
import mypythontools

mypythontools.paths.PROJECT_PATHS.add_ROOT_PATH_to_sys_path()

import predictit


def test_other_models_functions():
    tf_optimizers = predictit.models.tensorflow.get_optimizers_loses_activations()
    sklearn_regressors = predictit.models.sklearn_regression.get_all_models()

    sequences = mdp.create_model_inputs.make_sequences(np.random.randn(100, 1), 5)
    predictit.models.autoreg_LNU.train(sequences, plot=False)

    assert tf_optimizers and sklearn_regressors
