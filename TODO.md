# List of what could be done

## First to do

- [ ] Remove unnecessary copies (only data_for_predictions_df and using .values) and check if input data stays the same after all models computations - the same in preprocessing
- [ ] Add transformations results as next input in parallel
  - [ ] Lagged values
  - [ ] Datetime transformations - is workday, is holiday, day, night etc...
- [ ] Conjugate gradient, autoreg LNU but also other models both multiple and one column data
- [ ] Test on some larger files
- [ ] Feature selection (which columns to use) - E.g. https://towardsdatascience.com/deep-dive-into-catboost-functionalities-for-model-interpretation-7cdef669aeed, https://scikit-learn.org/stable/modules/feature_selection.html, https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
- [ ] data_preprocessing
  - [ ] In remove_outliers function do not remove it, but interpolate by neighbors option - same as in nans
- [ ] Optionally save and load trained models .npy format for numpy.By default save to trained models folder, as parameter save or load model elsewhere
- [ ] Ensamble learning - bagging - Best models average as one new model
- [ ] Segmentation for input data subselection - Try TICC method or Matrix profile method or something new
- [ ] Dask for big data (dask.np.aray and dask dataframe) - chunk datasource
- [ ] Binnig inputs and classify- E.g. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer
- [ ] Fast boosted hyperparametrs optimization method option - (optimize each parameter separately - then again...)
- [ ] Change GUI to vuetify - do tests on it.
- [ ] Numba optimization (probably just own models - LNU autoreg model and conjugate grad)
- [ ] Ability to load and save config to json to folder
- [ ] For error criterion use compare models values in normal predict to remove best decision_tree which is unfair on insample predictions - config option - real error criterion (computationly harder)
- [ ] List of string path files, not only one file - use glob

## Big Deals

- [ ] Rewrite own modules (autoreg LNU and Conjugate grad) core loops as C extension
- [ ] Analyze output (residuals - mean, std etc. in comparison with original data)
- [ ] Feature extraction - E.G. autoencoder to shrink input multivariate data to small vectors
- [ ] Other feature extraction - Choose columns not on correlation, but based on error lowerage with simple model
- [ ] Results postprocessing
- [ ] Implement nan values
  - [ ] Vectorize - Embedding
- [ ] New models
  - [ ] LigthGBM
  - [ ] Prophet
  - [ ] CatBoostClassifierm CatBoostRegressor
  - [ ] HONU
  - [ ] statsmodels.tsa.vector_ar.var_model import VAR
  - [ ] Levenberg-Marquardt
  - [ ] Random walk with same std and mean and trend as very few last data
  - [ ] Add some boost method (lgboost library..., own adaboost for multivariate data)
  - [ ] ETS method
  - [ ] Bayes classification - sklearn
- [ ] Transformations
  - [ ] Hilbert huang tranformation (pyhht) - predict each part separately and then sum to have result
- [ ] Big data
  - [ ] PCA on higher dimension data - try to predict all the columns and inverse transformation. Second option: comprime only other than predicted colums
  - [ ] Incremental learning
    - [ ] In sklearn (partial_fit)
- [ ] Create prediction type - Regression, classification (binning inputs), anomaly prediction in config
- [ ] Add scientific documentation into docstrings (equations and diagrams)
- [ ] Add exogenous parametrs where possible

## Deals

- [ ] Remove complete dataframe copy in plots.py
- [ ] In GUI not remove ansi, but do not color
- [ ] Change config comments https://www.sphinx-doc.org/en/1.4.8/ext/autodoc.html#directive-autoattribute and use in argparse in for loop
- [ ] Make copy of input data only once - remove in main
- [ ] Remove other columns option and use default_other_columns_length to create one_column_input data
- [ ] Define some set of datasets and evaluate results on commit with tag. save results into csv and evalueate KPI
- [ ] Tests for GUI and unit test for define inputs
- [ ] Test dataframe data input for statsmodels - date as index
- [ ] Intellisense option values from list somehow to give settings options
- [ ] Do repeate average and more from evaluate in multiprocessing loop and then use sorting as `a = {k: v for k, v in sorted(x.items(), key=lambda item: item[1]['a'])}` then use just slicing for plot and print results and Remove predicted_models_for_table and predicted_models_for_plot
- [ ] Try recreate config as data class with type hints and comments that will be parsable
- [ ] Full config to readthedocs documentation
- [ ] Print failed models apart and put not in table results
- [ ] Check if can some lists replace with sets (faster)
- [ ] In optimization, compare models on more results to reduce chance. First do create real-error-criterion - than use it.
- [ ] Remove sys.path.insert in main and do imports better way
- [ ] Translate and finish models **init** docstrings
- [ ] Add residuals matrix from results_matrix and add plotly box plot for residuals of models in main.py (https://plot.ly/python/box-plots/), or box plot for error criterion (data lengths and for repetitions) on compare_models
- [ ] Improve / edit existing models
  - [ ] For models with significant results diversity (ELM...) use optional bagging (mean of more results) in models
  - [ ] Add auto arima models from new libraries
  - [ ] Add stochastic gradient method to autoreg (adam or adagrad)
- [ ] Finish config presets
