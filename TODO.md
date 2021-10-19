# List of what could be done

Tagged with complexity and sorted by priority

- [ ] EASY - Define type for data input in mydatapreprocessing, import and use
- [ ] EASY - In predict multiple change plot name to predicted column
- [ ] EASY - Test on some larger files (Use warning if big data and hard settings)
- [ ] COMPLEX - Define some set of datasets (good for prediction) and evaluate results on commit with tag. save results into csv and ate KPI - Add data to default compare models
- [ ] COMPLEX - Create class Model. Configure models in its own settings (be able to configure various models in various way). Use instances of models (multiple concrete settings) Add save and load model method
- [ ] EASY - Ability to load and save config to json to folder (mostly for GUI)
- [ ] COMPLEX - Add exogenous parameters where possible
- [ ] COMPLEX - Dimensionality reduction - E.G. autoencoder and PCA to shrink input multivariate data to small vectors. Choose what to compress (not predicted column).
- [ ] COMPLEX - Rewrite GUI with mypythontools.pyvueeel
- [ ] COMPLEX - Feature selection - Choose columns not on correlation, but based on error lowerage with simple model (neural net (better) as well as regression (faster)) E.g. https://towardsdatascience.com/deep-dive-into-catboost-functionalities-for-model-interpretation-7cdef669aeed, https://scikit-learn.org/stable/modules/feature_selection.html, https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
- [ ] COMPLEX - Segmentation for input data subselection - Try TICC method or Matrix profile method or something new
- [ ] MEDIUM - Ensamble learning - bagging - Best models average as one new model
- [ ] COMPLEX - Analyze output (residuals - mean, std etc. in comparison with original data)
- [ ] MEDIUM - Add residuals matrix from results_matrix and add plotly box plot for residuals of models in main.py (https://plot.ly/python/box-plots/), or box plot for error criterion (data lengths and for repetitions) on compare_models
- [ ] MEDIUM - Add transformations results as next input in parallel (remove replacement as in diff transformation)
  - [ ] Add datetime transformations - is workday, is holiday, day, night etc...
  - [ ] Hilbert huang transformation (pyhht) - predict each part separately and then sum to have result
- [ ] COMPLEX Incremental learning (In sklearn partial_fit). Solve how to cope with statsmodels.
- [ ] MEDIUM - Be able to use validation mode in predict to get real error criteria for compare
- [ ] MEDIUM - In sklearn model - auto select model (use pipeline?)
- [ ] EASY - In optimization, compare models on more results to reduce chance.
- [ ] MEDIUM - Fast boosted hyperparametrs optimization method option - (optimize each parameter separately - then again...)
- [ ] COMPLEX - New models
  - [ ] LigthGBM
  - [ ] Prophet
  - [ ] CatBoostClassifierm CatBoostRegressor
  - [ ] HONU
  - [ ] statsmodels.tsa.vector_ar.var_model import VAR
  - [ ] Random walk with same std and mean and trend as very few last data
  - [ ] Add some boost method (lgboost library..., own adaboost for multivariate data)
  - [ ] ETS method
  - [ ] Bayes classification - sklearn
- [ ] COMPLEX - Improve / edit existing models
  - [ ] - Add stochastic gradient method to autoreg (adam or adagrad)
- [ ] MEDIUM - Add scientific documentation for own models into docstrings (equations and diagrams) for other models add links

- [ ] MEDIUM - Try multiprocessing.manager
- [ ] EASY - Try dataframe data input for statsmodels - date as index
- [ ] EASY - Test whether plotly is slower than matplotlib and document it in plots docstrings.
- [ ] EASY - Store tensorflow mlp and lstm separately and by default.
- [ ] EASY - Remove unnecessary copies (only data_for_predictions_df and using .values) and check if input data stays the same after all models computations - the same in preprocessing - use new inplace param
- [ ] EASY - Dask for big data (dask.np.aray and dask dataframe) - chunk datasource (optionally in settings)
- [ ] EASY - Test Numba optimization (probably just own models - LNU autoreg model and conjugate grad)
- [ ] EASY - Check if can some lists replace with sets (faster)
- [ ] Rewrite logging settings into config method
