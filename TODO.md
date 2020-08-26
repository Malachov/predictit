# List of what could be done

!! Repair difference transform - last value changing

## Big Deals

- [ ] Rewrite own modules (autoreg LNU and Conjugate grad) core loops as C extension
- [ ] Optionally save and load trained models .npy format for numpy.By default save to trained models folder, as parameter save or load model elsewhere
- [ ] Do repeate average and more from evaluate in multiprocessing loop and then use sorting as `a = {k: v for k, v in sorted(x.items(), key=lambda item: item[1]['a'])}` then use just slicing for plot and print results and Remove predicted_models_for_table and predicted_models_for_plot

- [ ] Option to unstandardized error criterion to be able to compare absolute results of configuration

- [ ] Remove lengths loop and do some common way to find optimal data length, smooting, segmentation etc...
- [ ] Analyze output (residuals - mean, std etc. in comparison with original data)
- [ ] Feature extraction - E.G. autoencoder to shrink input multivariate data to small vectors
- [ ] Results postprocessing
- [ ] Implement nan values
  - [ ] One hot encoding: OneHotEncoder(dtype=np.int, sparse=True) dataframe string to num
  - [ ] Vectorize - Embedding
- [ ] New models
  - [ ] Ensamble learning - bagging - Best models average as one new model
  - [ ] HONU
  - [ ] Levenberg-Marquardt
  - [ ] Random walk with same std and mean and trend as very few last data
  - [ ] Add some boost method (lgboost library..., own adaboost for multivariate data)
  - [ ] ETS method
- [ ] Transformations
  - [ ] Fast fourier transform
  - [ ] Rolling window and rolling std transformation
- [ ] Generate new features inputting the model - Use transformations (difference, rolling mean, rolling std...)
- [ ] Segmentation for input data subselection - Try TICC method or Matrix profile method or something new
- [ ] Big data
  - [ ] PCA on higher dimension data
  - [ ] Dask for big data (dask.np.aray and dask dataframe) - chunk datasource
  - [ ] Incremental learning
    - [ ] In sklearn (partial_fit)
- [ ] Create prediction type - Regression, classification (binning inputs), anomaly prediction in config
  - [ ] Binnig inputs and classify- E.g. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer
- [ ] Fast optimization model - (optimize each parameter separately and just combine)
- [ ] Add scientific documentation into docstrings (equations and diagrams)
- [ ] Add exogenous parametrs where possible
- [ ] Remake GUI to React - make components
- [ ] Numba optimization (probably just own models - LNU autoreg model and conjugate grad)
- [ ] - Feature selection (which columns to use) - E.g. https://scikit-learn.org/stable/modules/feature_selection.html, https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b

## Deals

- [ ] Try recreate config as data class with type hints and comments that will be parsable
- [ ] Change GUI to vuetify - do tests.
- [ ] For error criterion use compare models values in normal predict to remove best decision_tree which is unfair on insample predictions
- [ ] CI - set github actions to publish.sh script - recreate documentation and publish on pypi
- [ ] Change config comments https://www.sphinx-doc.org/en/1.4.8/ext/autodoc.html#directive-autoattribute
- [ ] To compare models add second table - summ of error
- [ ] Full config to readthedocs documentation
- [ ] Ability to load and save config to json to folder
- [ ] Auto detect if data_source is csv, txt, and setup in 'data' and add option all_files_from_folder with suffix ''. Remove relative path to data and automatic try it. Change config data source just to data and do automatic parsing - dataframe or array or path
- [ ] Use gui_start also on pypi
- [ ] Print failed models apart and put not in table results
- [ ] Test data from sklearn and csv test data from web - use in test_it.py - no big repository + users can use test data
- [ ] Check if can some lists replace with sets (faster)
- [ ] Test dataframe data input for statsmodels - date as index
- [ ] Data preprocessing
  - [ ] Binning inputs
  - [ ] In remove_outliers function do not remove it, but interpolate by neighbors option
- [ ] In optimization, compare models on more results to reduce chance
- [ ] Remove sys.path.insert in main and do imports better way
- [ ] Translate and finish models **init** docstrings
- [ ] Add residuals matrix from results_matrix and add plotly box plot for residuals of models in main.py (https://plot.ly/python/box-plots/), or box plot for error criterion (data lengths and for repetitions) on compare_models
- [ ] Improve / edit existing models
  - [ ] For models with significant results diversity (ELM...) use optional bagging (mean of more results) in models
  - [ ] Add auto arima models from new libraries
  - [ ] Conjugate gradient and autoreg LNU for multiple column data
  - [ ] Add stochastig gradient method to autoreg (adam or adagrad)
- [ ] Finish config presets
