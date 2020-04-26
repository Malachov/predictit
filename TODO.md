# List of what could be done

## Now working on
- [x] New criterion model - with lagged values / autocorrelation for anomaly prediction


## Big Deals

- [ ] Feature extraction - E.G. autoencoder to shrink input multivariate data to small vectors
- [ ] Analyze output (residuals - mean, std etc. in comparison with original data)
- [ ] Results postprocessing
- [ ] Implement nan values
    - [ ] One hot encoding: OneHotEncoder(dtype=np.int, sparse=True) dataframe string to num
    - [ ] Vectorize - Embedding
- [ ] Nem models
    - [ ] HONU
    - [ ] Levenberg-Marquardt
    - [ ] Random walk with same std and mean (and bagging (averaging))
    - [ ] Add some boost method (lgboost library..., own adaboost for multivariate data)
    - [ ] ETS method
- [ ] Serialize transformations and redived values as other columns
- [ ] Transformations
    - [ ] Fast fourier transform
    - [ ] Rolling window and rolling std transformation
- [ ] Generate new features inputting the model (difference, rolling mean, rolling std...)
- [ ] Input data selection (delete unusual input vectors)
- [ ] Data smoothing as a parametr
- [ ] Enable trained models output - Computed weights - just compute result and do not learn model again
- [ ] Big data
    - [ ] PCA on higher dimension data
    - [ ] Dask for big data (dask.np.aray and dask dataframe) - chunk datasource
    - [ ] Incremental learning
        - [ ] In sklearn (partial_fit)
- [ ] Create prediction type - Regression, classification (binning inputs), anomaly prediction in config
    - [ ] Binnig inputs - E.g. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer
- [ ] Ensamble learning - combine best models results to one result
- [ ] Fast optimization model - (optimize each parameter separately and just combine)
- [ ] Add scientific documentation into docstrings (equations and diagrams)
- [ ] Add exogenous parametrs where possible
- [ ] Finish TODOS in GUI
- [ ] Do segmentation in time series (Matrix profile method) - 1) Use in analyze, 2) Detect anomalies
- [ ] Numba optimization
- [ ] - Feature selection (which columns to use) - E.g. https://scikit-learn.org/stable/modules/feature_selection.html, https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
- [ ] Do validate predictions on repeatit from config - generate different models inputs and test inputs

## Deals

- [ ] Use gui_start also on pypi
- [ ] Change congig data source just to data and do automatic parsing - dataframe or array or path
- [ ] Print failed models apart and put not in table results
- [ ] Test data from sklearn and csv test data from web - use in test_it.py - no big repository
- [ ] Do table from detailed results (too long)
- [ ] Check if can some dictionaries replace with sets
- [ ] Test dataframe data input for statsmodels - date as index
- [ ] Separate Plot to function + Plot_all during learning option - with real validate data for comparison
- [ ] Add evaluation methods to visual test
- [ ] Data preprocessing
    - [ ] Binning inputs
    - [ ] In remove_outliers function do not remove it, but interpolate by neighbors
- [ ] Add residuals matrix from results_matrix
- [ ] In optimization, compare models on more results to reduce chance
- [ ] Remove sys.path.insert in main and do imports better way
- [ ] Translate and finish models __init__ docstrings
- [ ] Add plotly box plot for residuals of models in main.py (https://plot.ly/python/box-plots/), and box plot for error criterion (data lengths and for repetitions) on compare_models
- [ ] Improve / edit existing models
    - [ ] Repair Sarimax model
    - [ ] For models with significant results diversity (ELM...) use optional bagging (mean of more results)
    - [ ] Add auto arima models from new libraries
    - [ ] Conjugate gradient and autoreg LNU for multiple column data
    - [ ] Add stochastig gradient method to autoreg (adam or adagrad)
- [ ] Make waterfall diagram of spent time - preprocess - train - evaluation, plot - make progress bar for GUI
- [ ] In memory profile just call the function, do not copy and paste
- [ ] Finish config presets