# List of what could be done

## Now working on
- [x] New criterion model - with lagged values / autocorrelation for anomaly prediction


## Big Deals

- [ ] !!! Split all models on two functions - First train (or update) model - return trained model. Second only make predictions

    - Train must not be in repeat loop


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
- [ ] Serialize transformations as other columns
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
- [ ] Ensamble learning - combine best models results to one result
- [ ] Fast optimization model - (optimize each parameter separately and just combine)
- [ ] Add scientific documentation into docstrings (equations and diagrams)
- [ ] Add exogenous parametrs where possible
- [ ] Finish TODOS in GUI
- [ ] Do segmentation in time series (Matrix profile method) - 1) Use in analyze, 2) Detect anomalies
- [] Numba optimization

## Deals

- [ ] Separate Plot to function + Plot_all during learning option - with real validate data for comparison
- [ ] Add evaluation methods to visual test
- [ ] Config presets - Standard, Fast, Optimize 
- Data preprocessing
    - [ ] Binning inputs
    - [ ] Change data_prep to operate not on lists but arrays. Use numpy strides tricks
    - [ ] In remove_outliers function do not remove it, but interpolate by neighbors
- [ ] Add residuals matrix from results_matrix
- [ ] Remove sys.path.insert in main and do imports better way
- [ ] Translate and finish models __init__ docstrings
- [ ] Remove tensorflowit from models __init__ to config tesnsorflowit=0, with other computationaly hard models 
- [ ] Add plotly box plot for residuals of models in main.py (https://plot.ly/python/box-plots/), and box plot for error criterion (data lengths and for repetitions) on compare_models
- Improve / edit existing models
    - [ ] Repair Sarimax model
    - [ ] For models with significant results diversity (ELM...) use optional bagging (mean of more results)
    - [ ] Add auto arima models from new libraries
    - [ ] Join Autoreg_LNU and Autoreg_LNU_with.. into one model. Number of learning rates to argument
    - [ ] Tensorflow MLP batch into normal as parametr
    - [ ] Conjugate gradient and autoreg LNU for multiple column data
    - [ ] LSTM bidirectional and batch as parameter to normal LSTM
    - [ ] In autoreg add more epochs (learn not only once on each vector)
    - [ ] Add stochastig gradient method to autoreg (adam or adagrad)
- [ ] Ensure for appropriate dtype in calculations - Optional? (64 best? performance vs. error)
- [ ] Make waterfall diagram of spent time - preprocess - train - evaluation, plot - make progress bar for GUI