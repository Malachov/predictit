# List of what have been done in new versions

## 1.6x - 10/2020

[x] Annoying warnings filtered on import not on library call (and from multiprocessess)
[x] String embedding - One hot or label encoding
[x] Api change! Config renamed because class with capitalize C from config to Config. 'datetime_index' renamed to 'datetime_column' - Old scripts wont work
[x] Data_preprocessing and logging from misc moved into own projects and imported. Projects are called mylogging and mydatapreprocessing. It'necessary to have corresponding version
[x] Various transforms added as derived columns. For example: Difference transform, Rolling window and rolling std transformation, Distance from the mean.
[x] Fast fourier transform as new information.
[x] Short way of using functions with positional argument - predictit.main.predict(np.random.randn(1000). Same with predict_multiple and compare models.
[x] Devil terminology fail fixed(used batch instead of multistep in own models)
[x] New custom models added (ridge regresion and Levenberg-Marquardt)
[x] README.md derived from `__init__.py` for better consistency + Fixes for better help tooltip in IDE with JEDI (Pre-commit githook rewrote to python)

## 1.5x - 09/2020

- [x] New data input. Config Api changed! No data_source anymore. Setup path, url or python data in Config.data. Check configuration for some examples.
- [x] New data input formats - dictionary, list, parquet, json, h5 or url that return data in request
- [x] Data preprocessing functions possible now in dataframe
- [x] Data consolidation simplified - only in dataframe now
- [x] Remove nans changed. Now first columns with nans over threshold removed, then removed or replaced based on Config.
- [x] Data_load and data_consolidation arguments changed. Config instance no necessary now, so functions can be used outside this library.

## 1.4x - 08/2020

- [x] Travis CI building releases on Pypi as well on github.
- [x] Config redefined to class and so enabling intellisense in IDE.

  - API change!!! instead of config['value'] use Config.value. But Config.update({'variable': 'value'}) still works !!!

- [x] Creating inputs defined in visual test
- [x] Csv locale option - separator and decimal for correct data reading.
- [x] Tables remade in tabulate and returned as dataframe.
- [x] Return type 'all' removed (replaced with dataframe).

## v1.3x. - 05/2020

- [x] Config value optimization. Some config variable can be optimized on defined values. For each model the best option will be used
- [x] Plot results of all models versions (optmimized)
- [x] Option whether to evalueate error criterion on preprocessed data or on original data
- [x] Option to sort detailed result table by error or by name
- [x] Compare models redefined. More fair for all types of models, but repeatit = 1, so more samples necessary.

## v1.3x. - 04/2020

- [x] Many new models from sklearn - Decision trees, bagging, Gradient boosting...
- [x] Multiprocessing applied. Two options - 'pool' or 'process'
- [x] Data smoothing - Savitzky-Golay filter in data preprocessing
- [x] Publish script to generate documentation and push to pypi
- [x] Data processed on 'standard' shape [n_samples, n_features] so in same shape as dataframe
- [x] Config print_config function to print config file as help
- [x] Updated config values confirmation. If value misspelled (not in list) then error.
- [x] Detailed results in table (not printed)
- [x] Config how many results to plot and print
- [x] Line time and memory profiling

## v1.2x. - 03/2020

- [x] Validation mode in Config. Results are evaluated on data that was not in train data - used in compare_models.
- [x] Remove nan values - all column option was added
- [x] User colored warnings in misc (not only traceback warnings) and colorized error raising
- [x] Two options / modes of analyze - Originall data and preprocessed data

## v1.1. - 03/2020

- [x] Simple GUI (just config and output)
- [x] Added customizable config presets (fast, normal, optimize)
- [x] Creating inputs in new define_inputs module called from main (just once), not in models (for each model separatedly). It use no lists but numpy stride tricks
- [x] Optimize loop , find best models loop and predict loop joined into one main loop
- [x] Models divided into train and predict functions - Repeat loop is only on predict function
- [x] Redefined models, to use (X, y, x_input) tuple as input
- [x] Config values putted in dictionary [! other way to use it!]
- [x] Basic data postprocessing - Power transformation - Two options 1) On train data (change error criterion) 2) Only on output

## v1.0 - 03/2020 (first no in development version)

- [x] Tensorflow models architecture configurable with arguments - layers and its parameters in list.
- [x] Choose which models to optimize (comment out config models_parameters_limits dictionary)
- [x] Similiar models generalized to one (E.g. AR, ARMA, ARIMA > statsmodels_autoregressive)
- [x] Plot and data preprocessing are made in own module, not in main
- [x] New model similarity based error criterions (not exact point to point comparison) 1) Imported time warping 2) Own sliding window error
- [x] One more level of debug - stop at first warning - add warning exceptions to config to hide outer depracation etc. warnings
- [x] More user friendly warnings (color syntax highlighted) + Error separated from error location
