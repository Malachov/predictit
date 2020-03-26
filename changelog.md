# List of what have been done in new versions

## v1.1. - 03/2020

### Big Deals

- [x] Simple GUI (just config and output)
- [x] Added customizable config presets (fast, normal, optimize)
- [x] Creating inputs in new define_inputs model called from main (just once), 
        not in models (for each model separatedly).
- [x] Optimize loop , find best models loop and predict loop joined into one main loop
- [x] Models divided into train and predict functions
        - Repeat loop is only on predict function
- [x] Redefined models, to use (X, y, x_input) tuple as input
- [x] Config values putted in dictionary [! other way to use it!]
- [x] Basic data postprocessing - Power transformation (two options
         1) On train data (change error criterion)
         2) Only on output

### Small Deals
- [x] Choose which models to optimize (comment out config models_parameters_limits dictionary)
- [x] New model similarity based error criterions (not exact point to point comparison)
        1) Imported time warping
        2) Own sliding window error
- [x] One more level of debug - stop at first warning - add warning exceptions
- [x] More user friendly warnings (color syntax highlighted) + Error separated from error location 
- [x] Some memory and time profiling in tests