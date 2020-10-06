#%%

### Not implemented yet... have to find good datasets

import sys
import pandas as pd
from pathlib import Path
import inspect
import os

sys.path.insert(0, Path(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)).parents[1].as_posix())

import predictit
Config = predictit.configuration.Config

# 
Config.data_all = {
    "Monthly beer production": ("https://www.kaggle.com/shenba/time-series-datasets?resource=download/LphTPFq528ElkH1LARMb%2Fversions%2FhmEYjUTESVU5R2UGyga3%2Ffiles%2Fmonthly-beer-production-in-austr.csv&downloadHash=55484c1bf336cce44929bace0672fe8c4d08a1b3c5e64a58c810b27a7217b789", 1),
#     'Temp': "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
#     'Sales': "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv",
}

Config.request_datatype_suffix = 'csv'
Config.return_type = "models_error_criterion"

result = predictit.main.compare_models()

# models_kpi = pd.read_csv("models_kpi.csv")
# models_kpi.append(result['Models table']).to_csv()

#%%
import requests
loaded_data = requests.get("https://www.kaggle.com/shenba/time-series-datasets?resource=download/LphTPFq528ElkH1LARMb%2Fversions%2FhmEYjUTESVU5R2UGyga3%2Ffiles%2Fmonthly-beer-production-in-austr.csv&downloadHash=55484c1bf336cce44929bace0672fe8c4d08a1b3c5e64a58c810b27a7217b789").content
import requests
from contextlib import closing
import csv

# url = https://www.kaggle.com/shenba/time-series-datasets?resource=download/LphTPFq528ElkH1LARMb%2Fversions%2FhmEYjUTESVU5R2UGyga3%2Ffiles%2Fmonthly-beer-production-in-austr.csv&downloadHash=55484c1bf336cce44929bace0672fe8c4d08a1b3c5e64a58c810b27a7217b789

import csv
import requests

response = requests.get('https://www.kaggle.com/shenba/time-series-datasets?resource=download/LphTPFq528ElkH1LARMb%2Fversions%2FhmEYjUTESVU5R2UGyga3%2Ffiles%2Fmonthly-beer-production-in-austr.csv&downloadHash=55484c1bf336cce44929bace0672fe8c4d08a1b3c5e64a58c810b27a7217b789').content
reader = csv.DictReader(response)
for record in reader:
    print(record)
