#%%
from predict import predictit

try:
    predictit.main.predict()
except Exception as e:
    print(f'Found error: {e}')
