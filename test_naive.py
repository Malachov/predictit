#%%

import os
cwd = os.getcwd()
os.listdir(cwd)

import predictit.predictit as pre

pre.main.predict()

'''
try:
    predictit.predictit.main.predict()
except Exception as e:
    print(f'Found error: {e}')
'''
