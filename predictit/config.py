
# CWD - absolutní adresa modulu predikcí - není nutné pokud máte otevřenou celou složku v IDE
from . import models

cwd = r'C:\Users\daniel.malachov\Desktop\Diplomka'

# Zdroj dat v případě, že se nepoužívají data testovací
# Buď 'sql', nebo 'csv'
data_source = 'csv'

server = '.'
database = 'FK'  # Název databáze
index_col = 'DateBK'  # Název sloupce s datumem
freqs = 'D'

# Adresa CSV pro predikci pokud nebudou použita testovací data - První řádek obsahuje názvy sloupců, první sloupec datum
csv_adress = r'E:\VSCODE\Diplomka\test_data\daily-minimum-temperatures.csv'  # Adresa csv včetně názvu a přípony

predicted_columns_names = 'Temp'  #['SumNumber', 'SumDuration']  # Název sloupce jehož hodnota má být predikována


date_index = 0
predicts = 7  # Počet predikovaných hodnot - defaultně 7

datalength = 1000  # Posledních N prvků, které budou použity

data_transform = None  #'difference'  # 'difference' or None - Transformuje data na rozdíl mezi dvěma hodnotami
data_smooth = 0  # Uhladí data, čímž odstraní 'bílý šum' a lépe tak odolává overfittingu hodnota udáva míru uhlazení

# Výpočet je opakován několikrát na zkrácených datech, aby se vyloučila náhoda úspěchu modelu
repeatit = 2  # repeatit je počet opakování
other_columns = 0  # Pokud nula, tak k výpočtubude použit pouze predikovaný sloupec
lengths = 0  # Data rozdělí na různě dlouhé úseky a vybere ten nejlepší
confidence = 0.8  # Oblast nejistoty ve finálním grafu - vyšší hodnota znamená užší oblast - maximum 1
remove_outliers = 0  # Odstraní hodnoty odlehlé od průměru. Hodnota uvádí limit, nad který budou data smazána - jde o násobek standardní směrodatné odchylky
criterion = 'mape'  # 'mape' or 'rmse'
optimizeit = 0  # Najde optimální parametry modelů
compareit = 10  # S kolika modely bude výsledek srovnáván
debug = 1  # Debug - vypíše podrobné výsledky všech predikcí
last_row = 0  # Pokud 0, vymaže poslední nekompletní řádek
analyzeit = 0  # Analyzuje vstupní data - vypočíta autokorelaci
evaluate_test_data = 1  # Jestli bude model hodnocen podle testovacích dat a nebo pouze predikovaných dat
correlation_threshold = 0.2
optimizeit_details = 2  # 1 vypíše nejlepší parametry modelu, 2 vypíše každé zlepšení a parametry
optimizeit_limit = 5 # Jak dlouho může trvat výpočet jednoho parametru v sekundách
optimizeit_final = 0  # Znovu optimalizuje nejlepší model
plot = 1  # Pokud 1, vykreslí grafy výsledků nejlepší predikce
plotallmodels = 0  # Vykreslí všechny predikce všech modelů
piclkeit = 0  # Uloží testovací data na disk v serializované formě, čímž zrychlí načítání - nutno vypnout na nulu, aby se data nenačítala pokaždé
already_trained = 0  # Výpočetně náročné modely jako LSTM načíst z disku
saveit = 0
standardizeit = 0  # Standardizuje data od -1 do 1
normalizeit = 0  # Normalizuje data na směrodatnou odchylku 1 a průměr 0
compare = 1  # Zda budou data výsledků porovnávána s ostatními modely

# Data ned kterými bude testována predikce
data_all_pickle = {
#                    "Daily minimum temperatures": 'data0',
                    "Sin": 'data1'
#                    "Sign": 'data2',
#                    "Dynamic system": 'data3',
#                    "Reálná data klapky": 'data4'
                }

# Data pro finální predikci testovacích dat
data_name_for_predicts = "Sin"

## Modely, které budou použity k testování
# Podrobná dokumentace k modulu je přímo v models (__init__)
# Stačí Go to definition u models v prvním řádku from models import

used_models = {

            "AR (Autoregression)": models.ar,
            "ARMA": models.arma,
            "ARIMA (Autoregression integrated moving average)": models.arima,
#            "SARIMAX (Seasonal ARIMA)": models.sarima,

            "Autoregressive Linear neural unit": models.autoreg_LNU,
#            "Linear neural unit with weigths predict": models.autoreg_LNU_withwpred,
            "Conjugate gradient": models.cg,

#            "Extreme learning machine": models.elm,
#            "Gen Extreme learning machine": models.elm_gen,

#            "LSTM": models.lstm,
#            "Bidirectional LSTM": models.lstm_bidirectional,
#            "LSTM batch": models.lstm_batch,

            "Sklearn universal": models.sklearn_universal,

            "Bayes Ridge Regression": models.regression_bayes_ridge,
            "Hubber regression": models.regression_hubber,
            "Lasso Regression": models.regression_lasso,
            "Linear regression": models.regression_linear,
            "Ridge regression": models.regression_ridge,
            "Ridge regressionCV": models.regression_ridge_CV,

#            "Compare with average": models.compare_with_average
           }


'''
# For testing
used_models = {
#    "AR (Autoregression)": models.ar,
#    "ARMA": models.arma,
    "ARIMA (Autoregression integrated moving average)": models.arima
#    "SARIMAX (Seasonal ARIMA)": models.sarima,    
}
'''

# Pozor, jména modelů musí být identická s názvy v modelech
# Kolik regresivních členů - lagů, bude uvažováno
constant = None
n_steps_in = 100
output_shape = 'batch'
models_parameters = {

        #TODO
        "ETS": {"plot": 0},


        "AR (Autoregression)": {"plot": 0, 'method': 'cmle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs'},
        "ARMA": {"plot": 0, "p": 3, "q": 0, 'method': 'mle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs', 'forecast_type': 'in_sample'},
        "ARIMA (Autoregression integrated moving average)": {"p": 3, "d": 0, "q": 0, "plot": 0, 'method': 'css', 'ic': 'aic', 'trend': 'nc', 'solver': 'nm', 'forecast_type': 'out_of_sample'},
        "SARIMAX (Seasonal ARIMA)": {"plot": 0, "p": 4, "d": 0, "q": 0, "pp": 1, "dd": 0, "qq": 0, "season": 12, "method": "lbfgs", "trend": 'nc', "enforce_invertibility": False, "enforce_stationarity": False, "verbose": 0},


       # "ARIMA (Autoregression integrated moving average)": {"p": [1, maxorder], "d": [0,1], "q": order, 'method': ['css-mle', 'mle', 'css'], 'trend': ['c', 'nc'], 'solver': ['lbfgs', 'bfgs', 'newton', 'nm', 'cg'], 'forecast_type': ['in_sample', 'out_of_sample']},




        "Autoregressive Linear neural unit": {"plot": 0, "lags": n_steps_in, "mi": 1, "minormit": 0, "tlumenimi": 1},
        "Linear neural unit with weigths predict": {"plot": 0, "lags": n_steps_in, "mi": 1, "minormit": 0, "tlumenimi": 1},
        "Conjugate gradient": {"n_steps_in": 30, "epochs": 5, "constant": 1, "other_columns_lenght": None, "constant": None},

        "Extreme learning machine": {"n_steps_in": 20, "output_shape": 'one_step', "other_columns_lenght": None, "constant": None, "n_hidden": 20, "alpha": 0.3, "rbf_width": 0, "activation_func": 'selu'},
        "Gen Extreme learning machine": {"n_steps_in": 20, "output_shape": 'one_step', "other_columns_lenght": None, "constant": None, "alpha": 0.5},

        #dodelat lstm
        "LSTM": {"n_steps_in": 50, "save": saveit, "already_trained": 0, "epochs": 70, "units":50, "optimizer":'adam', "loss":'mse', "verbose": 1, "activation": 'relu', "timedistributed": 0, "metrics": ['mape']},
        "LSTM batch": {"n_steps_in": n_steps_in, "n_features": 1, "epochs": 70, "units": 50, "optimizer":'adam', "loss":'mse', "verbose": 1, 'dropout': 0, "activation":'relu'},
        "Bidirectional LSTM": {"n_steps_in": n_steps_in, "epochs": 70, "units": 50, "optimizer":'adam', "loss":'mse', "verbose": 0},

        "Sklearn universal": {"n_steps_in": n_steps_in, "output_shape": "one_step", "model": models.default_regressor, "constant": None},

        "Bayes Ridge Regression": {"n_steps_in": n_steps_in, "output_shape": output_shape, "other_columns_lenght": None, "constant": None, "alpha_1": 1.e-6, "alpha_2": 1.e-6, "lambda_1": 1.e-6, "lambda_2": 1.e-6},
        "Hubber regression": {"n_steps_in": n_steps_in, "output_shape": output_shape, "other_columns_lenght": None, "constant": None, "epsilon": 1.35, "alpha": 0.0001},
        "Lasso Regression": {"n_steps_in": n_steps_in, "output_shape": output_shape, "other_columns_lenght": None, "constant": None, "alpha": 0.6},
        "Linear regression": {"n_steps_in": n_steps_in, "output_shape": output_shape, "other_columns_lenght": None, "constant": None, },
        "Ridge regression": {"n_steps_in": n_steps_in, "output_shape": output_shape, "other_columns_lenght": None, "constant": None, "alpha": 1, "solver": 'auto'},
        "Ridge regressionCV": {"n_steps_in": n_steps_in, "output_shape": output_shape, "other_columns_lenght": None, "constant": None, "alphas": [1e-3, 1e-2, 1e-1, 1], "gcv_mode": 'auto'},

        }

# Hraniční hodnoty pro optimalizaci
# Pokud jsou povinná celá čísla, použijte například 2, pokud jsou možná desetinna, pište 2.0

steps = [2, 200]
alpha = [0.0, 1.0]
epochs = [2, 100]
units = [1, 100]
order = [0, 5]

maxorder = 6

fragments = 4
iterations = 2

fragments_final = 2 * fragments
iterations_final = 2 * iterations


# !! Vše co je zde musí být i v inicializačních parametrech výše, jinak error
# LSTM modely při tunování parametrů nenačítat z PC !!
models_parameters_limits = { 
        "AR (Autoregression)": {"ic": ['aic', 'bic', 'hqic', 't-stat'], "trend": ['c', 'nc'], "solver": ['bfgs', 'newton', 'nm', 'cg']},

        "ARMA": {"p": [1, maxorder], "q": order, 'method': ['css-mle', 'mle','css'], 'trend': ['c', 'nc'], 'solver': ['lbfgs', 'bfgs', 'newton', 'nm', 'cg', 'ncg', 'powell'], 'forecast_type': ['in_sample', 'out_of_sample']},
       # "ARIMA (Autoregression integrated moving average)": {"p": [1, maxorder], "d": [0,1], "q": order, 'method': ['css-mle', 'mle', 'css'], 'trend': ['c', 'nc'], 'solver': ['lbfgs', 'bfgs', 'newton', 'nm', 'cg'], 'forecast_type': ['in_sample', 'out_of_sample']},
        "ARIMA (Autoregression integrated moving average)": {"p": [1, maxorder], "d": [0,1], "q": order, 'method': ['css'], 'trend': ['c', 'nc'], 'solver': ['lbfgs', 'bfgs', 'newton', 'nm', 'cg'], 'forecast_type': ['in_sample', 'out_of_sample']},
        "SARIMAX (Seasonal ARIMA)": {"p": [1, maxorder], "d": order, "q": order, "pp": order, "dd": order, "qq": order, "season": order, "method": ['lbfgs', 'bfgs', 'newton', 'nm', 'cg', 'ncg', 'powell'], "trend" : ['n', 'c', 't', 'ct'], "enforce_invertibility": [True, False], "enforce_stationarity": [True, False], 'forecast_type': ['in_sample', 'out_of_sample']},

        "Autoregressive Linear neural unit": {"lags": steps, "mi": [1.0, 10.0], "minormit": [0, 1], "tlumenimi": [0.0, 100.0]},
        "Linear neural unit with weigths predict": {"lags": steps, "mi": [1.0, 10.0], "minormit": [0, 1], "tlumenimi": [0.0, 100.0]},
        "Conjugate gradient": {"n_steps_in": steps, "epochs": epochs, "constant": [None, 1], "other_columns_lenght": [None, steps[1]]},

        "Extreme learning machine": {"n_steps_in": steps, "n_hidden": [2, 300], "output_shape": ['batch', 'one_step'], "constant": [None, 1], "other_columns_lenght": [None, steps[1]], "alpha": alpha, "rbf_width": [0.0, 10.0], "activation_func": models.activations},
        "Gen Extreme learning machine": {"n_steps_in": steps, "alpha": alpha, "output_shape": ['batch', 'one_step'], "constant": [None, 1], "other_columns_lenght": [None, steps[1]]},

        "Sklearn universal": {"model": models.regressors, "n_steps_in": steps, "output_shape": ['batch', 'one_step'], "constant": [None, 1]},

        "Bayes Ridge Regression": {"n_steps_in": steps, "output_shape": ['batch', 'one_step'], "constant": [None, 1], "other_columns_lenght": [None, steps[1]], "alpha_1":[0.1e-6, 3e-6], "alpha_2":[0.1e-6, 3e-6], "lambda_1":[0.1e-6, 3e-6], "lambda_2":[0.1e-7, 3e-6]},
        "Lasso Regression": {"n_steps_in": steps, "output_shape": ['batch', 'one_step'], "constant": [None, 1], "other_columns_lenght": [None, steps[1]], "alpha": alpha},
        "Linear regression": {"n_steps_in": steps, "output_shape": ['batch', 'one_step'], "constant": [None, 1], "other_columns_lenght": [None, steps[1]]},
        "Hubber regression": {"n_steps_in": steps, "output_shape": ['batch', 'one_step'], "constant": [None, 1], "other_columns_lenght": [None, steps[1]], "epsilon": [1.01, 5.0], "alpha": alpha},
        "Ridge regression": {"n_steps_in": steps, "output_shape": ['batch', 'one_step'], "constant": [None, 1], "other_columns_lenght": [None, steps[1]], "alpha": alpha, "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']},
        "Ridge regressionCV": {"n_steps_in": steps, "output_shape": ['batch', 'one_step'], "constant": [None, 1], "other_columns_lenght": [None, steps[1]], "gcv_mode" : ['auto', 'svd', 'eigen']},

}
