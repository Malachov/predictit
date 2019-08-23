# predict-it
Library/framework for making predictions. Choose best of 20 models (ARIMA, regressions, LSTM...). Preprocess data and chose optimal parameters of predictions.

## Output
Output is plotly interactive graph or deploying to database.
![Printscreen of output HTML graph](/output_example.png)

## How to use
Download it. Open config.py. Two possible inputs. CSV or Database. Config is quite clear. Setup usually 1 or 0. Finally run main.py. With config.optimize_it it's pretty time consuming. So firs turn optimize_it off. Find best three models and then optimize. Do not try to optimize LSTM...
