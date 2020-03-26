"""Plot predicted data. Input data are divided on data history and results. Youi can choose between plotly (interactive) and matplotlib (faster)"""

from misc import traceback_warning, _JUPYTER
import pandas as pd
import confidence_interval
from predictit.config import config
import os


def plotit(history, predicted_models, plot_type='plotly', show=1, save=0, save_path='', plot_return=None, bounds='default'):

    import plotly as pl

    if save:
        if not save_path:
            save_path = os.path.normpath(os.path.expanduser('~/Desktop') + '/plot.html')

    best_model_name = list(predicted_models.keys())[0]
    complete_dataframe = history.copy()
    predicted_column_name = history.columns[0]

    if bounds == 'default':
        try:
            lower_bound, upper_bound = confidence_interval.bounds(history, predicts=config['predicts'], confidence=config['confidence'])
        except Exception:
            bounds = False
            traceback_warning("Error in compute confidence interval")

    complete_dataframe['Best prediction'] = None
    complete_dataframe['Lower bound'] = complete_dataframe['Upper bound'] = None

    last_date = history.index[-1]

    if isinstance(last_date, pd._libs.tslibs.timestamps.Timestamp):
        date_index = pd.date_range(start=last_date, periods=config['predicts'] + 1, freq=config['freq'])[1:]
        date_index = pd.to_datetime(date_index)

    else:
        date_index = list(range(last_date + 1, last_date + config['predicts'] + 1))

    results = pd.DataFrame(data={'Lower bound': lower_bound, 'Upper bound': upper_bound}, index=date_index) if bounds else pd.DataFrame(index=date_index)

    for i, j in predicted_models.items():
        if 'predictions' in j:
            results[i] = j['predictions']
            complete_dataframe[i] = None

    last_value = float(history.iloc[-1])

    complete_dataframe = pd.concat([complete_dataframe, results], sort=False)
    complete_dataframe.iloc[-config['predicts'] - 1] = last_value

    if plot_type == 'plotly':

        import plotly as pl

        if bounds:
            upper_bound = pl.graph_objs.Scatter(
                name = 'Upper Bound',
                x = complete_dataframe.index,
                y = complete_dataframe['Upper bound'],
                #mode = 'lines',
                marker = dict(color = '#444'),
                line = dict(width = 0),
                fillcolor = 'rgba(68, 68, 68, 0.3)',
                fill = 'tonexty')

        best_prediction = pl.graph_objs.Scatter(
            name = '1. {}'.format(best_model_name),
            x = complete_dataframe.index,
            y = complete_dataframe[best_model_name],
            #mode = 'lines',
            line = dict(color='rgb(51, 19, 10)', width=4),
            fillcolor = 'rgba(68, 68, 68, 0.3)',
            fill = 'tonexty' if bounds else None)

        if bounds:
            lower_bound = pl.graph_objs.Scatter(
                name ='Lower Bound',
                x = complete_dataframe.index,
                y = complete_dataframe['Lower bound'],
                marker = dict(color='#444'),
                line = dict(width=0))

        history = pl.graph_objs.Scatter(
            name = history.columns[0],
            x = complete_dataframe.index,
            y = complete_dataframe[predicted_column_name],
            line = dict(color='rgb(31, 119, 180)', width=3),
            fillcolor = 'rgba(68, 68, 68, 0.3)',
            fill = None)


        graph_data = [lower_bound, best_prediction, upper_bound, history] if bounds else [best_prediction, history]

        fig = pl.graph_objs.Figure(data=graph_data)

        for i in predicted_models:
            if i != best_model_name and i in complete_dataframe:
                fig.add_trace(pl.graph_objs.Scatter(
                    x = complete_dataframe.index,
                    y = complete_dataframe[i],
                    name = f"{predicted_models[i]['order']}. {i}")
                )

        fig.update_layout(
            yaxis = dict(title='Values'),
            title = {   'text': config['plot_name'],
                        'y': 0.9,  # if jupyter else 0.95,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
            titlefont = {'size': 28},
            showlegend = False,
            paper_bgcolor = '#d9f0e8'
        )

        if show:
            fig.show()

        if save:
            fig.write_html(save_path)

        if plot_return == 'div':
            fig.update_layout(
                title = None,
                height = 290,
                margin = {
                    'b': 35,
                    't': 35,
                    'pad': 4},
            )

            div = pl.offline.plot(fig, include_plotlyjs=False, output_type='div')

            return div

    if plot_type == 'matplotlib':
        if _JUPYTER:
            get_ipython().run_line_magic('matplotlib', 'inline')

        import matplotlib.pyplot as plt
        plt.rcParams["figure.figsize"] = (12, 8)

        complete_dataframe.plot()
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
        if save:
            plt.savefig(save_path)
        if show:
            plt.show()
