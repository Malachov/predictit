"""Plot predicted data. Input data are divided on data history and results. Youi can choose between plotly (interactive) and matplotlib (faster)"""

from predictit import misc
from predictit.config import config
import os


def plotit(complete_dataframe, plot_type='plotly', show=1, save=0, save_path='', plot_return=None, bounds='default', predicted_column_name='', best_model_name=''):


    if save:
        if not save_path:
            save_path = os.path.normpath(os.path.expanduser('~/Desktop') + '/plot.html')

    if plot_type == 'plotly':

        import plotly as pl

        complete_dataframe = complete_dataframe.copy()
        graph_data = []

        bounds = 1 if 'Upper Bound' in complete_dataframe and 'Lower Bound' in complete_dataframe else 0

        if bounds:
            upper_bound = pl.graph_objs.Scatter(
                name='Upper Bound',
                x=complete_dataframe.index,
                y=complete_dataframe['Upper bound'],
                #mode='lines',
                marker=dict(color='#444'),
                line=dict(width=0),
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty')

            complete_dataframe.drop('Upper Bound', axis=1, inplace=True)
            graph_data.append(upper_bound)

        if best_model_name in complete_dataframe:
            best_prediction = pl.graph_objs.Scatter(
                name=f'1. {best_model_name}',
                x=complete_dataframe.index,
                y=complete_dataframe[best_model_name],
                #mode='lines',
                line=dict(color='rgb(51, 19, 10)', width=4),
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty' if bounds else None)

            complete_dataframe.drop(best_model_name, axis=1, inplace=True)
            graph_data.append(best_prediction)

        if bounds:
            lower_bound = pl.graph_objs.Scatter(
                name='Lower Bound',
                x=complete_dataframe.index,
                y=complete_dataframe['Lower bound'],
                marker=dict(color='#444'),
                line=dict(width=0))

            complete_dataframe.drop('Lower Bound', axis=1, inplace=True)
            graph_data.append(lower_bound)

        if predicted_column_name in complete_dataframe:

            history_ax = pl.graph_objs.Scatter(
                name=predicted_column_name,
                x=complete_dataframe.index,
                y=complete_dataframe[predicted_column_name],
                line=dict(color='rgb(31, 119, 180)', width=3),
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill=None)

            complete_dataframe.drop(predicted_column_name, axis=1, inplace=True)
            graph_data.append(history_ax)

        fig = pl.graph_objs.Figure(data=graph_data)

        for i in complete_dataframe.columns:
            if i != best_model_name:
                fig.add_trace(pl.graph_objs.Scatter(
                    x=complete_dataframe.index,
                    y=complete_dataframe[i],
                    name=i)
                )

        fig.layout.update(
            yaxis=dict(title='Values'),
            title={'text': config['plot_name'],
                   'y': 0.9,  # if jupyter else 0.95,
                   'x': 0.5,
                   'xanchor': 'center',
                   'yanchor': 'top'},
            titlefont={'size': 28},
            showlegend=False,
            paper_bgcolor='#d9f0e8'
        )

        if show:
            fig.show()

        if save:
            fig.write_html(save_path)

        if plot_return == 'div':

            fig.layout.update(
                title=None,
                height=290,
                margin={
                    'b': 35,
                    't': 35,
                    'pad': 4},
            )

            div = pl.offline.plot(fig, include_plotlyjs=False, output_type='div')

            return div


    if plot_type == 'matplotlib':
        if misc._JUPYTER:
            get_ipython().run_line_magic('matplotlib', 'inline')

        import matplotlib.pyplot as plt
        plt.rcParams["figure.figsize"] = (12, 8)

        complete_dataframe.plot()
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
        if save:
            plt.savefig(save_path)
        if show:
            plt.show()
