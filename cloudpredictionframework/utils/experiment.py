import traceback
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import plotly.graph_objects as go
from tqdm import tqdm

from cloudpredictionframework.usage_prediction.networks.lstm_2layer import LSTM2Layer
from cloudpredictionframework.utils.dataset import filter_dataframe, split_dataframe, create_dataset, prepare_test_dataframe
from cloudpredictionframework.utils.analysis import analyze_experiment

pd.options.mode.chained_assignment = None


def run_experiment(dataframe, network, adfilter=None, metric='cpu.usage.average', show_plot=False,
                   show_tresholds=False, mark_anomalies=False):

    filtered_df = filter_dataframe(dataframe, adfilter, metric) if adfilter else dataframe
    base_filtered_df = prepare_test_dataframe(filtered_df, metric)

    sc: RobustScaler = RobustScaler().fit(base_filtered_df[[metric]])

    train, test = split_dataframe(base_filtered_df)

    train[metric] = sc.transform(train[[metric]])
    test[metric] = sc.transform(test[[metric]])

    x_train, y_train = create_dataset(train, train[metric], time_steps=1)

    network.fit_model(x_train, y_train, verbose=False)

    y_pred = sc.inverse_transform(network.predict(test))

    fig = visualize_result(adfilter, dataframe, mark_anomalies, metric, show_tresholds, y_pred)

    if show_plot:
        fig.show()

    return {'timestamp': dataframe['timestamp'], 'true': dataframe[metric],
            'prediction': y_pred, 'plot': fig}


def visualize_result(adfilter, dataframe, mark_anomalies, metric, show_tresholds, y_pred):
    fig = go.Figure()
    fig.add_scatter(x=dataframe['timestamp'],
                    y=dataframe[metric],
                    name="Actual resource consumption",
                    mode='lines',
                    line_width=1,
                    line=dict(color='black'))
    fig.add_scatter(x=dataframe['timestamp'],
                    y=y_pred.flatten(), name="Predicted resource consumption",
                    mode='lines',
                    line=dict(color='black', dash='dot'))
    if adfilter and show_tresholds:
        fig.add_trace(go.Scatter(x=adfilter.get_tresholds()['timestamp'],
                                 y=adfilter.get_tresholds()['upper_treshold'],
                                 mode='lines',
                                 fill=None,
                                 name='Anomaly Detection Treshold',
                                 line_color='black',
                                 line_width=0.5,
                                 showlegend=False,
                                 fillcolor='rgba(0, 0, 0, 0.1)'))
        fig.add_trace(go.Scatter(x=adfilter.get_tresholds()['timestamp'],
                                 y=adfilter.get_tresholds()['lower_treshold'],
                                 mode='lines',
                                 fill='tonexty',
                                 name='Anomaly Detection Treshold',
                                 line_color='black',
                                 line_width=0.5,
                                 fillcolor='rgba(0, 0, 0, 0.1)'))
    if adfilter and mark_anomalies:
        fig.add_scatter(x=adfilter.get_anomaly_overutil()['timestamp'],
                        y=adfilter.get_anomaly_overutil()['value'], name="Anomaly",
                        mode='markers',
                        marker={'color': 'black'},
                        marker_symbol='square-open-dot')
    title = str(adfilter) if adfilter else 'None'
    fig.update_layout(
        title='Filter = ' + title,
        xaxis_title='Date',
        yaxis_title=metric,
        yaxis_range=[0, 100]
    )

    return fig


def run_batch_experiment(dataframes, filters, verbose=False, show_progress=True, **kwargs):
    results = []
    dataframes = tqdm(dataframes) if show_progress else dataframes

    for df in dataframes:
        try:
            result = run_experiment(df, LSTM2Layer(input_shape=(1, 4)), **kwargs)
            if verbose:
                print('experiment results: ', analyze_experiment(result))
            tmp_res = [{'filter': 'None', 'result': analyze_experiment(result)}]

            for adfilter in filters:
                result = run_experiment(df, LSTM2Layer(input_shape=(1, 4)), adfilter=deepcopy(adfilter), **kwargs)
                if verbose:
                    print('experiment results: ', analyze_experiment(result))
                tmp_res.append({'filter': str(adfilter), 'result': analyze_experiment(result)})
            results.append(tmp_res)
        except Exception:
            print(traceback.format_exc())
    return results


def run_prediction_feedback(dataframe, network, adfilter=None, metric='cpu.usage.average', feedback_len=10,
                            show_plot=True):

    filtered_df = filter_dataframe(dataframe, adfilter, metric) if adfilter else dataframe

    base_filtered_df = prepare_test_dataframe(filtered_df, metric)

    sc: RobustScaler = RobustScaler().fit(base_filtered_df[[metric]])

    train, test = split_dataframe(base_filtered_df)

    train[metric] = sc.transform(train[[metric]])
    test[metric] = sc.transform(test[[metric]])

    x_train, y_train = create_dataset(train, train[metric], time_steps=1)

    network.fit_model(x_train, y_train, verbose=False)

    data_pred = test.reshape((test.shape[0], 1, test.shape[1]))

    for index in range(len(data_pred)):
        if index > feedback_len:
            data_pred[index][0][3] = network.predict(np.array([data_pred[index-1]]))
        else:
            network.predict(np.array([data_pred[index]]))

    y_pred = sc.inverse_transform(data_pred[:, :, 3])
    # data_pred contains unscaled raw predictions, need to scale them back

    fig = go.Figure()
    fig.add_scatter(x=dataframe['timestamp'], y=dataframe['cpu.usage.average'],
                    name="Actual resource consumption", mode='lines',
                    line=dict(color='black'))
    fig.add_scatter(x=dataframe['timestamp'], y=y_pred.flatten(),
                    name="Predicted resource consumption", mode='lines',
                    line=dict(color='black', dash='dot'))

    title = str(adfilter) if adfilter else 'None'
    fig.update_layout(
        title='Filter = ' + title,
        xaxis_title='Date',
        yaxis_title=metric,
        yaxis_range=[0, 100]
    )

    if show_plot:
        fig.show()
    return fig
