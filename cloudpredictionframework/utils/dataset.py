import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


def prepare_test_dataframe(dataframe, metric):
    new_df = pd.DataFrame()
    new_df['day_of_month'] = dataframe.timestamp.dt.day
    new_df['day_of_week'] = dataframe.timestamp.dt.dayofweek
    new_df['month'] = dataframe.timestamp.dt.month
    new_df[metric] = dataframe[metric]
    return new_df


def filter_dataframe(dataframe, filter, metric='cpu.usage.average'):
    filtered_df = dataframe.copy()
    for index, row in filtered_df.iterrows():
        filter.update(row['timestamp'], row[metric])
        if filter.get_current_state() == filter.states.overutil_anomaly:
            filtered_df.drop(index, inplace=True)

    filtered_df.reset_index(inplace=True)
    filtered_df.drop(columns=['index'], inplace=True)
    return filtered_df


def split_dataframe(dataframe, ratio=0.9):
    train_size = int(len(dataframe) * ratio)
    train, test = dataframe.iloc[0:train_size], dataframe.iloc[train_size:len(dataframe)]
    return train, test


def create_dataset(x, y, time_steps=1):
    xs, ys = [], []
    for i in range(len(x) - time_steps):
        v = x.iloc[i:(i + time_steps)].values
        xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(xs), np.array(ys)
