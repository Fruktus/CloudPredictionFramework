from collections import defaultdict
from math import sqrt
from statistics import stdev, mean
from sys import maxsize

import numpy as np
from sklearn.metrics import mean_squared_error


def analyze_batch_result(batch_result, full=False):
    filter_results = defaultdict(lambda: defaultdict(list))
    for df_run in batch_result:
        for filter_run in df_run:
            for metric in filter_run['result'].keys():
                filter_results[filter_run['filter']][metric].append(filter_run['result'][metric])
    if not full:
        for adfilter in filter_results.keys():
            for metric in filter_results[adfilter].copy():
                filter_results[adfilter][metric + '_stdev'] = stdev(filter_results[adfilter][metric])
                filter_results[adfilter][metric] = mean(filter_results[adfilter][metric])
    return filter_results


def analyze_experiment(ex_result: dict):
    # calculate extrema: overestimation, underestimation, crosscorrelation, average positive/negative/total difference,
    sample_len = len(ex_result['true'])

    total_difference = 0

    overestimate_difference = 0
    total_overestimates = 0

    underestimate_difference = 0
    total_underestimates = 0

    highest_overestimate = 0
    lowest_underestimate = maxsize

    correlation = float(np.correlate(ex_result['true'], ex_result['prediction'].flatten()))

    for sample in zip(ex_result['prediction'].flatten(), ex_result['true']):
        diff = sample[0] - sample[1]

        total_difference += abs(diff)

        if diff > 0:
            overestimate_difference += diff
            total_overestimates += 1
            highest_overestimate = diff if diff > highest_overestimate else highest_overestimate
        if diff < 0:
            underestimate_difference += diff
            total_underestimates += 1
            lowest_underestimate = diff if diff < lowest_underestimate else lowest_underestimate

    return {'avg_total_diff': total_difference/sample_len,
            'avg_overestimate_diff': overestimate_difference/total_overestimates,
            'avg_underestimate_diff': underestimate_difference/total_underestimates,
            'highest_overestimate': highest_overestimate,
            'lowest_underestimate': lowest_underestimate,
            'total_overestimates': total_overestimates,
            'total_underestimates': total_underestimates,
            'correlation': correlation,
            'RMSE': sqrt(mean_squared_error(ex_result['true'], ex_result['prediction'].flatten()))}