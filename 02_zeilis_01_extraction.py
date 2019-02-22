'''
Transforming extracted data for application of Zeilis et al.'s algorithm
Usage: python3 02_zeilis_01_extraction.py
'''
import os

import numpy as np
import pandas as pd

import constants

# Constants
NUMBER_OF_DIMENSIONS = constants.NUMBER_OF_DIMENSIONS
RESAMPLE_TIME_WINDOW = "M"
DIMENSION_NAMES = constants.DIMENSION_NAMES
EVENTS_TO_BURN_IN = constants.EVENTS_TO_BURN_IN

def __normalization_function(value_of_list, centering_value, min_of_scale, max_of_scale):
    return (value_of_list - centering_value) / (max_of_scale - min_of_scale)

def __read_dataset(some_dataset):
    path_to_current_dataset = constants.PATH_TO_DESTINATION_DATASET + some_dataset + "/" + some_dataset
    timestamp_list = [np.genfromtxt(path_to_current_dataset + dim_name + ".csv", 
                                    dtype=np.float, delimiter=",")[EVENTS_TO_BURN_IN:] 
                      for dim_name in DIMENSION_NAMES]
    begin_and_end_timestamp_list = [{"first": timestamp_list_dim[0], 
                                     "last": timestamp_list_dim[-1]}
                                    for timestamp_list_dim in timestamp_list]
    first_timestamp = min([begin_and_end_timestamps["first"]
                           for begin_and_end_timestamps in begin_and_end_timestamp_list])
    normalized_timestamp_list = [__normalization_function(timestamp_list[dim_i], 
                                                          first_timestamp, 0, 1) 
                                 for dim_i in range(NUMBER_OF_DIMENSIONS)]
    event_counts = [pd.Series(np.ones(len(timestamp_list[a_dim])),
                              index=list(map(pd.datetime.fromtimestamp, 
                                             timestamp_list[a_dim]*60**2))) \
                      .resample(RESAMPLE_TIME_WINDOW).sum() 
                    for a_dim, _ in enumerate(DIMENSION_NAMES)]
    return tuple(event_counts)

# prepare time series csv
def __prepare_time_series_csv(selected_datasets, age_in_days, mode="aggregated"):
    ts_df = {"Name": [], "Month": [], "Activity": []}
    for current_dataset in selected_datasets:
        try:
            qf, qnf, af, anf = __read_dataset(current_dataset)
        except IndexError:
            print(current_dataset)
            continue
        whole_dataset = (qf + af + qnf + anf).fillna(0)
        # ignore 0s in the beginning, if the sequence starts like that
        whole_dataset = whole_dataset.iloc[(whole_dataset.tolist().index(whole_dataset[whole_dataset != 0][0])):]
        print(current_dataset)
        whole_dataset_filtered = whole_dataset[(whole_dataset.index - whole_dataset.index[0]).days <= age_in_days].tolist()
        whole_dataset_len = min(len(whole_dataset), len(whole_dataset_filtered))
        ts_df["Name"] += np.repeat(current_dataset, whole_dataset_len).tolist()  # np.repeat(current_dataset, age_in_days / 30).tolist()
        ts_df["Month"] += np.arange(whole_dataset_len).tolist()  # np.arange(age_in_days / 30).tolist()
        ts_df["Activity"] += whole_dataset_filtered
    pd.DataFrame(ts_df).to_csv("se_ts.csv")

all_datasets = os.listdir(constants.PATH_TO_DESTINATION_DATASET)
too_small_datasets = ["sitecore", "esperanto", "ai", "monero"]
selected_datasets = list(set(all_datasets) - set(too_small_datasets))
age_in_days = 36 * 30  # a little less than 3 years
__prepare_time_series_csv(selected_datasets, age_in_days)
