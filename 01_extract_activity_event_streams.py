'''
Transforming StackExchange data, as collected via data_preparation/s01_extract_log_stackexchange.py of https://github.com/simonwalk/ActivityDynamics, for further processing
Usage: python3 01_extract_activity_event_streams.py
'''
import multiprocessing
import os

import numpy as np
import pandas as pd

import constants

DATASET_LIST = os.listdir(constants.PATH_TO_SOURCE_DATASETS + "answers")
TIME_UNIT_NORMALIZATION_CONSTANTS = {
    "seconds": 1 / 10**9,
    "minutes": (1 / 10**9) * (1 / 60),
    "hours": (1 / 10**9) * (1 / 60) * (1 / 60),
    "days": (1 / 10**9) * (1 / 60) * (1 / 60) * (1 / 24)
}
TIME_UNIT_NORMALIZATION_CONSTANT = TIME_UNIT_NORMALIZATION_CONSTANTS["hours"]

def __get_high_activity_users(activity_df):
    activity_df.index = activity_df["timestamp"]
    activity_df["count"] = np.ones(len(activity_df["timestamp"]))
    activity_df.drop("timestamp", axis="columns", inplace=True)
    activity_df = activity_df.groupby([pd.TimeGrouper("M"), "user"]).count()
    activity_df.rename(columns={"count":"activity"}, inplace=True)
    activity_df_high_activity_per_month = activity_df \
        .groupby("timestamp") \
        .agg(lambda values: np.percentile(values, constants.FREQUENT_PERCENTILE))
    activity_df.reset_index(inplace=True)
    activity_df_high_activity_per_month.reset_index(inplace=True)
    activity_df_high_activity_per_month.rename(columns={"activity":"high_activity"}, inplace=True)
    activity_df_high_activity_per_month["month_index"] = \
        activity_df_high_activity_per_month["timestamp"].apply(lambda a_date: a_date.year).map(str) + \
        "-" + \
        activity_df_high_activity_per_month["timestamp"].apply(lambda a_date: a_date.month).map(str)
    activity_df = activity_df.merge(activity_df_high_activity_per_month, left_on="timestamp", right_on="timestamp")
    activity_df["high_activity_user"] = activity_df["activity"] > activity_df["high_activity"]
    high_activity_users_per_month = {}
    low_activity_users_per_month = {}
    for month_index in activity_df_high_activity_per_month["month_index"]:
        high_activity_users_per_month[month_index] = \
            activity_df \
                .query("month_index == '{}'".format(month_index)) \
                .query("high_activity_user")["user"] \
                .tolist()
        low_activity_users_per_month[month_index] = \
            activity_df \
                .query("month_index == '{}'".format(month_index)) \
                .query("not high_activity_user")["user"] \
                .tolist()
    return high_activity_users_per_month, low_activity_users_per_month

def get_user_groups(some_path, some_dataset):
    the_path = some_path + "/{0}/{1}/{1}_activity_series_{0}.csv".format("questions", some_dataset)
    activity_df = pd.read_csv(the_path, sep="\t", parse_dates=[1], skipfooter=1, dtype={0: str}, engine="python").loc[:, ["user", "timestamp"]]
    hau_qs, lau_qs = __get_high_activity_users(activity_df)
    the_path = some_path + "/{0}/{1}/{1}_activity_series_{0}.csv".format("answers", some_dataset)
    activity_df = pd.read_csv(the_path, sep="\t", parse_dates=[1], skipfooter=1, dtype={0: str}, engine="python").loc[:, ["user", "timestamp"]]
    hau_as, lau_as = __get_high_activity_users(activity_df)
    return hau_qs, hau_as, lau_qs, lau_as

def load_dfs(some_path, some_dataset):
    freq_posters, freq_repliers, _, _ = get_user_groups(some_path, some_dataset)
    the_path = some_path + "/{0}/{1}/{1}_activity_series_{0}.csv".format("questions", some_dataset)
    df = pd.read_csv(the_path, sep="\t", parse_dates=[1], skiprows=[0], skipfooter=1,
                     header=None, dtype={0: str}, engine="python")
    df["month_index"] = df[1].apply(lambda a_date: a_date.year).map(str) + "-" + df[1].apply(lambda a_date: a_date.month).map(str)
    df["hau"] = df["month_index"].apply(lambda a_month: freq_posters[a_month])
    posts_freq = df.apply(lambda row: row[1] if row[2] in row["hau"] else None, axis=1).dropna()
    posts_nfreq = df.apply(lambda row: row[1] if row[2] not in row["hau"] else None, axis=1).dropna()
    the_path = some_path + "/{0}/{1}/{1}_activity_series_{0}.csv".format("answers", some_dataset)
    df = pd.read_csv(the_path, sep="\t", parse_dates=[1], skiprows=[0], skipfooter=1,
                     header=None, dtype={0: str}, engine="python")
    df["month_index"] = df[1].apply(lambda a_date: a_date.year).map(str) + "-" + df[1].apply(lambda a_date: a_date.month).map(str)
    df["hau"] = df["month_index"].apply(lambda a_month: freq_repliers[a_month])
    replies_freq = df.apply(lambda row: row[1] if row[2] in row["hau"] else None, axis=1).dropna()
    replies_nfreq = df.apply(lambda row: row[1] if row[2] not in row["hau"] else None, axis=1).dropna()
    return posts_freq, posts_nfreq, replies_freq, replies_nfreq

def __save_event_csvs(a_df, save_path, dataset_name, dataset_type):
    (a_df.astype(np.int64) * TIME_UNIT_NORMALIZATION_CONSTANT).sort_values().to_frame().T \
        .to_csv(save_path + "{0}/{0}_{1}.csv".format(dataset_name, dataset_type),
                index=False, header=False, mode="w")

def get_event_csvs(posts_freq, posts_nfreq, replies_freq, replies_nfreq, save_path, dataset_name):
    os.makedirs(save_path + dataset_name, exist_ok=True)
    __save_event_csvs(posts_freq, save_path, dataset_name, "posts_freq")
    __save_event_csvs(posts_nfreq, save_path, dataset_name, "posts_nfreq")
    __save_event_csvs(replies_freq, save_path, dataset_name, "replies_freq")
    __save_event_csvs(replies_nfreq, save_path, dataset_name, "replies_nfreq")

def process_dataset(some_dataset):
    print("handling {}".format(some_dataset))
    get_event_csvs(*load_dfs(constants.PATH_TO_SOURCE_DATASETS, some_dataset), \
                   constants.PATH_TO_DESTINATION_DATASET, some_dataset)

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=constants.NUMBER_OF_PROCESSES)
    pool.map(process_dataset, DATASET_LIST)
