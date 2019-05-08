'''
Permutation Experiments
Usage: python3 03_excitation_02_permutation.py
NOTE The parameter DIMS_TO_PERMUTE (per comparison type as stored in variable MODE) encodes which event-to-dimension associations to permute. It results from empirical observations of results from "03_excitation_01_effects.py". Also, note ANALYSIS_PERIOD_START and ANALYSIS_PERIOD_END can be set to cover only a period of interest.
'''
import functools
import json
import multiprocessing

import numpy as np
from tick.inference import HawkesExpKern

import constants


###################
# CONSTANTS
###################
NUMBER_PROCESSES = constants.NUMBER_OF_PROCESSES
TEST_DATA_PATH_ORIGIN = constants.PATH_TO_DESTINATION_DATASET
ANALYSIS_PERIOD = "m_3"
ANALYSIS_PERIOD_DICT = constants.ANALYSIS_PERIOD_DICT
ANALYSIS_PERIOD_OFFSET = ANALYSIS_PERIOD_DICT[ANALYSIS_PERIOD[:ANALYSIS_PERIOD.find("_")]] \
                         * int(ANALYSIS_PERIOD[ANALYSIS_PERIOD.find("_")+1:])
ANALYSIS_PERIOD_END = 12  # end analysis after ANALYSIS_PERIOD_END quarters
ANALYSIS_PERIOD_START = 8  # should be 1 for pcpa perm! if analysis period is 2, then it skips timetamps of 1st quarter
NUMBER_OF_DIMENSIONS = constants.NUMBER_OF_DIMENSIONS
PERMUTED_FITTING_REPETITIONS = 100
FIT_TYPE = "cqpa" # "cqca", "cqpa", "pqpa", "all". notation: c - casuals, p - power users, q - questions, a - answers, all - permute all dimensions
DIMS_TO_PERMUTE = [1, 2]  # grow_vs_dec - cqca: [1, 3], cqpa: [1, 2], pqpa: [0, 2]; stem_vs_human - all: list(range(4)), cqpa: [1, 2], cqca: [1, 3].
MODE = "STEM_VS_HUMAN" # "STEM_VS_HUMAN", "GROW_VS_DEC"
MODE_TO_DATASETS = constants.MODE_TO_DATASETS
DATASET_LIST = MODE_TO_DATASETS[MODE]
ALL_DATASET_LIST = []
for key in DATASET_LIST:
    ALL_DATASET_LIST += DATASET_LIST[key]
MIN_NUMBER_OF_EVENTS = constants.MIN_NUMBER_OF_EVENTS
DIMENSION_NAMES = constants.DIMENSION_NAMES
assert len(DIMENSION_NAMES) == NUMBER_OF_DIMENSIONS
#EXEC_TIME = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
def __normalization_function(value_of_list, centering_value, min_of_scale, max_of_scale):
    return (value_of_list - centering_value) / (max_of_scale - min_of_scale)
# previously determined beta values
FITTED_BETA = 2.288 if MODE == "GROW_VS_DEC" else (2.067 if MODE == "STEM_VS_HUMAN" else "unknown")
if FITTED_BETA == "unknown":
    raise Exception("unknown MODE")


print("beta: {}".format(FITTED_BETA))


###################
# MODELLING OVER TIME
###################
# Event to label permutation
def __eventlabel_permutation(list_of_events_per_dim):
    long_dim_array = sorted([(dim, event) for dim, event_list in enumerate(list_of_events_per_dim) for event in event_list], key=lambda e: e[1])
    event_dims = [event_dim for event_dim, event_time in long_dim_array if event_dim in DIMS_TO_PERMUTE]
    np.random.shuffle(event_dims)
    event_dim_shuffled_index = 0
    shuffled_long_dim_array = []
    for event_dim, event_time in long_dim_array:
        if event_dim in DIMS_TO_PERMUTE:
            new_event_dim = event_dims[event_dim_shuffled_index]
            event_dim_shuffled_index += 1
        else:
            new_event_dim = event_dim
        shuffled_long_dim_array.append((new_event_dim, event_time))
    result = [[] for i in range(NUMBER_OF_DIMENSIONS)]
    [result[dim].append(timestamp) for dim, timestamp in shuffled_long_dim_array]
    return [np.array(i) for i in result]

# Reading datasets
def __read_dataset_window(some_dataset, window_index):
    timestamp_list = [np.genfromtxt(TEST_DATA_PATH_ORIGIN + some_dataset + "/" + some_dataset + dim_name + ".csv", dtype=np.float, delimiter=",") for dim_name in DIMENSION_NAMES]
    timestamp_list = __eventlabel_permutation(timestamp_list)
    potential_window_start = [np.where(timestamp_list[dim] > timestamp_list[dim][0] + (window_index - 1) * ANALYSIS_PERIOD_OFFSET)
                              for dim in range(NUMBER_OF_DIMENSIONS)]
    potential_window_end = [np.where(timestamp_list[dim] <= timestamp_list[dim][0] + window_index * ANALYSIS_PERIOD_OFFSET)
                            for dim in range(NUMBER_OF_DIMENSIONS)]
    # check if all dimensions have events
    # np.where returns (x,) tuple, hence the following "hack"
    if all(map(len, [potential_window_start[dim][0] for dim in range(NUMBER_OF_DIMENSIONS)])):
        window_start = [np.min(potential_window_start[dim]) for dim in range(NUMBER_OF_DIMENSIONS)]
        window_end = [np.max(potential_window_end[dim]) for dim in range(NUMBER_OF_DIMENSIONS)]
        timestamp_list = [timestamp_list[dim][window_start[dim] : window_end[dim]]
                          for dim in range(NUMBER_OF_DIMENSIONS)]
        # check if all dimensions have enough events
        if all([len(timestamp_dim) > MIN_NUMBER_OF_EVENTS for timestamp_dim in timestamp_list]):
            begin_and_end_timestamp_list = [{"first": timestamp_dim[0], "last": timestamp_dim[-1]}
                                            for timestamp_dim in timestamp_list]
            first_timestamp = min([begin_and_end_timestamps["first"]
                                   for begin_and_end_timestamps in begin_and_end_timestamp_list])
            #print("  {} has len {}".format(some_dataset, tuple(map(len, timestamp_list))))
            return [__normalization_function(timestamp_list[dim_i], first_timestamp, 0, 1)
                    for dim_i in range(NUMBER_OF_DIMENSIONS)]

POOL = multiprocessing.Pool(processes=NUMBER_PROCESSES)
for dataset_type in DATASET_LIST.keys():
    PERIOD_RESULTS = []
    for time_span in range(ANALYSIS_PERIOD_START, ANALYSIS_PERIOD_END + 1):
        print("PROCESSING Q{}".format(time_span))
        parameter_results = {"mus": [], "alphas": [], "betas": []}
        for _ in range(PERMUTED_FITTING_REPETITIONS):
            __read_dataset_bound_window = functools.partial(__read_dataset_window, window_index=time_span)
            EVENT_TIMES = POOL.map(__read_dataset_bound_window, DATASET_LIST[dataset_type])
            EVENT_TIMES = [events for events in EVENT_TIMES if events is not None]
            learner = HawkesExpKern([[FITTED_BETA] * NUMBER_OF_DIMENSIONS] * NUMBER_OF_DIMENSIONS)
            learner.fit(EVENT_TIMES)
            parameter_results["mus"].append(np.array(learner.baseline).tolist())
            parameter_results["alphas"].append((learner.adjacency * np.array(learner.decays)).tolist())
            parameter_results["betas"].append(np.array(learner.decays).tolist())
        PERIOD_RESULTS.append({"mu": parameter_results["mus"],
                               "alpha": parameter_results["alphas"],
                               "beta": parameter_results["betas"],
                               "#datasets": len(EVENT_TIMES),
                               "quarter": time_span})
        EVENT_TIMES = None
    with open("quarter_permutation_{}_{}.json".format(dataset_type, FIT_TYPE), "w") as f:
        json.dump(PERIOD_RESULTS, f)
