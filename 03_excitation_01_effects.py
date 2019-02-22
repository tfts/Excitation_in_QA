'''
Uncovering Excitation: Estimating beta parameter and fitting of multivariate Hawkes
Usage: python3 03_excitation_01_effects.py
'''
import functools
import json
import multiprocessing

import hyperopt
import numpy as np
from tick.inference import HawkesExpKern


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
ANALYSIS_PERIOD_START = 1  # if analysis period is 2, then it skips timetamps of 1st quarter
MAXIMUM_BETA = 200
BETA_FIT_REPETITIONS = 15
ZERO = 1.e-8
MAX_HYPEROPT_EVALS = 100
NUMBER_OF_DIMENSIONS = constants.NUMBER_OF_DIMENSIONS
MODE = "STEM_VS_HUMAN" # "STEM_VS_HUMAN", "GROW_VS_DEC"
MODE_TO_DATASETS = constants.MODE_TO_DATASETS
DATASET_LIST = MODE_TO_DATASETS[MODE]
ALL_DATASET_LIST = []
for key in DATASET_LIST:
    ALL_DATASET_LIST += DATASET_LIST[key]
MIN_NUMBER_OF_EVENTS = constants.MIN_NUMBER_OF_EVENTS
SKIP_FITTING_BETA = True  # beta fitting can be computationally intensive
DIMENSION_NAMES = constants.DIMENSION_NAMES
BOOTSTRAP_REPETITIONS = constants.BOOTSTRAP_REPETITIONS
BOOTSTRAP_SAMPLES = len(DATASET_LIST[list(DATASET_LIST.keys())[0]])
assert len(DIMENSION_NAMES) == NUMBER_OF_DIMENSIONS
#EXEC_TIME = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
def __normalization_function(value_of_list, centering_value, min_of_scale, max_of_scale):
    return (value_of_list - centering_value) / (max_of_scale - min_of_scale)


###################
# FITTING BETA
###################
# Reading datasets
def __read_dataset(some_dataset):
    print("processing {}".format(some_dataset))
    timestamp_list = [np.genfromtxt(TEST_DATA_PATH_ORIGIN + some_dataset + "/" + some_dataset + dim_name + ".csv", dtype=np.float, delimiter=",") for dim_name in DIMENSION_NAMES]
    begin_and_end_timestamp_list = [{"first": timestamp_list_dim[0], "last": timestamp_list_dim[-1]}
                                    for timestamp_list_dim in timestamp_list]
    first_timestamp = min([begin_and_end_timestamps["first"]
                           for begin_and_end_timestamps in begin_and_end_timestamp_list])
    return [__normalization_function(timestamp_list[dim_i], first_timestamp, 0, 1) for dim_i in range(NUMBER_OF_DIMENSIONS)]

if not SKIP_FITTING_BETA:
    POOL = multiprocessing.Pool(processes=NUMBER_PROCESSES)
    EVENT_TIMES = POOL.map(__read_dataset, ALL_DATASET_LIST)

    # Fitting beta
    def __minimize_loglik_in_beta(event_times_dict_list, betas):
        beta_list = [[betas["beta"]] * NUMBER_OF_DIMENSIONS] * NUMBER_OF_DIMENSIONS
        learner = HawkesExpKern(beta_list)
        learner.fit(event_times_dict_list)
        return learner._solver_obj.get_history()["obj"][-1]

    fitted_betas = []
    print("starting beta fit")
    for iteration_nr in range(BETA_FIT_REPETITIONS):
        print("    beta iteration nr" + str(iteration_nr))
        fitted_beta = hyperopt.fmin(
            fn=lambda betas: __minimize_loglik_in_beta(EVENT_TIMES, betas),
            space={"beta": hyperopt.hp.uniform("beta", ZERO, MAXIMUM_BETA)},
            algo=hyperopt.tpe.suggest,
            max_evals=MAX_HYPEROPT_EVALS
        )["beta"]
        learner = HawkesExpKern([[fitted_beta] * NUMBER_OF_DIMENSIONS] * NUMBER_OF_DIMENSIONS)
        learner.fit(EVENT_TIMES)
        fitted_betas.append({"beta": fitted_beta,
                            "loglik": learner._solver_obj.get_history()["obj"][-1]})
    FITTED_BETA = min(fitted_betas, key=lambda i: i["loglik"])["beta"]
else: # beta fitting can be computationally intensive, so here are the results
    FITTED_BETA = 2.288 if MODE == "GROW_VS_DEC" else (2.067 if MODE == "STEM_VS_HUMAN" else raise Exception("unknown MODE"))

print("beta: {}".format(FITTED_BETA))


###################
# MODELLING OVER TIME
###################
# Reading datasets
def __read_dataset_window(some_dataset, window_index):
    timestamp_list = [np.genfromtxt(TEST_DATA_PATH_ORIGIN + some_dataset + "/" + some_dataset + dim_name + ".csv", dtype=np.float, delimiter=",") for dim_name in DIMENSION_NAMES]
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
        __read_dataset_bound_window = functools.partial(__read_dataset_window, window_index=time_span)
        EVENT_TIMES = POOL.map(__read_dataset_bound_window, DATASET_LIST[dataset_type])
        EVENT_TIMES = [events for events in EVENT_TIMES if events is not None]
        parameter_results = {"mus": [], "alphas": [], "betas": []}
        for _ in range(BOOTSTRAP_REPETITIONS):
            train_indexes = np.random.choice(range(len(EVENT_TIMES)), BOOTSTRAP_SAMPLES)
            train_set = [list(events_per_dimension) for events_per_dimension in list(np.array(EVENT_TIMES)[train_indexes])]
            learner = HawkesExpKern([[FITTED_BETA] * NUMBER_OF_DIMENSIONS] * NUMBER_OF_DIMENSIONS)
            learner.fit(train_set)
            parameter_results["mus"].append(learner.baseline.tolist())
            parameter_results["alphas"].append((learner.adjacency * np.array(learner.decays)).tolist())
            parameter_results["betas"].append(learner.decays)
        PERIOD_RESULTS.append({"mu": parameter_results["mus"],
                               "alpha": parameter_results["alphas"],
                               "beta": parameter_results["betas"],
                               "#datasets": len(EVENT_TIMES),
                               "quarter": time_span})
        EVENT_TIMES = None
    with open("quarter_excitation_{}.json".format(dataset_type), "w") as f:
        json.dump(PERIOD_RESULTS, f)
