'''
Prediction Experiments
Usage: python3 03_excitation_03_prediction.py [effect_type] [model_type] [time_span]
Usage example: python3 03_excitation_03_prediction.py early reduced 3
'''
import json
import os
import sys

import numpy as np
from scipy import stats
import sklearn.metrics as metrics
from tick.inference import HawkesExpKern
from tick.simulation import SimuHawkesExpKernels, SimuHawkesMulti

import constants

###################
# CONSTANTS
###################
TEST_DATA_PATH_ORIGIN = constants.PATH_TO_DESTINATION_DATASET
ANALYSIS_PERIOD = "m_3"
ANALYSIS_PERIOD_DICT = constants.ANALYSIS_PERIOD_DICT
ANALYSIS_PERIOD_OFFSET = ANALYSIS_PERIOD_DICT[ANALYSIS_PERIOD[:ANALYSIS_PERIOD.find("_")]] \
                         * int(ANALYSIS_PERIOD[ANALYSIS_PERIOD.find("_")+1:])
ANALYSIS_PERIOD_END = 12  # end analysis after ANALYSIS_PERIOD_END quarters
ANALYSIS_PERIOD_START = 1  # if analysis period is 2, then it skips timetamps of 1st quarter
NUMBER_OF_DIMENSIONS = constants.NUMBER_OF_DIMENSIONS
PRINT_FOLD_SIZES = True
INTENSITY_PARAMETERS_HISTORY = []
MODE = "GROW_VS_DEC"
MODE_TO_DATASETS = constants.MODE_TO_DATASETS
DATASET_LIST = MODE_TO_DATASETS[MODE]
MIN_NUMBER_OF_EVENTS = constants.MIN_NUMBER_OF_EVENTS
DIMENSION_NAMES = constants.DIMENSION_NAMES
assert len(DIMENSION_NAMES) == NUMBER_OF_DIMENSIONS
N_REALIZATIONS = 100  # 2, 10, 100
RESULTS_PATH_PREFIX = "quarter_prediction_"
RESULTS_PATH_SUFFIX = ".json"

HORIZON = "quarter"  # "month", "quarter", "week"
if HORIZON == "month":
    SIMU_END = ANALYSIS_PERIOD_OFFSET / int(ANALYSIS_PERIOD[-1])
elif HORIZON == "quarter":
    SIMU_END = ANALYSIS_PERIOD_OFFSET
elif HORIZON == "week":
    SIMU_END = ANALYSIS_PERIOD_OFFSET / int(ANALYSIS_PERIOD[-1]) / 4
EFFECT_TYPE = sys.argv[1]  # "s-e", "late", "early"
EFFECT_TIMESPAN_DICT = { # result of empirical observations of results from "03_excitation_01_effects.py"
    "s-e": range(8, 12),
    "late": range(8, 12),
    "early": range(1, 7)
}
EFFECT_TIMESPAN = EFFECT_TIMESPAN_DICT[EFFECT_TYPE]
#EXEC_TIME = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
def __normalization_function(value_of_list, centering_value, min_of_scale, max_of_scale):
    return (value_of_list - centering_value) / (max_of_scale - min_of_scale)
FITTED_BETA = 2.288  # previously determined beta value for growing-vs-declining comparison
print("beta: {}".format(FITTED_BETA))


###################
# PREDICTING OVER TIME
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

def __fit_std_Hawkes(dataset_events):
    learner = HawkesExpKern([[FITTED_BETA] * NUMBER_OF_DIMENSIONS] * NUMBER_OF_DIMENSIONS)
    learner.fit(dataset_events)
    mus = learner.baseline
    betas = learner.decays
    alphas = learner.adjacency
    return mus, alphas, betas

def __fit_current_quarter(dataset_events, model_type):
    # fit process to current dataset_events
    if model_type == "full":
        mus, alphas, betas = __fit_std_Hawkes(dataset_events)
    elif model_type == "baseline":
        mus = np.array([len(dim) for dim in dataset_events]) / ANALYSIS_PERIOD_OFFSET
        alphas = np.zeros((NUMBER_OF_DIMENSIONS, NUMBER_OF_DIMENSIONS))
        betas = np.ones(alphas.shape)
    elif model_type == "s-e":
        mus = []
        alphas = []
        for dim_i in range(NUMBER_OF_DIMENSIONS):
            learner = HawkesExpKern([[FITTED_BETA]])
            learner.fit([dataset_events[dim_i]])
            mus.append(learner.baseline[0].tolist())
            dim_alphas = np.zeros(NUMBER_OF_DIMENSIONS)
            dim_alphas[dim_i] = learner.adjacency[0][0].tolist()
            alphas.append(dim_alphas)
        betas = np.eye(NUMBER_OF_DIMENSIONS) * FITTED_BETA
    elif model_type == "reduced":
        if EFFECT_TYPE == "s-e":
            mus, alphas, betas = __fit_std_Hawkes(dataset_events)
            for i in range(NUMBER_OF_DIMENSIONS):
                alphas[i][i] = 0
        elif EFFECT_TYPE == "late":
            mus, alphas, betas = __fit_std_Hawkes(dataset_events)
            alphas[1][3] = 0
            alphas[3][1] = 0
        elif EFFECT_TYPE == "early":
            mus, alphas, betas = __fit_std_Hawkes(dataset_events)
            alphas[0][2] = 0
            alphas[1][2] = 0
            alphas[2][0] = 0
            alphas[3][0] = 0
            alphas[1][0] = 0
    else:
        raise Exception("unknown model_type")
    return mus, alphas, betas

def __get_statistic(event_times):
    statistics_dict = {"counts": len(event_times)}
    if len(event_times) <= 1:
        statistics_dict["interarrival"] = None
    else:
        statistics_dict["interarrival"] = [event_times[i + 1] - event_times[i] for i in range(len(event_times) - 1)]
    return statistics_dict

def __simulate_statistics(mus, alphas, betas):
    # simulate next time_span statistics
    simu_params = SimuHawkesExpKernels(adjacency=alphas, decays=betas,
                                    baseline=mus, end_time=SIMU_END, 
                                    verbose=False)
    if simu_params.spectral_radius() > 1.:
        return None
    multi = SimuHawkesMulti(simu_params, n_simulations=N_REALIZATIONS)
    multi.simulate()
    # check simulated statistics
    count_statistics = [[], [], [], []]
    interarrival_statistics = [[], [], [], []]
    for realization in multi.timestamps:
        for dim_i, dim in enumerate(realization):
            dim_statistics = __get_statistic(dim)
            count_statistics[dim_i].append(dim_statistics["counts"])
            interarrival_statistics[dim_i].append(dim_statistics["interarrival"])
    result_dict = {"counts": np.mean(count_statistics, axis=1).tolist(), "interarrival": interarrival_statistics}
    result_list = []
    for dim_i in range(NUMBER_OF_DIMENSIONS):
        result_list.append({"counts": result_dict["counts"][dim_i], "interarrival": result_dict["interarrival"][dim_i]})
    return result_list

def __load_setup(filename):
    if not os.path.isfile(filename):
        with open(filename, "w") as f:
            f.write("{}\n")
    with open(filename, "r") as f:
        return json.load(f)

model_type = sys.argv[2]  # ["full", "reduced", "baseline", "s-e"]
results_path = RESULTS_PATH_PREFIX + EFFECT_TYPE + RESULTS_PATH_SUFFIX
predictions = __load_setup(results_path)
if model_type not in predictions:
    predictions[model_type] = {}
# for time_span in EFFECT_TIMESPAN:
time_span = int(sys.argv[3])
predictions[model_type][time_span] = {}
predictions[model_type][time_span]["instances"] = []
predictions[model_type][time_span]["true_count"] = []
predictions[model_type][time_span]["predicted_count"] = []
predictions[model_type][time_span]["true_interarrival"] = []
predictions[model_type][time_span]["predicted_interarrival"] = []
for dataset_type in DATASET_LIST.keys():
    # for time_span in range(ANALYSIS_PERIOD_START, ANALYSIS_PERIOD_END + 1):
    print("PROCESSING Q{}, MODEL {}".format(time_span, model_type))
    for dataset_i, dataset_name in enumerate(DATASET_LIST[dataset_type]):
        # read datasets
        dataset_events = __read_dataset_window(dataset_name, time_span)
        if dataset_events is None:
            continue
        print("  " + DATASET_LIST[dataset_type][dataset_i])
        dataset_events_next_period = __read_dataset_window(dataset_name, time_span + 1)
        if dataset_events_next_period is None:
            continue
        for dim_i, dim in enumerate(dataset_events_next_period):
            dataset_events_next_period[dim_i] = dim[np.where(dim <= SIMU_END)]
        # get actual statistics
        actual_statistics = [__get_statistic(dataset_events_next_period[dim_i]) for dim_i in range(NUMBER_OF_DIMENSIONS)]
        # get simulated statistics
        mus, alphas, betas = __fit_current_quarter(dataset_events, model_type)
        simulated_statistics = __simulate_statistics(mus, alphas, betas)
        if simulated_statistics is None:
            continue
        # compute and save prediction results
        predictions[model_type][time_span]["instances"].append(DATASET_LIST[dataset_type][dataset_i])
        predictions[model_type][time_span]["true_count"].append([actual_statistics[dim_i]["counts"] for dim_i in range(NUMBER_OF_DIMENSIONS)])
        predictions[model_type][time_span]["predicted_count"].append([simulated_statistics[dim_i]["counts"] for dim_i in range(NUMBER_OF_DIMENSIONS)])
        predictions[model_type][time_span]["true_interarrival"].append([actual_statistics[dim_i]["interarrival"] for dim_i in range(NUMBER_OF_DIMENSIONS)])
        predictions[model_type][time_span]["predicted_interarrival"].append([simulated_statistics[dim_i]["interarrival"] for dim_i in range(NUMBER_OF_DIMENSIONS)])
# if not any(np.isnan(predictions[time_span]["burstiness_predicted"][dim]))
predictions[model_type][time_span]["rmse"] = []
predictions[model_type][time_span]["KS"] = []
for dim in range(NUMBER_OF_DIMENSIONS):
    dim_counts_true = np.array(predictions[model_type][time_span]["true_count"])[:, dim]
    dim_counts_predicted = np.array(predictions[model_type][time_span]["predicted_count"])[:, dim]
    dim_counts_true_nonnan = dim_counts_true[~np.isnan(dim_counts_predicted)]
    dim_counts_predicted_nonnan = dim_counts_predicted[~np.isnan(dim_counts_predicted)]
    dim_rmse = round(np.sqrt(metrics.mean_squared_error(dim_counts_true_nonnan, dim_counts_predicted_nonnan)).tolist(), 2)
    predictions[model_type][time_span]["rmse"].append(dim_rmse)
    dim_interarrival_true = np.array(predictions[model_type][time_span]["true_interarrival"])[:, dim]
    dim_interarrival_predicted = np.array(predictions[model_type][time_span]["predicted_interarrival"])[:, dim]
    # computing mean of ks between each realization and true
    dim_ks_results = []
    for dataset_i, _ in enumerate(dim_interarrival_true):
        dim_ks_i_realizations = []
        for realization_i in range(N_REALIZATIONS):
            if (dim_interarrival_true is None or 
                    dim_interarrival_predicted is None or 
                    dim_interarrival_true[dataset_i] is None or 
                    dim_interarrival_predicted[dataset_i] is None or 
                    dim_interarrival_predicted[dataset_i][realization_i] is None):
                continue
            dim_ks_i_realization, _ = stats.ks_2samp(dim_interarrival_true[dataset_i], dim_interarrival_predicted[dataset_i][realization_i])
            if dim_ks_i_realization is not None:
                dim_ks_i_realizations.append(dim_ks_i_realization)
        if dim_ks_i_realizations is not None:
            dim_ks_results.append(np.mean(dim_ks_i_realizations))
    predictions[model_type][time_span]["KS"].append(np.mean([res for res in dim_ks_results if res is not None and not np.isnan(res)]).tolist())
rmse = predictions[model_type][time_span]["rmse"]
ks = predictions[model_type][time_span]["KS"]
print("Q{} has rmse {} and KS {}"
        .format(time_span,
                rmse,
                ks))
predictions[model_type][time_span] = {}
predictions[model_type][time_span]["rmse"] = rmse
predictions[model_type][time_span]["KS"] = ks
with open(results_path, "w") as f:
    json.dump(predictions, f)

# NOTE value of ks_2samp of .18 corresponds to pvalue 0.0025 so REJECT EQUALITY. KS = .12 (.135) <=> pvalue .10 (.5), so KS of .12 (.135) or lower are non-significant at all usual significance levels
