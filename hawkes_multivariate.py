'''
Multivariate Hawkes Process
'''
import datetime
import multiprocessing
from bisect import bisect_left
from collections import Counter

import lmfit
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

import constants
from hawkes_univariate import UnivariateHawkes

class MultivariateHawkes(object):
    '''Multivariate Hawkes process object

    Parameters
    ----------
    mu : list of floats
        Base intensities per dimension
    alpha : matrix of floats
        Intensity jumps on event per dimension^2
    beta : matrix of floats
        Self-excitation decay per dimension^2
    '''

    def __init__(self, mu, alpha, beta):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

    def pprint(self):
        print("### mu ###")
        print(np.array(self.mu))
        print("### alpha ###")
        print(np.array(self.alpha))
        print("### beta ###")
        print(np.array(self.beta))
        print("############")

    def lambda_at(self, dimension_m, time_t, event_times):
        '''Compute instantaneous intensity rate lambda_m at time t

        Parameters
        ----------
        time_t : float
            Time tick on which to evaluate intensity
        event_times : dict of lists of floats
            History of events (values have to be in ascending order) per dimension (key)

        Returns
        -------
        intensity : float
            Intensity of multivariate Hawkes Process at time t
        '''
        intensity = self.mu[dimension_m]
        for dimension_n in event_times:
            events_n_up_to_time_t = \
                event_times[dimension_n][:bisect_left(event_times[dimension_n], time_t)]
            # note that we are adding to the base intensities,
            #  so we pass base intensity mu to lambda_at of 0
            intensity_parameters_m_n = {
                "mu": 0,
                "alpha": self.alpha[dimension_m][dimension_n],
                "beta": self.beta[dimension_m][dimension_n]
            }
            dummy_univariate_hawkes = UnivariateHawkes(**intensity_parameters_m_n)
            # summing up over events of a certain dimension up to time_ti, over all dimensions n
            intensity += dummy_univariate_hawkes.lambda_at(time_t, np.array(events_n_up_to_time_t))
        return intensity

    def get_intensities(self, times, event_times):
        '''Compute intensity rates lambda(t) over a set of (regularly spaced) time stamps

        Parameters
        ----------
        times : list of floats
            List of time ticks (has to be in ascending order)
        event_times : dict of lists of floats
            History of events (values have to be in ascending order) per dimension (key)

        Returns
        -------
        intensities : dict of lists of floats
            Multidimensional intensities of Hawkes Process over input times
        '''
        intensities = {}
        for dimension_m in event_times:
            intensities[dimension_m] = []
            for time_ti in times:
                intensities[dimension_m].append(self.lambda_at(dimension_m, time_ti, event_times))
        return intensities

    def get_samples(self, times):
        '''Sample from multivariate Hawkes process

        Parameters
        ----------
        times : list of floats
            List of time ticks (has to be in ascending order and start with 0)

        Returns
        -------
        event_times : dict of lists of floats
            Sampled events per dimension
        '''
        event_times = {dimension_m: [] for dimension_m, _ in enumerate(self.mu)}
        candidate_event_time = 0
        simulation_end = times[-1]
        while candidate_event_time < simulation_end:
            lambda_bar = np.sum([self.lambda_at(dimension_m, candidate_event_time, event_times) for dimension_m in event_times])
            # inverse sampling of inter-event time til next event, which gets rejected if unlikely (hence lambda_bar)
            w = -np.log(np.random.uniform()) / lambda_bar
            candidate_event_time += w
            D = np.random.uniform()
            if (D * lambda_bar <= np.sum([self.lambda_at(dimension_m, candidate_event_time, event_times) for dimension_m in event_times]) and
                    candidate_event_time <= simulation_end):
                # searching for candidate_dimension to assign sampled event to
                candidate_dimension = 0
                while D * lambda_bar > np.sum([self.lambda_at(dimension_m, candidate_event_time, event_times) for dimension_m in range(candidate_dimension + 1)]):
                    candidate_dimension += 1
                event_times[candidate_dimension].append(candidate_event_time)
        return event_times

    def plot_counts(self, event_times, plot_title=None):
        '''Plot counts per time interval for given events. Saves plot to disk.

        Parameters
        ----------
        times : list of floats
            Set of time ticks (has to be in ascending order)
        events : list of floats
            Set of events (has to be in ascending order)
        plot_title : string
            Optional string to append to resulting plot's filename
        '''
        # Computing event counts per time unit
        counts = []
        for dim_i in event_times:
            counts_dict = Counter(np.floor(event_times[dim_i]).astype(np.int64))
            counts_list = []
            for time_index in range(1, int(np.ceil(event_times[dim_i][-1]) + 1)):
                if time_index in counts_dict:
                    counts_list.append(counts_dict[time_index])
                else:
                    counts_list.append(0)
            counts.append(counts_list)
        for dim_i, counts_list in enumerate(counts):
            print("number of events in dim {}: {}".format(dim_i, np.sum(counts_list)))
        # Plotting
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42
        rcParams['text.usetex'] = True
        number_of_dimensions = len(self.mu)
        times = [np.arange(1, event_times[dim_i][-1] + 1).astype(np.int64) for dim_i in range(number_of_dimensions)]
        fig, ax_list = plt.subplots(1, number_of_dimensions, figsize=(14, 4))
        for ax_i, ax in enumerate(ax_list):
            ax.plot(times[ax_i], counts[ax_i])
            ax.set_title('Number of events')
            ax.set_xlabel('Time')
            ax.set_ylabel('number of events in dimension {}'.format(ax_i + 1))
            ax.grid()
        fig.text(.005, .012, self.__format_parameters_for_plot_caption())
        plt.tight_layout(w_pad=5)
        plt.savefig("multivariate_counts_per_time_unit_{}_{}.png"
                    .format(datetime.datetime.today().strftime('%Y%m%d-%H%M%S'), plot_title))

    def plot_intensities(self, times, event_times, plot_title=None):
        '''Plot intensity rate over time for given events. Saves plot to disk.

        Parameters
        ----------
        times : list of floats
            Set of time ticks (has to be in ascending order)
        events : list of floats
            Set of events (has to be in ascending order)
        plot_title : string
            Optional string to append to resulting plot's filename
        '''
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42
        rcParams['text.usetex'] = True
        fig, ax_list = plt.subplots(1, len(self.mu), figsize=(14, 4))
        for ax_i, ax in enumerate(ax_list):
            ax.plot(times, self.get_intensities(times, event_times)[ax_i])
            ax.plot(event_times[ax_i], [0.05] * len(event_times[ax_i]), '|')
            ax.set_title('Simulated Arrival Intensity')
            ax.set_xlabel('Time')
            ax.set_ylabel(r'$\lambda^{}(t)$'.format(ax_i + 1))
            ax.grid()
        fig.text(.005, .012, self.__format_parameters_for_plot_caption())
        plt.tight_layout(w_pad=5)
        plt.savefig("multivariate_intensity_over_time_{}_{}.png"
                    .format(datetime.datetime.today().strftime('%Y%m%d-%H%M%S'), plot_title))

    def __format_parameters_for_plot_caption(self):
        mu_matrix = r"\mu = \left[ \begin{array}{c} "
        alpha_matrix = r"\alpha = \left[ \begin{array}{" + \
            r"c" * len(self.mu) + r"} "
        beta_matrix = r"\beta = \left[ \begin{array}{" + \
            r"c" * len(self.mu) + r"} "
        for dimension_m, _ in enumerate(self.mu):
            mu_matrix += r"{} \\".format(self.mu[dimension_m])
            alpha_matrix += " & ".join([str(round(a, 2)) for a in self.alpha[dimension_m]]) + r" \\"
            beta_matrix += " & ".join([str(round(b, 2)) for b in self.beta[dimension_m]]) + r" \\"
        mu_matrix = mu_matrix[:-1] + r"end{array} \right] "
        alpha_matrix = alpha_matrix[:-1] + r"end{array} \right] "
        beta_matrix = beta_matrix[:-1] + r"end{array} \right] "
        return r"$ {} {} {} $".format(mu_matrix, alpha_matrix, beta_matrix)

    def log_likelihood(self, event_times):
        '''Compute log-likelihood for observed event history given multidimensional Hawkes Process parameters.
        This follows the log-likelihood proposition 2.2 of "Likelihood Function of Multivariate Hawkes Processes" by Y. Chen.

        Parameters
        ----------
        event_times : dict of lists of floats
            History of events (values have to be in ascending order) per dimension (key)

        Returns
        -------
        log_likelihood : float
            log-likelihood of input data given Hawkes Process parameters
        '''
        log_likelihood = 0
        last_timestamp = max([max(event_times[dimension_m]) for dimension_m in event_times])
        for dimension_m in event_times:
            log_likelihood_1st_summand = -self.mu[dimension_m] * last_timestamp
            log_likelihood_2nd_summand = 0
            for dimension_n in event_times:
                alpha_m_n = self.alpha[dimension_m][dimension_n]
                beta_m_n = self.beta[dimension_m][dimension_n]
                log_likelihood_2nd_summand += alpha_m_n / beta_m_n * \
                    sum([1 - np.exp(-beta_m_n * (last_timestamp - event_n))
                         for event_n in event_times[dimension_n]])
            log_likelihood_3rd_summand = 0
            for event_m_k in range(len(event_times[dimension_m]) - 1, -1, -1):
                log_likelihood_3rd_summand += \
                    np.log(self.mu[dimension_m] + sum([self.alpha[dimension_m][dimension_n] * 
                        self.__R_imperative(event_m_k, dimension_m, dimension_n, event_times) for dimension_n in event_times]))
            log_likelihood += log_likelihood_1st_summand - log_likelihood_2nd_summand + log_likelihood_3rd_summand
        return log_likelihood

    def __R_imperative(self, index_k, dimension_m, dimension_n, event_times):
        event_m_k = event_times[dimension_m][index_k]
        events_n = np.array(event_times[dimension_n])
        events_n_i = np.extract(events_n < event_m_k, events_n)
        return sum(np.exp(-self.beta[dimension_m][dimension_n] * (event_m_k - events_n_i)))

def fit_hawkes(event_times_dict_list, predefined_beta_matrix=None, pool=None):
    '''Fit a multidimensional Hawkes Process to observed sequences

    Parameters
    ----------
    event_times_dict_list : list of dict of lists of floats
        List of sequences of event dicts, with process dimension as key and list of ascending timestamps (floats) as values
    predefined_beta_matrix : matrix of floats
        Self-excitation decay per dimension^2. If equals None, then it will be a parameter to fit.
    Returns
    -------
    fitted_hawkes : MultivariateHawkes
        New Hawkes Process with fitted parameters
    '''
    parameters = lmfit.Parameters()
    for dimension_m in event_times_dict_list[0]:
        parameters.add("mu_{}".format(dimension_m), min=0, value=0.9)
        for dimension_n in event_times_dict_list[0]:
            parameters.add("alpha_{}{}".format(dimension_m, dimension_n), min=0, value=0.9)
            if not predefined_beta_matrix:
                parameters.add("beta_{}{}".format(dimension_m, dimension_n), min=0, value=0.9)
    
    def __extract_parameters(parameters):
        mu_list = []
        alpha_matrix = []
        beta_matrix = []
        for dim_m in event_times_dict_list[0]:
            mu_list.append(parameters["mu_{}".format(dim_m)])
            alpha_matrix.append([])
            beta_matrix.append([])
            for dim_n in event_times_dict_list[0]:
                alpha_matrix[dim_m].append(parameters["alpha_{}{}".format(dim_m, dim_n)])
                if not predefined_beta_matrix:
                    beta_matrix[dim_m].append(parameters["beta_{}{}".format(dim_m, dim_n)])
        return {"mu": mu_list,
                "alpha": alpha_matrix,
                "beta": beta_matrix if not predefined_beta_matrix else predefined_beta_matrix}

    def log_likelihood_wrapper(parameters, event_times_dict_list, pool):
        current_hawkes_process = MultivariateHawkes(**__extract_parameters(parameters))
        current_hawkes_process.pprint()
        '''
        We compute -1 * log_likelihood, because we use algorithms that minimize,
        and finding parameters that maximize likelihood is equivalent to finding those that minimize -log_likelihood
        '''
        result = pool.map(current_hawkes_process.log_likelihood, event_times_dict_list)
        return list(np.array(result) * -1)
        # return [-1 * current_hawkes_process.log_likelihood(event_times_dict) for event_times_dict in event_times_dict_list]

    if not pool:
        pool = multiprocessing.Pool(processes = constants.NUMBER_OF_PROCESSES)
    minimizer = lmfit.Minimizer(log_likelihood_wrapper, parameters, fcn_args=(event_times_dict_list, pool))
    result = minimizer.minimize(method="lbfgsb")

    lmfit.report_fit(result)

    return {"parameters": __extract_parameters(parameters), 
            "average_negativeloglikelihood": np.mean(log_likelihood_wrapper(parameters, event_times_dict_list, pool))}


if __name__ == "__main__":
    # CONSTANTS FROM "https://www.math.fsu.edu/~ychen/research/multiHawkes.pdf" (retrieved July 17, 2017)
    TIMES = np.linspace(0, 750, 10000)
    INTENSITY_PARAMETERS = {
        "mu": [0.1, 0.5],
        "alpha": [[0.1, 0.7], [0.5, 0.2]],
        "beta": [[1.2, 1.0], [0.8, 0.6]]
    }

    # TESTS
    multivariate_hawkes = MultivariateHawkes(**INTENSITY_PARAMETERS) 
    EVENT_TIMES = multivariate_hawkes.get_samples(TIMES)
    multivariate_hawkes.plot_intensities(TIMES, EVENT_TIMES)
    fit_hawkes(EVENT_TIMES, INTENSITY_PARAMETERS["beta"])

