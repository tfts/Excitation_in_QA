'''
Univariate Hawkes Process
'''
import datetime
from bisect import bisect_left

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np


class UnivariateHawkes(object):
    '''Univariate Hawkes process object

    Parameters
    ----------
    mu : float
        Base intensity
    alpha : float
        Intensity jump on event
    beta : float
        Self-excitation decay
    '''

    def __init__(self, mu, alpha, beta):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

    def lambda_at(self, time_t, event_times):
        '''Compute instantaneous intensity rate lambda at time t

        Parameters
        ----------
        time_t : float
            Time tick on which to evaluate intensity
        event_times : list of floats
            History of events (has to be in ascending order)

        Returns
        -------
        intensity : float
            Intensity of Hawkes Process at time t
        '''
        intensity = self.mu + np.sum(self.alpha * np.exp(-self.beta * (time_t - event_times)))
        return intensity

    def get_intensities(self, times, event_times):
        '''Compute intensity rates lambda(t) over a set of (regularly spaced) time stamps

        Parameters
        ----------
        times : list of floats
            List of time ticks (has to be in ascending order)
        event_times : list of floats
            History of events (has to be in ascending order)

        Returns
        -------
        intensities : list of floats
            Intensities of Hawkes Process over input times
        '''
        intensities = []
        for time_ti in times:
            event_times_up_to_ti = event_times[:bisect_left(event_times, time_ti)]
            intensities.append(self.lambda_at(time_ti, event_times_up_to_ti))
        return intensities

    def get_samples(self, times):
        '''Sample from univariate Hawkes process

        Parameters
        ----------
        times : list of floats
            List of time ticks (has to be in ascending order and start with 0)

        Returns
        -------
        event_times : list of floats
            Sampled event times
        '''
        event_times = []
        candidate_event_time = 0
        simulation_end = times[-1]
        while candidate_event_time < simulation_end:
            lambda_bar = self.lambda_at(candidate_event_time, np.array(event_times))
            # w is inverse sampling of inter-event time til next event,
            # which gets rejected if unlikely (hence lambda_bar)
            w = -np.log(np.random.uniform()) / lambda_bar
            candidate_event_time += w
            if (np.random.uniform() * lambda_bar <= self.lambda_at(candidate_event_time, np.array(event_times)) and
                    candidate_event_time <= simulation_end):
                event_times.append(candidate_event_time)
        return event_times

    def plot(self, times, event_times):
        '''Plot intensity rate over time for given events. Saves plot to disk.

        Parameters
        ----------
        times : list of floats
            List of time ticks (has to be in ascending order)
        event_times : list of floats
            History of events (has to be in ascending order)
        '''
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42
        #rcParams['text.usetex'] = True
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(times, self.get_intensities(times, event_times))
        ax.plot(event_times, [0.05] * len(event_times), '|', color='k')
        ax.set_title('Simulated Arrival Intensity')
        ax.set_xlabel('Time')
        ax.set_ylabel('Intensity')
        fig.text(.01, .03,
                 r"$\mu = {}, \alpha = {}, \beta = {}$" \
                    .format(self.mu, self.alpha, self.beta))
        fig.tight_layout()
        plt.grid()
        plt.savefig("univariate_intensity_over_time_{}.png" \
            .format(datetime.datetime.today().strftime('%Y%m%d-%H%M%S')))


if __name__ == "__main__":
    # CONSTANTS FROM "https://www.math.fsu.edu/~ychen/research/Thinning algorithm.pdf" (retrieved July 17, 2017)
    EVENTS = [4.8, 5.2, 7.5, 8.8, 10.9, 11.3, 11.4, 12.4]
    TIMES = np.linspace(0, 13, 500)
    INTENSITY_PARAMETERS = {
        "alpha": 0.6,
        "beta": 0.8,
        "mu": 1.2
    }

    # TESTS
    univariate_hawkes = UnivariateHawkes(**INTENSITY_PARAMETERS)
    EVENTS = univariate_hawkes.get_samples(TIMES)
    univariate_hawkes.plot(TIMES, EVENTS)
