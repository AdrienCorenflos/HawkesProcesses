import numpy as np
import matplotlib.pyplot as plt


class MultivariateHawkesProcess(object):
    """
    Class aiming at simulating, fitting,
    checking a linear multivariate Hawkes process

    :param lambda_0_list: initial intensities for the processes
    :param t_0: starting time of the process, should be 0.
    :param cross_alphas: size of cross-jumps in the intensity
    :param cross_betas: decay influence of past jumps parameter

    :return: returns a list of jumps via simulate
    """

    def __init__(self, lambda_0_list, t_0, cross_alphas, cross_betas, seed=1):
        np.random.seed(seed)

        ########################################
        #       Constants  initialisation      #
        ########################################

        self.lambda_0_array = np.array(lambda_0_list)
        self.t_0 = t_0
        self.cross_alphas = np.array(cross_alphas)
        self.cross_betas = np.array(cross_betas)
        self.dimension = len(lambda_0_list)

        ########################################
        #       Variables  initialisation      #
        ########################################

        self._max_intensity = np.sum(lambda_0_list)
        self._t = t_0
        self._s = t_0
        self._lambda_participation = np.zeros((self.dimension, self.dimension))
        self._us = np.repeat(t_0, self.dimension)
        self._last_accepted = None

        ########################################
        #        Results  initialisation       #
        ########################################

        self.check = []
        self._jumps = [[] for _ in range(self.dimension)]
        self._finished = False

    ########################################
    #       Random methods definition      #
    ########################################

    @staticmethod
    def u():
        """
        Returns an instance of a uniform
        """
        return np.random.uniform()

    @staticmethod
    def exp(lamb):
        """
        Returns an instance of an exponential law
        """
        return np.random.exponential(1 / lamb)

    def _lambda_fun(self, update=False):
        res = self._lambda_participation * np.exp(- self.cross_betas * (self._s - self._us))
        if update:
            self._lambda_participation = res
        res = res + np.diag(self.lambda_0_array)

        res = np.sum(res, axis=0)

        return res

    ########################################
    #     Routine step nb 1 : initialise   #
    ########################################

    def _first_jump(self, T):
        self._s = self.exp(self._max_intensity)
        self._finished = self._s > T
        self._is_initial = False

        if not self._finished:
            d = self.u()
            cum_sum_lambdas = np.cumsum(self.lambda_0_array)
            n_0 = np.searchsorted(cum_sum_lambdas, d * self._max_intensity,
                                  'right')
            self._t = self._s
            self._last_jump = self._t
            self._jumps[n_0].append(self._s)
            self._last_accepted = n_0

    ########################################
    #      Routine step nb 3 : continue    #
    ########################################

    def _accepted_routine(self):
        """
        If the last jump proposal has been accepted, add it to the
        accepted jumps.
        """
        self._t = self._s
        self._jumps[self._last_accepted].append(self._t)
        self._last_jump = self._t

    def _continue_routine(self):
        if self._last_accepted is not None:
            self._accepted_routine()

    ########################################
    #     Routine step nb 2 : main work    #
    ########################################

    def _routine(self, T):
        """
        This is the main routine for the algorithm.
        If the last jump has been accepted, impact it, then check next
        jump and continue.
        """
        if self._last_accepted is not None:
            n_0_intensity = self.cross_alphas[:, self._last_accepted].flatten()
            self._lambda_participation[self._last_accepted] += n_0_intensity
            self._max_intensity = np.sum(np.sum(self._lambda_participation)) + np.sum(self.lambda_0_array)

        self._us[:] = self._s
        self._s += self.exp(self._max_intensity)
        if self._s > T:
            self._finished = True

        if not self._finished:
            d = self.u()
            lambdas = self._lambda_fun(update=True)
            if self._max_intensity * d > np.sum(lambdas):
                self._last_accepted = None
                self._max_intensity = np.sum(np.sum(lambdas))
                self._continue_routine()

            else:
                lambdas = np.cumsum(lambdas)
                n_0 = np.searchsorted(lambdas, self._max_intensity * d, 'right')
                self._last_accepted = n_0
                self._continue_routine()

    ########################################
    #    Routine : launch the simulation   #
    ########################################

    def simulate(self, T):
        self._first_jump(T)
        while not self._finished:
            self._routine(T)

        result = self._jumps[:]
        return result

    def test_jumps_distribution(self):
        """
        The following algorithm is a transformation of the jump times so as to
        get Exp(1.) distributed interval times.
        The probplot at the end is there so we can check it on a QQplot
        """
        a = np.zeros((self.dimension, self.dimension))
        last_t = np.zeros(self.dimension)
        last_last_t = np.zeros(self.dimension)
        last_tau = np.zeros(self.dimension)

        taus = [[0.] for _ in range(self.dimension)]

        for m in range(self.dimension):
            for t in self._jumps[m]:
                val = self.lambda_0_array[m] * (t - last_t[m])
                for n in range(self.dimension):
                    n_jumps = [jump for jump in self._jumps[n] if last_last_t[m] <= jump < last_t[m]]
                    beta = self.cross_betas[m][n]
                    alpha = self.cross_alphas[m][n]
                    a[m][n] *= np.exp(-beta * (last_t[m] - last_last_t[m]))
                    a[m][n] += np.sum(np.exp([-beta * (last_t[m] - jump) for jump in n_jumps]))
                    n_jumps = [jump for jump in self._jumps[n] if last_t[m] <= jump < t]
                    val += alpha / beta * ((1 - np.exp(-beta * (t - last_t[m]))) * a[m][n] + np.sum(
                        1. - np.exp([-beta * (t - jump) for jump in n_jumps])))
                last_tau[m] += val
                taus[m].append(last_tau[m])

                last_last_t[m] = last_t[m]
                last_t[m] = t
        import scipy.stats as stats
        plt.figure(figsize=(15, 10))
        stats.probplot(np.diff(taus[0]), dist='expon', plot=plt, fit=True)
        stats.probplot(np.diff(taus[1]), dist='expon', plot=plt, fit=True)

    @staticmethod
    def log_likelihood(jump_times, m, dimension, T):
        """

        :param jump_times: jump times for every component
        :param m: int
        mth hawkes component

        :param dimension: max component
        :param T: end of time interval observation
        :return: log likelihood function
        """
        m_jump_times = jump_times[m]

        def likelihood(x):
            lambda_0 = x[0]
            cross_alphas_line = x[1:dimension + 1]
            cross_betas_line = x[-dimension:]
            r = np.zeros(dimension)
            prev_jump = m_jump_times[0]
            val = np.log(lambda_0)
            val -= np.sum(cross_alphas_line / cross_betas_line * (1 - np.exp(-cross_betas_line * (T - prev_jump))))
            for jump in m_jump_times[1:]:
                for n in range(dimension):
                    if n == m:
                        r[n] += 1.
                        r[n] *= np.exp(-cross_betas_line[n] * (jump - prev_jump))
                    else:
                        r[n] *= np.exp(-cross_betas_line[n] * (jump - prev_jump))
                        n_jumps = [n_jump for n_jump in jump_times[n] if prev_jump <= n_jump < jump]
                        r[n] += np.sum(np.exp([-cross_betas_line[n] * (jump - n_jump) for n_jump in n_jumps]))
                val += np.log(lambda_0 + np.dot(cross_alphas_line, r))
                val -= np.sum(cross_alphas_line / cross_betas_line * (1 - np.exp(-cross_betas_line * (T - jump))))
                prev_jump = jump
            val += T * (1 - lambda_0)
            return -val

        return likelihood

    ########################################
    #             Fitting method           #
    ########################################

    @classmethod
    def fit(cls, jumps, T, boundary=None):
        from scipy.optimize import minimize
        dimension = len(jumps)
        boundaries = [(0, boundary) for _ in range(2 * dimension + 1)]

        likelihoods = [cls.log_likelihood(jumps, i, dimension, T) for i in range(dimension)]
        fits = [minimize(likelihood,
                         x0=[1. for _ in range(2 * dimension + 1)],
                         bounds=boundaries) for likelihood in likelihoods]
        params = [fit.x for fit in fits]
        return params

        fitted_hawkes = cls(lambda_0, 0., alpha, beta)
        fitted_hawkes._jumps = jumps
        return fitted_hawkes
