import numpy as np
from bisect import bisect_left
import matplotlib.pyplot as plt


class HawkesProcess(object):
    def __init__(self, lambda_0, t_0, alpha, beta, seed=1):
        """
        Almost pure python class aiming at simulating, fitting,
        checking a linear univariate Hawkes process. It relies on numpy for
        numerical functions, randomness and visualisation.

        :param lambda_0: initial intensity for the process
        :param t_0: starting time of the process, should be 0.
        :type alpha: size of jumps in the intensity
        :type beta: decay influence of past jumps parameter

        :return: returns a list of jumps via simulate
        """
        ########################################
        #     Random state initialisation      #
        ########################################
        np.random.seed(seed)

        ########################################
        #       Constants  initialisation      #
        ########################################

        self.lambda_0 = lambda_0
        self.t_0 = t_0
        self.alpha = alpha
        self.beta = beta

        ########################################
        #       Variables  initialisation      #
        ########################################

        self._lamb = lambda_0
        self._t = t_0
        self._s = t_0
        self._u = t_0
        self._lambda_last_jump = self._lamb
        self._last_jump = t_0
        self._last_accepted = True
        self._is_initial = True

        ########################################
        #        Results  initialisation       #
        ########################################

        self.check = []
        self._jumps = []
        self._lambdas = [self.lambda_0]
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

    def lambda_fun(self):
        """
        Compute the jump intensity at current point
        """
        lambda_part = self._lamb - self.lambda_0
        lambda_part *= np.exp(- self.beta * (self._s - self._u))
        res = self.lambda_0 + lambda_part
        self.check.append((self._s, res))
        return res

    ########################################
    #     Routine step nb 1 : initialise   #
    ########################################

    def _first_jump(self, T):
        self._s = self.exp(self._lamb)
        self._finished = self._s > T
        self._is_initial = False
        if not self._finished:
            self._t = self._s
            self._last_jump = self._t
            self._jumps.append(self._s)

    ########################################
    #      Routine step nb 3 : continue    #
    ########################################

    def _accepted_routine(self):
        """
        If the last jump proposal has been accepted, add it to the
        accepted jumps.
        """
        self._t = self._s
        self._jumps.append(self._t)
        self._last_jump = self._t

    def _continue_routine(self):
        if self._last_accepted:
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
        if self._last_accepted:
            self._lamb += self.alpha
            self.check.append((self._s + 1e-6, self._lamb))
        self._u = self._s
        self._s += self.exp(self._lamb)

        if self._s > T:
            self._finished = True

        if not self._finished:
            temp_lamb = self.lambda_fun()
            self._last_accepted = self.u() <= temp_lamb / self._lamb
            self._lamb = temp_lamb
            self._continue_routine()

    ########################################
    #    Routine : launch the simulation   #
    ########################################

    def simulate(self, T):
        """
        Launch the routine until you're done
        """
        self._first_jump(T)
        while not self._finished:
            self._routine(T)

        result = self._jumps[:]
        return result

    ########################################
    #        Reinitialisation method       #
    ########################################

    def _clear(self, seed=None):
        """
        Reinitialise the process.
        """
        if seed is not None:
            self = self.__init__(self.lambda_0,
                                 self.t_0,
                                 self.alpha,
                                 self.beta,
                                 seed)
        else:
            self = self.__init__(self.lambda_0,
                                 self.t_0,
                                 self.alpha,
                                 self.beta)

    ########################################
    #         Visualisation methods        #
    ########################################

    def instant_lambda(self, x):
        """
        This is to compute the value of the intensity at any moment
        given that the jumps have already been simulated.

        :param x: point to evaluate
        :return: value of the intensity
        """
        t = self._jumps
        idx = bisect_left(t, x)
        d = np.array(x - t[0:idx])
        reg = self.alpha * np.sum(np.exp(-self.beta * d))
        return self.lambda_0 + reg

    def series_lambda(self, series):
        """
        Vectorial version of instant_lambda

        :param series: iterable on which to evaluate the intensity
        :return: list of intensity values

        """
        return [self.instant_lambda(each) for each in series]

    ########################################
    #         Coherence test method        #
    ########################################

    def test_jumps_distribution(self):
        """
        The following algorithm is a transformation of the jump times so as to
        get Exp(1.) distributed interval times.
        The probplot at the end is there so we can check it on a QQplot
        """
        a = 0
        last_t = 0.
        last_last_t = 0.
        last_tau = 0.

        tau = [last_tau]
        for t in self._jumps:
            a = 1. + np.exp(-self.beta * (last_t - last_last_t)) * a
            last_last_t = last_t

            val = self.lambda_0 * (t - last_t)
            val += self.alpha / self.beta * (1. - np.exp(-self.beta * (t - last_t))) * a

            last_tau += val
            tau.append(last_tau)
            last_t = t

        import scipy.stats as stats
        plt.figure(figsize=(15, 10))
        stats.probplot(np.diff(tau), dist='expon', plot=plt, fit=True)

    ########################################
    #      Log likelihood for fitting      #
    ########################################

    @staticmethod
    def log_likelihood(jump_times):
        """
        Return the log likelihood function given some jumps
        """
        last_jump = jump_times[-1]

        def likelihood(x):
            lambda_0, alpha, beta = x
            r = 0.
            prev_jump = jump_times[0]
            val = np.log(lambda_0)
            val -= alpha / beta * (1 - np.exp(-beta * (last_jump - prev_jump)))
            for jump in jump_times[1:]:
                r += 1.
                r *= np.exp(-beta * (jump - prev_jump))
                val += np.log(lambda_0 + alpha * r)
                val -= alpha / beta * (1. - np.exp(-beta * (last_jump - jump)))
                prev_jump = jump

            return -(val + last_jump * (1. - lambda_0))

        return likelihood

    ########################################
    #             Fitting method           #
    ########################################

    @classmethod
    def fit(cls, jumps):
        from scipy.optimize import minimize

        boundaries = ((0, None), (0, None), (0, None))

        likelihood = cls.log_likelihood(jumps)
        fit = minimize(likelihood, x0=[1., 1., 1.], bounds=boundaries)
        lambda_0, alpha, beta = fit.x

        fitted_hawkes = cls(lambda_0, 0., alpha, beta)
        fitted_hawkes._jumps = jumps
        return fitted_hawkes
