import numpy as np
import numba as nb

spec = [
    ('lambda_0', nb.float64), ('t_0', nb.float64), ('alpha', nb.float64), ('beta', nb.float64), ('seed', nb.int64),
    ('maxsize', nb.int64),

    ('lamb', nb.float64), ('t', nb.float64), ('s', nb.float64), ('u', nb.float64),
    ('lambda_last_jump', nb.float64), ('last_jump', nb.float64), ('last_accepted', nb.boolean),
    ('is_initial', nb.boolean), ('nb_jumps', nb.int64),
    ('jumps', nb.float64[:]), ('finished', nb.boolean)
]


@nb.jitclass(spec)
class NumbaHawkesProcess(object):
    """
    This classes aims at replicating the algorithm exposed in the pure python
    HawkesProcess class above in a numba fashion.
    It will only implement the simulation in itself as all external modules but numpy
    can't be used.
    The signature of the class has to be specified (spec above) and all
    arrays have to be instantiated with a specific length (maxsize).

    """

    def __init__(self, lambda_0, t_0, alpha, beta, seed, maxsize):
        np.random.seed(seed)
        self.lambda_0 = lambda_0
        self.t_0 = t_0
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.maxsize = maxsize

        self.lamb = self.lambda_0
        self.t = t_0
        self.s = t_0
        self.u = t_0
        self.lambda_last_jump = self.lamb
        self.last_jump = t_0
        self.last_accepted = True
        self.is_initial = True
        self.nb_jumps = 0

        self.jumps = np.empty(maxsize, dtype=np.float64)
        self.finished = False

    @staticmethod
    def uniform():
        return np.random.uniform(0., 1.)

    @staticmethod
    def exp(lamb):
        return np.random.exponential(1 / lamb)

    def _first_jump(self, T):

        self.s = self.exp(self.lamb)
        self.finished = self.s > T
        self.is_initial = False
        if not self.finished:
            self.t = self.s
            self.last_jump = self.t
            self.jumps[self.nb_jumps] = self.s
            self.nb_jumps += 1

    def _accepted_routine(self):
        self.t = self.s
        self.jumps[self.nb_jumps] = self.t
        self.nb_jumps += 1
        self.last_jump = self.t

    def _continue_routine(self):
        if self.last_accepted:
            self._accepted_routine()

    def lambda_fun(self):
        lambda_part = self.lamb - self.lambda_0
        lambda_part *= np.exp(-self.beta * (self.s - self.u))
        res = self.lambda_0 + lambda_part
        return res

    def _routine(self, T):
        if self.last_accepted:
            self.lamb += self.alpha
        self.u = self.s
        self.s += self.exp(self.lamb)

        if self.s > T:
            self.finished = True

        if not self.finished:
            temp_lamb = self.lambda_fun()
            self.last_accepted = self.uniform() <= temp_lamb / self.lamb
            self.lamb = temp_lamb
            self._continue_routine()

    def _clear(self):
        self = self.__init__(self.lambda_0,
                             self.t_0,
                             self.alpha,
                             self.beta,
                             self.seed,
                             self.maxsize)

    def simulate(self, T):
        self._first_jump(T)
        while not self.finished:
            self._routine(T)

        result = self.jumps.copy()
        return result

