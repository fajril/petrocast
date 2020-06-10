import numpy as np
from scipy import optimize

from tqdm import tqdm

from petrocast.models import crmodel

class ICRMRegression():
    """ Integrated Capacitance-Resistive Model Regression
    ---

    Based on Nguyen (2012).
    Inherent assumptions that MUST be satisfied:
    1. No Aquifer
    
    Parameters
    ---
    rate : list
        production rate
    bhp : list
        bottom-hole pressure
    cumprod : list
        cumulative production
    """
    def __init__(self, rate: list, bhp :list, cumprod: list):
        self._rate = np.asarray(rate)
        self._bhp = np.asarray(bhp)
        self._cumprod = np.asarray(cumprod)

    def _bootstrap(self, sample=1000, seed=None, pres_init_guess: float=1000,
            tau_guess: float=1, pvct_guess=1000):
        prod_data = np.stack((self._rate, self._bhp, self._cumprod), axis=1)
        row = prod_data.shape[0]
        np.random.seed(seed)
        idx = np.random.randint(0, row, (sample, row))
        prod_sample = prod_data[idx]
        bootstrap_result = [self._fit_data(prod_sample[s, :, 0], prod_sample[s, :, 1],
                                prod_sample[s, :, 2], pres_init_guess, tau_guess, pvct_guess)
                            for s in tqdm(range(sample))]
        pres_init = np.asarray([bootstrap_result[i].x[0] for i in range(sample)])
        tau = np.asarray([bootstrap_result[i].x[1] for i in range(sample)])
        pvct = np.asarray([bootstrap_result[i].x[2] for i in range(sample)])
        return pres_init, tau, pvct

    def _fit_data(self, rate: list, bhp: list, cumprod: list,
            pres_init_guess: float=1000, tau_guess: float=1, pvct_guess: float=100):
        rate = np.asarray(rate)
        bhp = np.asarray(bhp)
        cumprod = np.asarray(cumprod)
        init = np.asarray([pres_init_guess, tau_guess, pvct_guess])
        result = optimize.least_squares(
            lambda x: cumprod - crmodel.icrm(rate, bhp, x[0], x[1], x[2]), init)
        return result

    def fit(self, pres_init_guess: float=1000, tau_guess: float=1, pvct_guess: float=100):
        """ Fit the production data with ICRM model.
        Parameters
        ---
        pres_init_guess : float
            initial guess for initial reservoir pressure [P]
        tau_guess : float
            initial guess for time constant [T]
        pvct_guess : float
            initial guess for pore volume total compressibility [V/P]
        """
        result = self._fit_data(
            self._rate, self._bhp, self._cumprod, pres_init_guess, tau_guess, pvct_guess)
        self._pres_init = result.x[0]
        self._tau = result.x[1]
        self._pvct = result.x[2]

    def dynamic_pore_volume(self, compr_total: float=1E-6):
        return self._pvct/compr_total

    def productivity_index(self):
        return self._pvct/self._tau