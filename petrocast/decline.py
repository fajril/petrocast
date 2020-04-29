import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats.distributions import t

from functools import partial

import matplotlib.pyplot as plt
import logging

class ArpsDecline():
    """ 
    Arps Decline Curve Analysis
    ===

    Based on Arps Equation (1945).
    Inherent assumptions that MUST be satisfied:
    1. The well is draining a constant drainage area, that is,
       the well is in a boundary-dominated flow condition.
    2. The well is produced at or near capacity.
    3. The well is produced at a constant bottom-hole pressure.

    There are three types of curve: Exponential, Hyperbolic,
    and Harmonic Curve. 

    Parameters
    ---
    cumprod : list
        Cumulative production
    rate : list, numpy array
        Production rate
    """
    def __init__(self, cumprod: list, rate: list):
        self._best_curve = ''
        self._MODEL_NAME = {'exp':'exponential', 'har':'harmonic', 'hyp':'hyperbolic'}
        
        self._rate = np.asarray(rate)
        self._cumprod = np.asarray(cumprod)

        if len(cumprod) != len(rate):
            raise IndexError("cumulative production and rate must have the same length")
    
    def _exp_rate(self, cumprod: list, rate_init: float, d_rate: float):
        cumprod = np.asarray(cumprod)
        return rate_init - d_rate * cumprod

    def _jac_exp_rate(self, cumprod: list, rate_init: float, d_rate: float):
        return np.array([np.ones(len(cumprod)), -cumprod])
    
    def _exp_cum(self, rate: list, rate_init: float, d_rate: float):
        rate = np.asarray(rate)
        return (rate_init - rate)/d_rate

    def _har_rate(self, cumprod: list, rate_init: float, d_rate: float):
        cumprod = np.asarray(cumprod)
        return np.exp(np.log(rate_init) - d_rate * cumprod / rate_init)
    
    def _har_cum(self, rate: list, rate_init: float, d_rate: float):
        rate = np.asarray(rate)
        return np.log(rate_init/rate)*rate_init/d_rate

    def _hyp_rate(self, cumprod: list, rate_init: float, d_rate: float, b_exp: float):
        cumprod = np.asarray(cumprod)
        const = rate_init/(d_rate * (1 - b_exp))
        return np.exp(np.log(1 - cumprod/const)/(1 - b_exp) + np.log(rate_init))

    def _hyp_cum(self, rate: list, rate_init: float, d_rate: float, b_exp: float):
        rate = np.asarray(rate)
        const = rate_init/(d_rate * (1 - b_exp))
        return const * (1 - np.power(rate/rate_init, (1 - b_exp)))
    
    def _fit_data(self, cumprod, rate, d_rate_init: float=1E-5, b_exp_init: float=0.5, 
                  verbose: int=0):
        if verbose > 0:
            print("start fitting exponential model:\n")
        exp_har_init = np.asarray([rate[0], d_rate_init])
        exp_result = optimize.least_squares(lambda x: \
            rate - self._exp_rate(cumprod, x[0], x[1]), \
            exp_har_init, verbose=verbose)

        if verbose > 0:
            print("\nstart fitting harmonic model:\n")
        har_result = optimize.least_squares(lambda x: \
            rate - self._har_rate(cumprod, x[0], x[1]), \
            exp_har_init, verbose=verbose)

        # Use initial rate and decline rate from exponential model to improve the algorithm
        if verbose > 0:
            print("\nstart fitting hyperbolic model:\n")
        hyp_init = np.asarray([rate[0], d_rate_init, b_exp_init])
        hyp_result = optimize.least_squares(lambda x: \
            rate - self._hyp_rate(cumprod, x[0], x[1], x[2]), \
            hyp_init, verbose=verbose, bounds=([1E-3, 1E-9, 1E-4], [np.inf, np.inf, 1 - 1E-4]))
        
        return exp_result, har_result, hyp_result
    
    def _boostrap(self,sample=1000, seed=None, d_rate_init: float=1E-5, b_exp_init: float=0.5):
        prod_data = np.stack((self._cumprod, self._rate), axis=1)
        idx = np.random.randint(0, len(self._cumprod), size=(sample, prod_data.shape[0]))
        rate_init_exp, rate_init_har, rate_init_hyp = [], [], []
        d_rate_exp, d_rate_har, d_rate_hyp = [], [], []
        b_exp_hyp = []
        for s in range(sample):
            cumprod, rate = prod_data[idx[s, :], 0], prod_data[idx[s, :], 1]
            exp, har, hyp = self._fit_data(cumprod, rate)
            rate_init_exp.append(exp.x[0])
            rate_init_har.append(har.x[0])
            rate_init_hyp.append(hyp.x[0])
            d_rate_exp.append(exp.x[1])
            d_rate_har.append(har.x[1])
            d_rate_hyp.append(hyp.x[1])
            b_exp_hyp.append(hyp.x[2])


        rate_init = {'exp':rate_init_exp, 'har':rate_init_har, 'hyp':rate_init_hyp}
        d_rate = {'exp':d_rate_exp, 'har':d_rate_har, 'hyp':d_rate_hyp}
        b_exp = {'exp':[0]*sample, 'har':[1]*sample, 'hyp':b_exp_hyp}
        return rate_init, d_rate, b_exp
            

    def fit(self,  d_rate_init: float=1E-5, b_exp_init: float=0.5,
            verbose: int=0, sig: float=0.2):
        """ 
        Fit the production model with the curves.
        This method does not provide time based forecast to prevent
        wrong fitting.

        Parameters
        ---
        d_rate_init : float
            initial guess for decline rate (D)
        b_exp_init : float
            Initial guess for decline exponent (b)
        verbose : int
            0: work silently
            1: display a termination report
            2: display progress during iterations
        sig : float
            significant level.
            Default value is 0.2 since PRMS 2018 is using P90 to P10 percentile
        """
        
        exp_result, har_result, hyp_result = self._fit_data(self._cumprod, self._rate,
                                                            d_rate_init=d_rate_init,
                                                            b_exp_init=b_exp_init,
                                                            verbose=verbose)
        
        self._rate_init = {'exp':exp_result.x[0], 'har':har_result.x[0], 'hyp':hyp_result.x[0]}
        self._d_rate = {'exp':exp_result.x[1], 'har':har_result.x[1], 'hyp':hyp_result.x[1]}
        self._b_exp = {'exp':0, 'har':1, 'hyp':hyp_result.x[2]}
        self._cost_val = {'exp':exp_result.cost, 'har':har_result.cost, 'hyp':hyp_result.cost}
        self._message = {'exp':exp_result.message, 'har':har_result.message, 'hyp':hyp_result.message}
        self._status = {'exp':exp_result.status, 'har':har_result.status, 'hyp':hyp_result.status}

        self._best = min(self._cost_val, key=self._cost_val.get)
        
        print(f'Best fitting model: {self._MODEL_NAME[self._best]}')
        logging.debug(f'Best fitting \n', \
                        'method: {self._best}, cost value: {self._cost_val[self._best]}')
        logging.debug(f'Selected model \n', \
                        'method: {selected_model}, cost value: {self._cost_val[self._best]}')
        

    def predict_rate(self, cumprod: list, selected_curve: str='auto'):
        """
        Generate forecast based on specified curve.

        Parameters
        ---
        cumprod : float
            cumulative production
        selected_curve : str
            exp: Exponential curve
            hyp: Hyperbolic curve
            har: Harmonic curve
            auto: use the best fit curve
            all: use all curve
        
        Returns
        ---
        rate : np.array
        """
        pass

    def predict_eur(self, rate_eur: float=0, model: str='best', sample: int=1000, seed=None):
        """
        Calculate Estimated Ultimate Recovery (EUR)
        based on specified curve.

        Parameters
        ---
        rate_eur : float, list, ndarray
            Usually assigned with minimum economic or technical rate.

        selected_curve : str
            exp: Exponential curve
            hyp: Hyperbolic curve
            har: Harmonic curve
            best: use the best fit curve
        
        Returns
        ---
        eur : float
            return EUR from selected curve.
        """
        self._rate_eur = rate_eur
        if model == 'best':
            self._model = self._best
        else:
            self._model = model
        
        func_cum = {'exp':self._exp_cum,
                    'har':self._har_cum,
                    'hyp':partial(self._hyp_cum, b_exp=self._b_exp['hyp'])}
        
        self._eur2p = func_cum[self._model](rate_eur, \
                                               self._rate_init[self._model], \
                                               self._d_rate[self._model])

        rate_init, d_rate, b_exp = self._boostrap(sample=sample, seed=seed)
        self._cumprod_frcast = np.linspace(self._cumprod[-1], 2 * self._eur2p, 1000)
        self._rate_frcast_all = np.zeros(len(self._cumprod_frcast))
        for s in range(sample):
            func_bootstrap = {'exp':self._exp_rate,
                              'har':self._har_rate,
                              'hyp':partial(self._hyp_rate, b_exp=b_exp['hyp'][s])}
            rate_frcast = func_bootstrap[self._model](self._cumprod_frcast, \
                                                      rate_init[self._model][s], \
                                                      d_rate[self._model][s])
            self._rate_frcast_all = np.vstack((self._rate_frcast_all, rate_frcast))
        self._rate_frcast_all = np.delete(self._rate_frcast_all, 0, 0)

        self._rate_frcast_1p = np.percentile(self._rate_frcast_all, 10, axis=0)
        self._idx_1p = sum(self._rate_frcast_1p >= rate_eur)
        self._rate_frcast_2p = np.percentile(self._rate_frcast_all, 50, axis=0)
        self._idx_2p = sum(self._rate_frcast_2p >= rate_eur)
        self._rate_frcast_3p = np.percentile(self._rate_frcast_all, 90, axis=0)
        self._idx_3p = sum(self._rate_frcast_3p >= rate_eur)
        print(self._idx_3p)
        #print(self._rate_frcast_3p)

        return self._cumprod_frcast[self._idx_1p], self._cumprod_frcast[self._idx_2p], self._cumprod_frcast[self._idx_3p]

    def plot(self, plot_type: str='cartesian', figsize=(12, 8)):
        """
        Plot decline curve model.

        Parameters
        ---
        plot_type : str
            'cartesian', 'semilog', 'loglog'
        """
        _, ax = plt.subplots(figsize=figsize)
        ax.scatter(self._cumprod, self._rate, color='grey')
        ax.set_title(f"Arps Decline Model\nSelected Method: {self._MODEL_NAME[self._model]}")

        func_rate = {'exp':self._exp_rate,
                     'har':self._har_rate,
                     'hyp':partial(self._hyp_rate, b_exp=self._b_exp['hyp'])}

        ax.plot(self._cumprod, \
            func_rate[self._model](self._cumprod, self._rate_init[self._model], \
            self._d_rate[self._model]), color='grey')
        for plot in range(self._rate_frcast_all.shape[0]):
            idx = sum(self._rate_frcast_all[plot, :] >= self._rate_eur)
            ax.plot(self._cumprod_frcast[:idx], self._rate_frcast_all[plot, :idx], color='orange', alpha=0.02)
        
        ax.plot(self._cumprod_frcast[:self._idx_1p], self._rate_frcast_1p[:self._idx_1p], color='red', linestyle='--')
        ax.plot(self._cumprod_frcast[:self._idx_2p], self._rate_frcast_2p[:self._idx_2p], color='red', linestyle='--')
        ax.plot(self._cumprod_frcast[:self._idx_3p], self._rate_frcast_3p[:self._idx_3p], color='red', linestyle='--')

        ax.axhline(self._rate_eur, color='k', linestyle='dotted')
        ax.grid(True, which='major', axis='y')
        ax.set_ylim(bottom=1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        #ax.set_yscale('log')

    def summary(self):
        """
        Provide summary of fitting calculations.
        """
        print(f'Exponential message: {self._message["exp"]}')
        print(f'Harmonic message: {self._message["har"]}')
        print(f'Hyperbolic message: {self._message["hyp"]}')

        df = pd.DataFrame([self._rate_init, self._d_rate, self._b_exp,
                           self._cost_val, self._status])
        df = df.rename(index={0:'initial rate', 1:'decline rate', 2:'b exponent',
                              3:'cost value', 4:'status', 5:'ci'})
        df = df.rename(columns=self._MODEL_NAME)
        return df
        
