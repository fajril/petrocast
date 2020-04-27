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
    
    def fit(self, d_rate_init: float=1E-5, b_exp_init: float=0.5, verbose: int=0, alpha: float=0.2):
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
        alpha : float
            significant level.
            Default value is 0.2 since PRMS 2018 is using P90 to P10 percentile
        """
        

        if verbose > 0:
            print("start fitting exponential model:\n")
        exp_har_init = np.asarray([self._rate[0], d_rate_init])
        exp_result = optimize.least_squares(lambda x: \
            self._rate - self._exp_rate(self._cumprod, x[0], x[1]), \
            exp_har_init, verbose=verbose)

        if verbose > 0:
            print("\nstart fitting harmonic model:\n")
        har_result = optimize.least_squares(lambda x: \
            self._rate - self._har_rate(self._cumprod, x[0], x[1]), \
            exp_har_init, verbose=verbose)

        # Use initial rate and decline rate from exponential model to improve the algorithm
        if verbose > 0:
            print("\nstart fitting hyperbolic model:\n")
        hyp_init = np.asarray([self._rate[0], d_rate_init, b_exp_init])
        hyp_result = optimize.least_squares(lambda x: \
            self._rate - self._hyp_rate(self._cumprod, x[0], x[1], x[2]), \
            hyp_init, verbose=verbose, bounds=([1E-3, 1E-9, 1E-10], [np.inf, np.inf, 1 - 1E-10]))
 
        self._rate_init = {'exp':exp_result.x[0], 'har':har_result.x[0], 'hyp':hyp_result.x[0]}
        self._d_rate = {'exp':exp_result.x[1], 'har':har_result.x[1], 'hyp':hyp_result.x[1]}
        self._b_exp = {'exp':0, 'har':1, 'hyp':hyp_result.x[2]}
        self._cost_val = {'exp':exp_result.cost, 'har':har_result.cost, 'hyp':hyp_result.cost}
        self._message = {'exp':exp_result.message, 'har':har_result.message, 'hyp':hyp_result.message}
        self._status = {'exp':exp_result.status, 'har':har_result.status, 'hyp':hyp_result.status}

        # calculate confidence interval
        # -----------------------------

        ssr_exp = np.sum(np.power( \
            self._rate - self._exp_rate(self._cumprod, self._rate_init['exp'], self._d_rate['exp']), 2))
        ssr_har = np.sum(np.power(\
            self._rate - self._har_rate(self._cumprod, self._rate_init['har'], self._d_rate['har']), 2))
        ssr_hyp = np.sum(np.power(\
            self._rate - self._hyp_rate(self._cumprod, self._rate_init['hyp'], self._d_rate['hyp'], self._b_exp['hyp']), 2))

        dof = max(0, len(self._cumprod) - 2) # 2 parameters: initial rate and decline rate
        dof_hyp = max(0, len(self._cumprod) - 3) # 3 parameters: initial rate, decline rate, and b exponent
        
        sigma_exp = np.sqrt(ssr_exp / dof)
        sigma_har = np.sqrt(ssr_har / dof)
        sigma_hyp = np.sqrt(ssr_hyp / dof_hyp)

        t_val = t.ppf(1 - alpha/2, dof)
        t_val_hyp = t.ppf(1 - alpha/2, dof_hyp)

        self._ci = {'exp':t_val*sigma_exp,
                   'har':t_val*sigma_har,
                   'hyp':t_val_hyp*sigma_hyp}

        self._best = min(self._cost_val, key=self._cost_val.get)
        print(f'Best fitting model: {self._MODEL_NAME[self._best]}')
        logging.debug(f'Best Fitting \n', \
                        'method: {self._best}, cost value: {self._cost_val[self._best]}')

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

    def predict_eur(self, rate_eur: float=0, model: str='best'):
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
        self._eur2p = {'exp':self._exp_cum(rate_eur, self._rate_init['exp'], self._d_rate['exp']),
                       'har':self._har_cum(rate_eur, self._rate_init['har'], self._d_rate['har']),
                       'hyp':self._hyp_cum(rate_eur, self._rate_init['hyp'], self._d_rate['hyp'], self._b_exp['hyp'])}
 
        self._eur1p = {'exp':self._eur2p['exp'] - self._ci['exp'],
                       'har':self._eur2p['har'] - self._ci['har'],
                       'hyp':self._eur2p['hyp'] - self._ci['hyp']}

        self._eur3p = {'exp':self._eur2p['exp'] + self._ci['exp'],
                       'har':self._eur2p['har'] + self._ci['har'],
                       'hyp':self._eur2p['hyp'] + self._ci['hyp']}

        
        if model == 'best':
            return self._eur1p[self._best], self._eur2p[self._best], self._eur3p[self._best]
        else:
            return self._eur1p[model], self._eur2p[model], self._eur3p[model]

    def plot(self, plot_type: str='cartesian', model: str='best'):
        """
        Plot decline curve model.

        Parameters
        ---
        plot_type : str
            'cartesian', 'semilog', 'loglog'
        model : str
            exp: Exponential curve
            hyp: Hyperbolic curve
            har: Harmonic curve
            best: use the best fit curve
            all: use all curve
        """
        if model == 'best':
            selected_model = self._best
        else:
            selected_model = model

        cumprod_frcast_1p = np.linspace(self._cumprod[-1], self._eur1p[selected_model])
        cumprod_frcast_2p = np.linspace(self._cumprod[-1], self._eur2p[selected_model])
        cumprod_frcast_3p = np.linspace(self._cumprod[-1], self._eur3p[selected_model])
        _, ax = plt.subplots()
        ax.scatter(self._cumprod, self._rate)
        ax.set_title(f"Arps Decline Model\nSelected Method: {self._MODEL_NAME[selected_model]}")

        func_rate = {'exp':self._exp_rate,
                     'har':self._har_rate,
                     'hyp':partial(self._hyp_rate, b_exp=self._b_exp['hyp'])}

        ax.plot(self._cumprod, \
            func_rate[selected_model](self._cumprod, self._rate_init[selected_model], \
            self._d_rate[selected_model]))
        ax.plot(cumprod_frcast_2p, \
            func_rate[selected_model](cumprod_frcast_2p, self._rate_init[selected_model], \
            self._d_rate[selected_model]), 'k--')
        ax.plot(cumprod_frcast_3p, \
            func_rate[selected_model](cumprod_frcast_3p, self._rate_init[selected_model], \
            self._d_rate[selected_model]) + self._ci[selected_model], 'k--')
        ax.plot(cumprod_frcast_1p, \
            func_rate[selected_model](cumprod_frcast_1p, self._rate_init[selected_model], \
            self._d_rate[selected_model]) - self._ci[selected_model], 'k--')

        ax.grid(True, which='major', axis='y')
        ax.set_ylim(bottom=1)
        #ax.set_yscale('log')

    def summary(self):
        """
        Provide summary of fitting calculations.
        """
        print(f'Exponential message: {self._message["exp"]}')
        print(f'Harmonic message: {self._message["har"]}')
        print(f'Hyperbolic message: {self._message["hyp"]}')

        df = pd.DataFrame([self._rate_init, self._d_rate, self._b_exp,
                           self._cost_val, self._status, self._ci])
        df = df.rename(index={0:'initial rate', 1:'decline rate', 2:'b exponent',
                              3:'cost value', 4:'status', 5:'ci'})
        df = df.rename(columns=self._MODEL_NAME)
        return df
        
