import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats.distributions import norm
from scipy.stats import probplot

from functools import partial

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from tqdm import tqdm
import logging

from petrocast.models import arpsmodel

class ArpsRegression():
    """ Arps Decline Curve Analysis
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
    time_offset : float
        time offset for first data point
    """
    def __init__(self, cumprod: list, rate: list, time_offset: float=0):
        self._MODEL_NAME = {'exp':'exponential', 'har':'harmonic', 'hyp':'hyperbolic'}
        
        self._rate = np.asarray(rate)
        self._cumprod = np.asarray(cumprod)
        self._time_offset = np.asarray(time_offset)

        if len(cumprod) != len(rate):
            raise IndexError("cumulative production and rate must have the same length")
    
    def _fit_exp(self, cumprod, rate, d_rate_init: float=1E-5):
        init = np.asarray([rate[0], d_rate_init])
        result = optimize.least_squares(lambda x: \
            rate - arpsmodel.exp_rate_cumprod(cumprod, x[0], x[1]), init, verbose=self._verbose)
        return result

    def _fit_har(self, cumprod, rate, d_rate_init: float=1E-5):
        init = np.asarray([rate[0], d_rate_init])
        result = optimize.least_squares(lambda x: 
            rate - arpsmodel.har_rate_cumprod(cumprod, x[0], x[1]), init, verbose=self._verbose)
        return result
    
    def _fit_hyp(self, cumprod, rate, d_rate_init: float=1E-5, b_exp_init: float=0.5):
        init = np.asarray([rate[0], d_rate_init, b_exp_init])
        rate
        result = optimize.least_squares(lambda x: \
            rate - arpsmodel.hyp_rate_cumprod(cumprod, x[0], x[1], x[2]), \
            init, verbose=self._verbose, bounds=([1E-3, 1E-9, 1E-10], [np.inf, np.inf, 1 - 1E-10]))
        return result

    def _fit_data(self, cumprod, rate, model: str='best', d_rate_init: float=1E-5, b_exp_init: float=0.5, 
                  verbose: int=0):
        if model == 'best':
            exp_result = self._fit_exp(cumprod, rate, d_rate_init)
            har_result = self._fit_har(cumprod, rate, d_rate_init)
            hyp_result = self._fit_hyp(cumprod, rate, d_rate_init, b_exp_init)
            cost = {'exp':exp_result.cost, 'har':har_result.cost, 'hyp':hyp_result.cost}
            all_result = {'exp':exp_result, 'har':har_result, 'hyp':hyp_result}
            self._best = min(cost, key=cost.get)
            self._model = self._best
            result = all_result[self._model]
        else:
            func_fit = {'exp':self._fit_exp,
                        'har':self._fit_har,
                        'hyp':partial(self._fit_hyp, b_exp_init=b_exp_init)}
            self._model = model
            result = func_fit[model](cumprod, rate, d_rate_init)
        if verbose > 0:
                print(f'Best fitting model: {self._MODEL_NAME[self._best]}')
                print(f'Selected model: {self._MODEL_NAME[self._model]}')
        return result
    
    def _bootstrap(self, model, sample=1000, seed=None, d_rate_init: float=1E-5, b_exp_init: float=0.5):
        prod_data = np.stack((self._cumprod, self._rate), axis=1)
        row = prod_data.shape[0]
        np.random.seed(seed)
        idx = np.random.randint(0, row, (sample, row))
        prod_sample = prod_data[idx]
        bootstrap_result = [self._fit_data(
                                prod_sample[s, :, 0], prod_sample[s, :, 1], model,
                                d_rate_init, b_exp_init)
                            for s in tqdm(range(sample))]
        rate_init = np.asarray([bootstrap_result[i].x[0] for i in range(sample)])
        d_rate = np.asarray([bootstrap_result[i].x[1] for i in range(sample)])
        if model == 'exp':
            b_exp = np.zeros(sample)
        elif model == 'har':
            b_exp = np.ones(sample)
        else:
            b_exp = np.asarray([bootstrap_result[i].x[2] for i in range(sample)])
        return rate_init, d_rate, b_exp
    
    def _uncertainty_forecast(self, rate_forecast, sample: int=1000, seed=None):
        # Calculate uncertainty in time and cumulative production.
        rate_init, d_rate, b_exp = self._bootstrap(self._model, sample=sample, seed=seed)
        if self._model == 'exp':
            cumprod_forecast = np.asarray([arpsmodel.exp_cumprod_rate(rate_forecast, rate_init[s], d_rate[s])
                                            for s in range(sample)])
            time_forecast = np.asarray([arpsmodel.exp_time_rate(rate_forecast, rate_init[s], d_rate[s])
                                        for s in range(sample)])
        elif self._model == 'har':
            cumprod_forecast = np.asarray([arpsmodel.har_cumprod_rate(rate_forecast, rate_init[s], d_rate[s])
                                            for s in range(sample)])
            time_forecast = np.asarray([arpsmodel.har_time_rate(rate_forecast, rate_init[s], d_rate[s])
                                        for s in range(sample)])
        else:
            cumprod_forecast = np.asarray([arpsmodel.hyp_cumprod_rate(rate_forecast, rate_init[s], 
                                                                        d_rate[s], b_exp[s])
                                            for s in range(sample)])
            time_forecast = np.asarray([arpsmodel.hyp_time_rate(rate_forecast, rate_init[s], 
                                                                d_rate[s], b_exp[s])
                                        for s in range(sample)])
        return time_forecast, cumprod_forecast

    def fit(self, model: str='best', d_rate_init: float=1E-5, b_exp_init: float=0.5,
            verbose: int=0, sig: float=0.2):
        """ Fit the production data with the curves.
        This method intentionally does not provide time axis based fitting.

        Parameters
        ---
        model : str
            model type:
            'exp': Arps' exponential model
            'har': Arps' harmonic model
            'hyp': Arps' hyperbolic model
            'best': find the best model that fit the data
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
        self._verbose = verbose
        result = self._fit_data(self._cumprod, self._rate, model, d_rate_init=d_rate_init, 
                                b_exp_init=b_exp_init, verbose=verbose)
        self._rate_init = result.x[0]
        self._d_rate = result.x[1]
        if self._model == 'exp':
            self._b_exp = 0
        elif self._model == 'har':
            self._b_exp = 1
        else:
            self._b_exp = result.x[2]
        self._cost_val = result.cost
        self._message = result.message
        self._status = result.status

        self._func_time = {'exp':arpsmodel.exp_time_rate,
                           'har':arpsmodel.har_time_rate,
                           'hyp':partial(arpsmodel.hyp_time_rate, b_exp=self._b_exp)}

        self._func_cum = {'exp':arpsmodel.exp_cumprod_rate,
                          'har':arpsmodel.har_cumprod_rate,
                          'hyp':partial(arpsmodel.hyp_cumprod_rate, b_exp=self._b_exp)}
        
        self._func_rate = {'exp':arpsmodel.exp_rate_cumprod,
                           'har':arpsmodel.har_rate_cumprod,
                           'hyp':partial(arpsmodel.hyp_rate_cumprod, b_exp=self._b_exp)}

        logging.debug(f'Best fitting \n', \
                        'method: {self._best}, cost value: {self._cost_val[self._best]}')
        logging.debug(f'Selected model \n', \
                        'method: {self._model}, cost value: {self._cost_val[self._model]}')
        
    def eur(self, rate_eur: float=1):
        """Calculate best EUR from specified rate

        Parameter
        ---
        rate_eur : float
            Minimum rate based on commercial or technical limit.
        
        Return
        ---
        eur : float
            Estimated Ultimate Recovery
        """
        self._rate_eur = rate_eur
        eur = self._func_cum[self._model](rate_eur, self._rate_init, self._d_rate)
        return eur

    def time(self, rate_eur: float=1):
        """Calculate Time from specified rate

        Parameter
        ---
        rate_eur : float
            Minimum rate based on commercial or technical limit.
        
        Return
        ---
        eur : float
            Estimated Ultimate Recovery
        """
        self._rate_eur = rate_eur
        time = self._func_time[self._model](rate_eur, self._rate_init, self._d_rate)
        return time + self._time_offset

    def forecast(self, rate_eur: float=1, num: int=50):
        """ Forecast production profile.

        Parameters
        ---
        rate_eur : float
            Minimum rate based on commercial or technical limit.
        num : int
            Number of data points

        return
        ---
        time : ndarray
            time series
        cumprod : ndarray
            cumulative production
        rate : ndarray
            production rate
        """
        rate_frcast = np.linspace(self._rate[-1], rate_eur, num)
        cumprod_frcast_mid = self._func_cum[self._model](rate_frcast, \
                                                            self._rate_init, \
                                                            self._d_rate)
        time_frcast_mid = self._func_time[self._model](rate_frcast, \
                                                           self._rate_init, \
                                                           self._d_rate)

        return time_frcast_mid, cumprod_frcast_mid, rate_frcast

    def calculate_uncertainty_prediction(self, rate_eur, sample: int=1000, seed=None):
        """ Calculate uncertainty based on specified Arps model.

        Parameters
        ---
        rate_eur : float
            Minimum rate based on commercial or technical limit.
        sample : int
            Sample size to estimate the uncertainty
        seed : int
            Random seed

        
        """
        self._rate_eur = rate_eur
        rate_forecast = np.linspace(self._rate[-1], self._rate_eur)
        self._time_mc, self._cumprod_mc = self._uncertainty_forecast(rate_forecast, sample, seed)

    def uncertainty_eur(self):
        """ Calculate low, best, high case of EUR
        According to PRMS 2018, low = P90, best = P50, high = P10

        Return
        ---
        P90 : float
            low case EUR
        P50 : float
            most likely EUR
        P10 : float
            high case EUR
        """
        P90 = np.percentile(self._cumprod_mc[:, -1], 10)
        P10 = np.percentile(self._cumprod_mc[:, -1], 90)
        return P90, self.eur(self._rate_eur), P10

    def uncertainty_time(self):
        """ Calculate low, best, high case of Time
        According to PRMS 2018, low = P90, best = P50, high = P10

        Return
        ---
        P90 : float
            low case EUR
        P50 : float
            most likely EUR
        P10 : float
            high case EUR
        """
        P90 = np.percentile(self._time_mc[:, -1], 10)
        P10 = np.percentile(self._time_mc[:, -1], 90)
        return P90 + self._time_offset, self.time(self._rate_eur), P10 + self._time_offset

    def uncertainty_histplot(self, xcumprod: bool=True, color_style: str='oil', 
                                 hidey: bool=True, figsize=(12, 8)):
        """ Plot EUR uncertainty based on specified Arps model.

        Parameters
        ---
        rate_eur : float
            Minimum rate based on commercial or technical limit.
        sample : int
            Sample size to estimate the uncertainty
        seed : int
            Random seed
        xcumprod : bool
            True for cumulative production, False for time
        color_style : str
            set 'oil' theme or 'gas' theme
        hidey : bool
            hide probability density axis
        figsize : tuple
            Figure size
        """
        plt.style.use('ggplot')
        if color_style == 'gas':
            regclr = 'brown'
            sampleclr = 'coral'
        else:
            regclr = 'darkgreen'
            sampleclr = 'limegreen'
        _, ax = plt.subplots(figsize=figsize)
        if xcumprod:
            loc, scale = norm.fit(self._cumprod_mc[:, -1])
            ax.hist(self._cumprod_mc[:, -1], bins='auto', density=True, color=sampleclr, alpha=0.5)
        else:
            loc, scale = norm.fit(self._time_mc[:, -1])
            ax.hist(self._time_mc[:, -1], bins='auto', density=True, color=sampleclr, alpha=0.5)

        dist = norm(loc=loc, scale=scale)
        xaxis = np.linspace(dist.ppf(0.0005), dist.ppf(0.9995))
        normal = ax.plot(xaxis, dist.pdf(xaxis), color=regclr)
        if xcumprod:
            P90 = np.percentile(self._cumprod_mc[:, -1], 10)
            P50 = self.eur(self._rate_eur)
            P10 = np.percentile(self._cumprod_mc[:, -1], 90)
        else:
            P90 = np.percentile(self._time_mc[:, -1], 10)
            P50 = self.time(self._rate_eur)
            P10 = np.percentile(self._time_mc[:, -1], 90)
        low = ax.axvline(P90, color=regclr, linestyle='--', alpha=0.5)
        mid = ax.axvline(P50, color=regclr, linestyle='--', alpha=0.5)
        hgh = ax.axvline(P10, color=regclr, linestyle='--', alpha=0.5)

        ax.set_xlim(xaxis[0], xaxis[-1])
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        if xcumprod:
            ax.set_title((f'EUR Distribution based on Arps Model\n'
                        f'Selected Method: {self._MODEL_NAME[self._model]}'))
            ax.set_xlabel('Estimated Ultimate Recovery')
        else:
            ax.set_title((f'Time Distribution at EUR based on Arps Model\n'
                        f'Selected Method: {self._MODEL_NAME[self._model]}'))
            ax.set_xlabel('Time')
        ax.set_ylabel('Probability Density')
        
        if hidey:
            ax.yaxis.set_ticklabels([])
        ax.legend([normal, low, mid, hgh], ('PDF',
                                            f'P90: {round(P90, 2)}', 
                                            f'P50: {round(P50, 2)}', 
                                            f'P10: {round(P10, 2)}'))
        plt.show()

    def forecast_plot(self, bootstrap: bool=False, xcumprod: bool=True, color_style: str='oil', figsize=(12, 8)):
        """ Plot decline curve model.

        Parameters
        ---
        rate_eur : float
            Minimum rate based on commercial or technical limit.
        sample : int
            Sample size to estimate the uncertainty
        seed : int
            Random seed
        xcumprod : bool
            True for cumulative production, False for time
        color_style : str
            set 'oil' theme or 'gas' theme
        figsize : tuple
            Figure size
        """
        plt.style.use('ggplot')
        if color_style == 'gas':
            histclr = 'brown'
            regclr = 'brown'
            cbandsclr = 'brown'
            sampleclr = 'coral'
        else:
            histclr = 'darkgreen'
            regclr = 'darkgreen'
            cbandsclr = 'darkgreen'
            sampleclr = 'limegreen'

        _, ax = plt.subplots(figsize=figsize)
        time, cumprod, rate = self.forecast(self._rate_eur)
        rate_forecast = np.linspace(self._rate[-1], self._rate_eur)
        if xcumprod:
            if bootstrap:
                ax.plot(self._cumprod_mc.transpose(), 
                        np.tile(rate_forecast, (self._cumprod_mc.shape[0], 1)).transpose(), 
                        color=sampleclr, alpha=0.03) # plot bootstrap result
            ax.plot(cumprod, rate, color=cbandsclr, linestyle='--') # P50 forecast
            ax.scatter(self._cumprod, self._rate, color=histclr) # data points
            
            rate = self._func_rate[self._model](self._cumprod, self._rate_init, self._d_rate)
            ax.plot(self._cumprod, rate, color=regclr) # regression line

            if bootstrap:
                cumprod = np.percentile(self._cumprod_mc, 10, axis=0)
                ax.plot(cumprod, rate_forecast, color=cbandsclr, linestyle='--')
                cumprod = np.percentile(self._cumprod_mc, 90, axis=0)
                ax.plot(cumprod, rate_forecast, color=cbandsclr, linestyle='--')

            if self._model != 'exp':
                ax.set_yscale('log')
                ax.grid(which='minor', axis='y')
            if self._model == 'hyp':
                ax.set_xscale('log')
                ax.grid(which='major', axis='x')
                ax.grid(which='minor', axis='x')
            ax.set_xlabel('Cumulative Production')
        else:
            if bootstrap:
                ax.plot((self._time_mc + self._time_offset).transpose(), 
                        np.tile(rate_forecast, (self._time_mc.shape[0], 1)).transpose(), 
                        color=sampleclr, alpha=0.03) # plot bootstrap result
            ax.plot(time + self._time_offset, rate, color=cbandsclr, linestyle='--') # P50 forecast
            rate = self._func_rate[self._model](self._cumprod, self._rate_init, self._d_rate)
            time = self._func_time[self._model](rate, self._rate_init, self._d_rate)
            ax.scatter(time + self._time_offset, self._rate, color=histclr) #data points
            ax.plot(time + self._time_offset, rate, color=regclr) # regression line
            
            if bootstrap:
                time = np.percentile(self._time_mc, 10, axis=0)
                ax.plot(time + self._time_offset, rate_forecast, color=cbandsclr, linestyle='--')
                time = np.percentile(self._time_mc, 90, axis=0)
                ax.plot(time + self._time_offset, rate_forecast, color=cbandsclr, linestyle='--')

            ax.set_yscale('log')
            ax.grid(which='minor', axis='y')
            if self._model != 'exp':
                ax.set_xscale('log')
                ax.grid(which='major', axis='x')
                ax.grid(which='minor', axis='x')
            ax.set_xlabel('Time')

        ax.set_title((f"Arps Decline Model\n"
                      f"Selected Method: {self._MODEL_NAME[self._model]}"))
        ax.axhline(self._rate_eur, color='k', linestyle='dotted')
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.set_ylim(bottom=1)
        ax.set_ylabel('Rate')
        plt.show()

    def residual_analysis_plot(self, figsize=(12, 6)):
        """ Generate Residual and Probability Plot

        Parameters
        ---
        figsize : tuple
            Figure size
        """
        plt.style.use('ggplot')
        rate = self._func_rate[self._model](self._cumprod, self._rate_init, self._d_rate)
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        
        ax[0].scatter(rate, rate - self._rate)
        ax[0].axhline(0, linestyle='--')
        ax[0].set_xlabel('Rate')
        ax[0].set_ylabel('Residual')
        ax[0].set_title(f"Residual Plot")
        _, result = probplot(rate - self._rate, plot=ax[1])
        _, _, r = result
        fig.suptitle(('Residual Analysis\n'
                      f'Selected Method: {self._MODEL_NAME[self._model]}\n'
                      r'$R^2 =$' + f'{round(r**2, 4)}'))
        
    def summary(self):
        """
        Provide summary of fitting calculations.
        """
        print(f'Message: {self._message}')

        df = pd.DataFrame([self._rate_init, self._d_rate, self._b_exp,
                           self._cost_val, self._status])
        df = df.rename(index={0:'initial rate', 1:'decline rate', 2:'b exponent',
                              3:'cost value', 4:'status'})
        df = df.rename(columns={0:self._MODEL_NAME[self._model]})
        return df
        