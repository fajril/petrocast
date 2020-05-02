import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats.distributions import t
from scipy.stats.distributions import norm

from functools import partial

import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib.ticker import FuncFormatter

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
    def __init__(self, cumprod: list, rate: list, time: list=None):
        self._best_curve = ''
        self._MODEL_NAME = {'exp':'exponential', 'har':'harmonic', 'hyp':'hyperbolic'}
        
        self._rate = np.asarray(rate)
        self._cumprod = np.asarray(cumprod)
        self._time = np.asarray(time)

        if len(cumprod) != len(rate):
            raise IndexError("cumulative production and rate must have the same length")
    
    def _exp_rate(self, cumprod: list, rate_init: float, d_rate: float):
        cumprod = np.asarray(cumprod)
        return rate_init - d_rate * cumprod

    def _exp_time(self, rate: list, rate_init: float, d_rate: float):
        rate = np.asarray(rate)
        return np.log(rate_init/rate) / d_rate

    def _exp_cum(self, rate: list, rate_init: float, d_rate: float):
        rate = np.asarray(rate)
        return (rate_init - rate)/d_rate

    def _har_rate(self, cumprod: list, rate_init: float, d_rate: float):
        cumprod = np.asarray(cumprod)
        return np.exp(np.log(rate_init) - d_rate * cumprod / rate_init)

    def _har_time(self, rate: list, rate_init: float, d_rate: float):
        rate = np.asarray(rate)
        return (1./rate - 1./rate_init) * rate_init/d_rate

    def _har_cum(self, rate: list, rate_init: float, d_rate: float):
        rate = np.asarray(rate)
        return np.log(rate_init/rate)*rate_init/d_rate

    def _hyp_rate(self, cumprod: list, rate_init: float, d_rate: float, b_exp: float):
        cumprod = np.asarray(cumprod)
        const = rate_init/(d_rate * (1 - b_exp))
        return np.exp(np.log(1 - cumprod/const)/(1 - b_exp) + np.log(rate_init))

    def _hyp_time(self, rate: list, rate_init: float, d_rate: float, b_exp: float):
        rate = np.asarray(rate)
        return (np.exp(b_exp*(np.log(rate_init) - np.log(rate))) - 1)/(b_exp * d_rate)

    def _hyp_cum(self, rate: list, rate_init: float, d_rate: float, b_exp: float):
        rate = np.asarray(rate)
        const = rate_init/(d_rate * (1 - b_exp))
        return const * (1 - np.power(rate/rate_init, (1 - b_exp)))

    def _fit_exp(self, cumprod, rate, d_rate_init: float=1E-5):
        init = np.asarray([rate[0], d_rate_init])
        result = optimize.least_squares(lambda x: \
            rate - self._exp_rate(cumprod, x[0], x[1]), init, verbose=self._verbose)
        return result

    def _fit_har(self, cumprod, rate, d_rate_init: float=1E-5):
        init = np.asarray([rate[0], d_rate_init])
        result = optimize.least_squares(lambda x: 
            rate - self._har_rate(cumprod, x[0], x[1]), init, verbose=self._verbose)
        return result
    
    def _fit_hyp(self, cumprod, rate, d_rate_init: float=1E-5, b_exp_init: float=0.5):
        init = np.asarray([rate[0], d_rate_init, b_exp_init])
        rate
        result = optimize.least_squares(lambda x: \
            rate - self._hyp_rate(cumprod, x[0], x[1], x[2]), \
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
        bootstrap_result = [self._fit_data(prod_sample[s, :, 0], prod_sample[s, :, 1], model)
                            for s in range(sample)]
        rate_init = np.asarray([bootstrap_result[i].x[0] for i in range(sample)])
        d_rate = np.asarray([bootstrap_result[i].x[1] for i in range(sample)])
        if model == 'exp':
            b_exp = np.zeros(sample)
        elif model == 'har':
            b_exp = np.ones(sample)
        else:
            b_exp = np.asarray([bootstrap_result[i].x[2] for i in range(sample)])
        return rate_init, d_rate, b_exp
 
    def fit(self, model: str='best', d_rate_init: float=1E-5, b_exp_init: float=0.5,
            verbose: int=0, sig: float=0.2):
        """ 
        Fit the production model with the curves.
        This method does not provide time based fitting.

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

        self._func_time = {'exp':self._exp_time,
                           'har':self._har_time,
                           'hyp':partial(self._hyp_time, b_exp=self._b_exp)}

        self._func_cum = {'exp':self._exp_cum,
                          'har':self._har_cum,
                          'hyp':partial(self._hyp_cum, b_exp=self._b_exp)}
        
        self._func_rate = {'exp':self._exp_rate,
                           'har':self._har_rate,
                           'hyp':partial(self._hyp_rate, b_exp=self._b_exp)}

        logging.debug(f'Best fitting \n', \
                        'method: {self._best}, cost value: {self._cost_val[self._best]}')
        logging.debug(f'Selected model \n', \
                        'method: {self._model}, cost value: {self._cost_val[self._model]}')
        
    def eur(self, rate_eur: float=1):
        self._rate_eur = rate_eur
        
        eur = self._func_cum[self._model](rate_eur, \
                                            self._rate_init, \
                                            self._d_rate)
        return eur

    def forecast(self, rate_eur: float=1):
        """
        Forecast production profile.

        return
        ---
        time : ndarray
            time series
        cumprod : ndarray
            cumulative production
        rate : ndarray
            production rate
        """
        rate_frcast = np.linspace(self._rate[-1], rate_eur)
        cumprod_frcast_mid = self._func_cum[self._model](rate_frcast, \
                                                            self._rate_init, \
                                                            self._d_rate)
        time_frcast_mid = self._func_time[self._model](rate_frcast, \
                                                           self._rate_init, \
                                                           self._d_rate)

        return time_frcast_mid, cumprod_frcast_mid, rate_frcast

    def uncertainty_eur(self, rate_eur, sample: int=1000, seed=None):
        _, cumprod = self._uncertainty_forecast(rate_eur, sample, seed)
        P90 = np.percentile(cumprod, 10)
        P10 = np.percentile(cumprod, 90)
        return P90, self.eur(rate_eur), P10

    def uncertainty_eur_histplot(self, rate_eur: float=10, sample=1000, 
                                    seed=None, xcumprod: bool=True, 
                                    color_style: str='oil', figsize=(12, 8)):
        
        plt.style.use('ggplot')
        if color_style == 'gas':
            regclr = 'brown'
            sampleclr = 'coral'
        else:
            regclr = 'darkgreen'
            sampleclr = 'limegreen'
        _, ax = plt.subplots(figsize=figsize)
        _, cumprod = self._uncertainty_forecast(rate_eur, sample, seed)
        loc, scale = norm.fit(cumprod)
        ax.hist(cumprod, bins='auto', density=True, color=sampleclr, alpha=0.5)

        dist = norm(loc=loc, scale=scale)
        eur = np.linspace(dist.ppf(0.0005), dist.ppf(0.9995))
        normal = ax.plot(eur, dist.pdf(eur), color=regclr)
        P90 = np.percentile(cumprod, 10)
        P50 = self.eur(rate_eur)
        P10 = np.percentile(cumprod, 90)
        low = ax.axvline(P90, color=regclr, linestyle='--', alpha=0.5)
        mid = ax.axvline(P50, color=regclr, linestyle='--', alpha=0.5)
        hgh = ax.axvline(P10, color=regclr, linestyle='--', alpha=0.5)

        ax.set_xlim(eur[0], eur[-1])
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.set_title(f'EUR Distribution based on Arps Model\nSelected Method: {self._MODEL_NAME[self._model]}')
        ax.set_ylabel('Probability Density')
        ax.set_xlabel('Estimated Ultimate Recovery')
        ax.legend([normal, low, mid, hgh], ('PDF',
                                            f'P90: {round(P90, 2)}', 
                                            f'P50: {round(P50, 2)}', 
                                            f'P10: {round(P10, 2)}'))
        plt.show()

    def uncert_rate(self,xcumprod=True, percentile=50):
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

    def _uncertainty_forecast(self, rate_forecast, sample: int=1000, seed=None):
        """
        Calculate uncertainty for Estimated Ultimate Recovery (EUR)
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
        rate_init, d_rate, b_exp = self._bootstrap(self._model, sample=sample, seed=seed)
        if self._model == 'exp':
            cumprod_frcast = np.asarray([self._exp_cum(rate_forecast, rate_init[s], d_rate[s])
                                            for s in range(sample)])
            time_frcast = np.asarray([self._exp_time(rate_forecast, rate_init[s], d_rate[s])
                                        for s in range(sample)])
        elif self._model == 'har':
            cumprod_frcast = np.asarray([self._har_cum(rate_forecast, rate_init[s], d_rate[s])
                                            for s in range(sample)])
            time_frcast = np.asarray([self._har_time(rate_forecast, rate_init[s], d_rate[s])
                                        for s in range(sample)])
        else:
            cumprod_frcast = np.asarray([self._hyp_cum(rate_forecast, rate_init[s], d_rate[s], b_exp[s])
                                            for s in range(sample)])
            time_frcast = np.asarray([self._hyp_time(rate_forecast, rate_init[s], d_rate[s], b_exp[s])
                                        for s in range(sample)])
        return time_frcast, cumprod_frcast

    def uncertainty_forecast_plot(self, rate_eur: float=10, sample=1000, 
                                    seed=None, xcumprod: bool=True, 
                                    color_style: str='oil', figsize=(12, 8)):
        """
        Plot decline curve model.

        Parameters
        ---
        color_style : str
            'cartesian', 'semilog', 'loglog'
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
        rate_forecast = np.linspace(self._rate[-1], rate_eur)
        time_forecast_all, cumprod_forecast_all = self._uncertainty_forecast(rate_forecast, sample, seed)
        time, cumprod, rate = self.forecast(rate_eur)
        
        if xcumprod:
            for s in range(cumprod_forecast_all.shape[0]): # plot bootstrap result
                ax.plot(cumprod_forecast_all[s], rate_forecast, color=sampleclr, alpha=0.03)
            ax.plot(cumprod, rate, color=cbandsclr, linestyle='--') # P50 forecast
            ax.scatter(self._cumprod, self._rate, color=histclr) # data points
            
            rate = self._func_rate[self._model](self._cumprod, self._rate_init, self._d_rate)
            ax.plot(self._cumprod, rate, color=regclr) # regression line

            cumprod = np.percentile(cumprod_forecast_all, 10, axis=0)
            ax.plot(cumprod, rate_forecast, color=cbandsclr, linestyle='--')
            cumprod = np.percentile(cumprod_forecast_all, 90, axis=0)
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
            for s in range(time_forecast_all.shape[0]): # plot bootstrap result
                ax.plot(time_forecast_all[s], rate_forecast, color=sampleclr, alpha=0.03)
            ax.plot(time, rate, color=cbandsclr, linestyle='--') # P50 forecast
            rate = self._func_rate[self._model](self._cumprod, self._rate_init, self._d_rate)
            time = self._func_time[self._model](rate, self._rate_init, self._d_rate)
            ax.scatter(time, self._rate, color=histclr) #data points
            ax.plot(time, rate, color=regclr) # regression line
            
            time = np.percentile(time_forecast_all, 10, axis=0)
            ax.plot(time, rate_forecast, color=cbandsclr, linestyle='--')
            time = np.percentile(time_forecast_all, 90, axis=0)
            ax.plot(time, rate_forecast, color=cbandsclr, linestyle='--')

            ax.set_yscale('log')
            ax.grid(which='minor', axis='y')
            if self._model != 'exp':
                ax.set_xscale('log')
                ax.grid(which='major', axis='x')
                ax.grid(which='minor', axis='x')
            ax.set_xlabel('Time')

        ax.set_title(f"Arps Decline Model\nSelected Method: {self._MODEL_NAME[self._model]}")
        ax.axhline(rate_eur, color='k', linestyle='dotted')
        #ax.grid(which='major', axis='y')
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.set_ylim(bottom=1)
        ax.set_ylabel('Rate')
        
        plt.show()

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
                              3:'cost value', 4:'status'})
        df = df.rename(columns=self._MODEL_NAME)
        return df
        
