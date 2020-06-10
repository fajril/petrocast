import numpy as np

def exp_rate_cumprod(cumprod: list, rate_init: float, d_rate: float):
    """Exponential Model. 
    This function calculates rate from cumulative production.

    Parameters
    ---
    cumprod : list, ndarray
        cumulative production
    rate_init : float
        initial rate
    d_rate : float
        decline rate
    Return
    ---
    rate : ndarray
        production rate
    """
    cumprod = np.asarray(cumprod)
    return rate_init - d_rate * cumprod

def exp_rate_time(time: list, rate_init: float, d_rate: float):
    """Exponential Model.
    This function calculates rate from time.

    Parameters
    ---
    time : list, ndarray
        time series
    rate_init : float
        initial rate
    d_rate : float
        decline rate
    Return
    ---
    rate : ndarray
        production rate
    """
    time = np.asarray(time)
    return rate_init * np.exp(-d_rate * time)

def exp_cumprod_rate(rate: list, rate_init: float, d_rate: float):
    """Exponential Model.
    This function calculates cumulative production from rate.

    Parameters
    ---
    rate : list, ndarray
        production rate
    rate_init : float
        initial rate
    d_rate : float
        decline rate
    Return
    ---
    cumprod : ndarray
        cumulative production
    """
    rate = np.asarray(rate)
    return (rate_init - rate)/d_rate

def exp_cumprod_time(time: list, rate_init: float, d_rate: float):
    """Exponential Model.
    This function calculates cumulative production from time.

    Parameters
    ---
    time : list, ndarray
        time series
    rate_init : float
        initial rate
    d_rate : float
        decline rate
    Return
    ---
    cumprod : ndarray
        cumulative production
    """
    raise NotImplementedError()

def exp_time_cumprod(cumprod: list, rate_init: float, d_rate: float):
    """Exponential Model.
    This function calculates time from cumulative production.

    Parameters
    ---
    cumprod : list, ndarray
        cumulative production
    rate_init : float
        initial rate
    d_rate : float
        decline rate
    Return
    ---
    time : ndarray
        time series
    """
    raise NotImplementedError()

def exp_time_rate(rate: list, rate_init: float, d_rate: float):
    """Exponential Model.
    This function calculates time from rate.

    Parameters
    ---
    rate : list, ndarray
        production rate
    rate_init : float
        initial rate
    d_rate : float
        decline rate
    Return
    ---
    time : ndarray
        time series
    """
    rate = np.asarray(rate)
    return np.log(rate_init/rate) / d_rate

def har_rate_cumprod(cumprod: list, rate_init: float, d_rate: float):
    """Harmonic Model. 
    This function calculates rate from cumulative production.

    Parameters
    ---
    cumprod : list, ndarray
        cumulative production
    rate_init : float
        initial rate
    d_rate : float
        decline rate
    Return
    ---
    rate : ndarray
        production rate
    """
    cumprod = np.asarray(cumprod)
    return np.exp(np.log(rate_init) - d_rate * cumprod / rate_init)

def har_rate_time(time: list, rate_init: float, d_rate: float):
    """Harmonic Model.
    This function calculates rate from time.

    Parameters
    ---
    time : list, ndarray
        time series
    rate_init : float
        initial rate
    d_rate : float
        decline rate
    Return
    ---
    rate : ndarray
        production rate
    """
    time = np.asarray(time)
    return rate_init / (1 + d_rate * time)

def har_cumprod_rate(rate: list, rate_init: float, d_rate: float):
    """Harmonic Model.
    This function calculates cumulative production from rate.

    Parameters
    ---
    rate : list, ndarray
        production rate
    rate_init : float
        initial rate
    d_rate : float
        decline rate
    Return
    ---
    cumprod : ndarray
        cumulative production
    """
    rate = np.asarray(rate)
    return np.log(rate_init/rate)*rate_init/d_rate

def har_cumprod_time(time: list, rate_init: float, d_rate: float):
    """Harmonic Model.
    This function calculates cumulative production from time.

    Parameters
    ---
    time : list, ndarray
        time series
    rate_init : float
        initial rate
    d_rate : float
        decline rate
    Return
    ---
    cumprod : ndarray
        cumulative production
    """
    raise NotImplementedError()

def har_time_rate(rate: list, rate_init: float, d_rate: float):
    """Harmonic Model.
    This function calculates time from rate.

    Parameters
    ---
    rate : list, ndarray
        production rate
    rate_init : float
        initial rate
    d_rate : float
        decline rate
    Return
    ---
    time : ndarray
        time series
    """
    rate = np.asarray(rate)
    return (1./rate - 1./rate_init) * rate_init/d_rate

def har_time_cumprod(cumprod: list, rate_init: float, d_rate: float):
    """Harmonic Model.
    This function calculates time from cumulative production.

    Parameters
    ---
    cumprod : list, ndarray
        cumulative production
    rate_init : float
        initial rate
    d_rate : float
        decline rate
    Return
    ---
    time : ndarray
        time series
    """
    raise NotImplementedError()

def hyp_rate_cumprod(cumprod: list, rate_init: float, d_rate: float, b_exp: float):
    """Hyperbolic Model. 
    This function calculates rate from cumulative production.

    Parameters
    ---
    cumprod : list, ndarray
        cumulative production
    rate_init : float
        initial rate
    d_rate : float
        decline rate
    Return
    ---
    rate : ndarray
        production rate
    """
    cumprod = np.asarray(cumprod)
    const = rate_init/(d_rate * (1 - b_exp))
    return np.exp(np.log(1 - cumprod/const)/(1 - b_exp) + np.log(rate_init))

def hyp_rate_time(time: list, rate_init: float, d_rate: float, b_exp: float):
    """Hyperbolic Model.
    This function calculates rate from time.

    Parameters
    ---
    time : list, ndarray
        time series
    rate_init : float
        initial rate
    d_rate : float
        decline rate
    Return
    ---
    rate : ndarray
        production rate
    """
    time = np.asarray(time)
    return rate_init / np.power(1 + b_exp*d_rate*time, 1/b_exp)

def hyp_cumprod_rate(rate: list, rate_init: float, d_rate: float, b_exp: float):
    """Hyperbolic Model.
    This function calculates cumulative production from rate.

    Parameters
    ---
    rate : list, ndarray
        production rate
    rate_init : float
        initial rate
    d_rate : float
        decline rate
    Return
    ---
    cumprod : ndarray
        cumulative production
    """
    rate = np.asarray(rate)
    const = rate_init/(d_rate * (1 - b_exp))
    return const * (1 - np.power(rate/rate_init, (1 - b_exp)))

def hyp_cumprod_time(time: list, rate_init: float, d_rate: float, b_exp: float):
    """Hyperbolic Model.
    This function calculates cumulative production from time.

    Parameters
    ---
    time : list, ndarray
        time series
    rate_init : float
        initial rate
    d_rate : float
        decline rate
    Return
    ---
    cumprod : ndarray
        cumulative production
    """
    raise NotImplementedError()

def hyp_time_rate(rate: list, rate_init: float, d_rate: float, b_exp: float):
    """Hyperbolic Model.
    This function calculates time from rate.

    Parameters
    ---
    rate : list, ndarray
        production rate
    rate_init : float
        initial rate
    d_rate : float
        decline rate
    Return
    ---
    time : ndarray
        time series
    """
    rate = np.asarray(rate)
    return (np.exp(b_exp*(np.log(rate_init) - np.log(rate))) - 1)/(b_exp * d_rate)

def hyp_time_cumprod(cumprod: list, rate_init: float, d_rate: float, b_exp: float):
    """Harmonic Model.
    This function calculates time from cumulative production.

    Parameters
    ---
    cumprod : list, ndarray
        cumulative production
    rate_init : float
        initial rate
    d_rate : float
        decline rate
    Return
    ---
    time : ndarray
        time series
    """
    raise NotImplementedError()
