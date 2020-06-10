import numpy as np

def tau(pv: float, compr_total: float, pi: float):
    """ Time Constant

    Parameters
    ---
    pv : float
        pore volume
    compr_total : float
        total compressibility
    pi : float
        productivity index
    """
    return pv*compr_total/pi

def pvct(pv: float, compr_total: float):
    """ Pore Volume times Total Compressibility

    Parameters
    ---
    pv : float
        pore volume
    compr_total : float
        total compressibility
    Return
    pvct : float
        pore volume total compressibility
    """
    return pv*compr_total

def icrm(rate: list, bhp: list, pres_init: float, tau: float, pvct: float):
    """ Integrated Capacitance-Resistive Model (Nguyen, 2012)

    Parameters
    ---
    rate : list
        production rate [V/T]
    bhp : list
        bottom-hole pressure [P]
    pres_init : float
        initial reservoir pressure [P]
    tau : float
        time constant [T]
    pvct : float
        pore volume total compressibility [V/P]
    Return
    ---
    cumprod : ndarray
        cumulative production [V]
    """
    rate = np.asarray(rate)
    bhp = np.asarray(bhp)
    cumprod = pvct*pres_init - tau*rate - pvct*bhp
    return cumprod