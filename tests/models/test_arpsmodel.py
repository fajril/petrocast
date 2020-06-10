import pytest

from petrocast.models import arpsmodel

def test_exp_rate_cumprod():
    d_rate = 0.000418
    rate_init = 344
    rate = [197]
    cumprod = [352000]

    rate_calc = arpsmodel.exp_rate_cumprod(cumprod, rate_init, d_rate)
    cumprod_calc = arpsmodel.exp_cumprod_rate(rate, rate_init, d_rate)

    assert rate == pytest.approx(rate_calc, 0.1)
    assert cumprod == pytest.approx(cumprod_calc, 1000)

def test_hyp_rate_cumprod():
    d_rate_hyp = 0.001
    b_exp = 0.5195

    rate_init = 10
    rate = [7.16088839552745]
    cumprod = [3085.35]

    rate_calc = arpsmodel.hyp_rate_cumprod(cumprod, rate_init, d_rate_hyp, b_exp)
    cumprod_calc = arpsmodel.hyp_cumprod_rate(rate, rate_init, d_rate_hyp, b_exp)

    assert rate == pytest.approx(rate_calc)
    assert cumprod == pytest.approx(cumprod_calc)

   
