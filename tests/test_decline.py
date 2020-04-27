import pytest

from petrocast import ArpsDecline


def test_exp_rate_cumprod():
    d_rate = 0.000418
    rate_init = 344
    rate = [197]
    cumprod = [352000]

    model = ArpsDecline(cumprod, rate)
    rate_calc = model._exp_rate(cumprod, rate_init, d_rate)
    cumprod_calc = model._exp_cum(rate, rate_init, d_rate)

    assert rate == pytest.approx(rate_calc, 0.1)
    assert cumprod == pytest.approx(cumprod_calc, 1000)

def test_hyp_rate_cumprod():
    d_rate_hyp = 0.001
    b_exp = 0.5195

    rate_init = 10
    rate = [7.16088839552745]
    cumprod = [3085.35]

    model = ArpsDecline(cumprod, rate)
    rate_calc = model._hyp_rate(cumprod, rate_init, d_rate_hyp, b_exp)
    cumprod_calc = model._hyp_cum(rate, rate_init, d_rate_hyp, b_exp)

    assert rate == pytest.approx(rate_calc)
    assert cumprod == pytest.approx(cumprod_calc)

def test_fit():
    rate = [320, 336, 304, 309, 272, 248, 208, 197, 184, 176, 184]
    cumprod = [16000, 32000, 48000, 96000, 160000,
               240000, 304000, 352000, 368000, 384000, 400000]

    d_rate_exp = 0.0003493302962652074
    d_rate_har = 0.00043039326675591407
    d_rate_hyp = 0.00041888177025878635
    model = ArpsDecline(cumprod, rate)
    model.fit()

    assert d_rate_exp == pytest.approx(model._d_rate['exp'], 1)
    assert d_rate_har == pytest.approx(model._d_rate['har'], 1)
    assert d_rate_hyp == pytest.approx(model._d_rate['hyp'], 1)
    
