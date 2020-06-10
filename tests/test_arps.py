import pytest

from petrocast import ArpsRegression

def test_eur():

    rate = [320, 336, 304, 309, 272, 248, 208, 197, 184, 176, 184]
    cumprod = [16000, 32000, 48000, 96000, 160000,
               240000, 304000, 352000, 368000, 384000, 400000]

    eur = 837176.9134253453
    model = ArpsRegression(cumprod, rate)
    model.fit()

    assert model.eur() == pytest.approx(eur)
