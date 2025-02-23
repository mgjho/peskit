import numpy as np
from lmfit import CompositeModel
from lmfit.models import LorentzianModel

from peskit.common.function import convolve
from peskit.fit.broadening.model import GaussianBroadeningModel


def test_convolve():
    center = 5.0
    x = np.linspace(-10, 10, 200)
    lor_model = LorentzianModel()
    gau_model = GaussianBroadeningModel()
    lor_b_model = CompositeModel(lor_model, gau_model, convolve)
    lor_b = lor_b_model.eval(x=x, center=center)
    expected_values = np.array(
        [
            0.00078413,
            0.00085171,
            0.00091974,
            0.00098766,
            0.00105497,
            0.0011212,
            0.00118595,
            0.00124885,
            0.00130964,
            0.00136811,
            0.00142412,
            0.00147763,
            0.00152865,
            0.00157726,
        ]
    )
    assert np.allclose(
        lor_b[:14], expected_values
    ), f"Expected {expected_values}, but got {lor_b[:14]}"
