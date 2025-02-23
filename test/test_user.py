# import numpy as np
# from lmfit.models import LorentzianModel

# from peskit.fit.fermi_dirac.model import FermiDiracModel


# def test_convolve():
#     x = np.linspace(-10, 10, 300)
#     lor_model = LorentzianModel()
#     center = 2.0
#     gau_model = GaussianConvolveModel(prefix="g_")
#     lor = lor_model.eval(x=x, center=center)


# model = LorentzianModel(prefix="p0_") * FermiDiracModel()
# model = CompositeModel(model, GaussianConvolveModel(sigma=1.0), convolve)
# guess = data.fit.guess(model=model, input_core_dims="eV")

# g_center = guess.params.get(params_name="g_center")
# print(g_center)
# guess.params.assign()
# guess["g_center"].set(value=0.0, vary=False)
# result = data.fit(model, guess, input_core_dims="eV", method="least_squares")
