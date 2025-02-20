from lmfit.models import LorentzianModel

from peskit.fit.fermi_dirac.model import FermiDiracModel
from peskit.sim import get_cut


# from lmfit.lineshapes import gaussian
def test_user():
    data = get_cut()
    data.plot()
    model = LorentzianModel(prefix="p0_") * FermiDiracModel()
    # model = CompositeModel(model, GaussianConvolveModel(sigma=1.0), convolve)
    guess = data.fit.guess(model=model, input_core_dims="eV")

    # g_center = guess.params.get(params_name="g_center")
    # print(g_center)
    # guess.params.assign()
    # guess["g_center"].set(value=0.0, vary=False)
    result = data.fit(model, guess, input_core_dims="eV", method="least_squares")
