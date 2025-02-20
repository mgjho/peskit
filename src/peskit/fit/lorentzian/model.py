from peskit.fit.lorentzian.function import lorentzian
from lmfit import Model
from lmfit.models import COMMON_GUESS_DOC, COMMON_INIT_DOC, update_param_vals
class LorentzianModel(Model):
    """A model based on a Lorentzian or Cauchy-Lorentz distribution function."""

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs,):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(lorentzian, **kwargs)

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = self.make_params()
        pars[f"{self.prefix}center_real"].set(value=x.mean(), min=x.min()*2, max=x.max()*2)
        pars[f"{self.prefix}center_imag"].set(value=-0.1, max=0, min=-1.0)
        pars[f"{self.prefix}amplitude_real"].set(value=1.0, min=1e-4, max=100.0)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC
