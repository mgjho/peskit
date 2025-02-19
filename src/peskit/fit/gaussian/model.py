from lmfit.lineshapes import gaussian
import lmfit as lf
class GaussianConvolveModel(lf.Model):
    """ Model for Gaussian convolution. """
    def __init__(
        self,
        independent_vars=["x"],
        prefix="g_",
        missing="drop",
        name=None,
        sigma: float | None = 0.01,
        **kwargs,
    ):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {"prefix": prefix, "missing": missing, "independent_vars": independent_vars}
        )
        self.sigma = sigma
        super().__init__(gaussian, **kwargs)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars[f"{self.prefix}amplitude"].set(value=1.0, min=0, max=100.0)
        pars[f"{self.prefix}center"].set(value=0, vary=False)
        if self.sigma is not None:
            pars[f"{self.prefix}sigma"].set(value=self.sigma, vary=False)
        else:
            pars[f"{self.prefix}sigma"].set(value=1.0, min=1e-4, max=100.0)

        return lf.models.update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC