import lmfit as lf

from peskit.common.distribution import fermi_dirac
from peskit.fit.fermi_dirac.function import fermi_dirac_linbkg_broad


class FermiDiracModel(lf.Model):
    """A model for the Fermi Dirac function."""

    def __init__(
        self,
        independent_vars=["x"],
        prefix="",
        missing="drop",
        name=None,
        temp: float | None = None,
        center: float | None = None,
        **kwargs,
    ):
        self.temp = temp
        self.center = center
        """Defer to lmfit for initialization."""
        kwargs.update(
            {"prefix": prefix, "missing": missing, "independent_vars": independent_vars}
        )
        super().__init__(fermi_dirac, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        if self.temp is None:
            self.set_param_hint("temp", value=10.0, min=0.0, max=300.0)
        else:
            self.set_param_hint("temp", value=self.temp, vary=False)
        if self.center is None:
            self.set_param_hint("center", value=0.0, min=-0.05, max=0.05)
        else:
            self.set_param_hint("center", value=self.center, vary=False)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()
        if self.temp is None:
            pars[f"{self.prefix}temp"].set(value=10.0, min=0, max=300.0, vary=True)
        else:
            pars[f"{self.prefix}temp"].set(value=self.temp, vary=False)
        if self.center is None:
            pars[f"{self.prefix}center"].set(value=0, min=-1, max=1)
        else:
            pars[f"{self.prefix}center"].set(value=self.center, vary=False)
        # if self.temp is None:
        #     pars[f"{self.prefix}temp"].set(value=10.0, min=0, max=300.0, vary=True)
        # else:
        #     pars[f"{self.prefix}temp"].set(value=self.temp, vary=False)
        # pars[f"{self.prefix}center"].set(value=0, min=-1, max=1)

        return lf.models.update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class FermiDiracLinbkgBroadModel(lf.Model):
    """A model for the Fermi Dirac function."""

    def __init__(
        self,
        independent_vars=["x"],
        prefix="",
        missing="drop",
        name=None,
        **kwargs,
    ):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {"prefix": prefix, "missing": missing, "independent_vars": independent_vars}
        )
        super().__init__(fermi_dirac_linbkg_broad, **kwargs)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()

        pars[f"{self.prefix}back0"].set(value=0.0, min=0)
        pars[f"{self.prefix}back1"].set(value=0.0, min=0)
        pars[f"{self.prefix}temp"].set(value=10.0, min=0, max=300.0)
        pars[f"{self.prefix}dos0"].set(value=data.max() / 2, min=0, max=data.max())
        pars[f"{self.prefix}dos1"].set(
            value=(data.max() - data.min()) / (x.max() - x.min()), min=0
        )
        pars[f"{self.prefix}center"].set(value=0, min=-1, max=1)
        pars[f"{self.prefix}resolution"].set(value=0.01)

        return lf.models.update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC
