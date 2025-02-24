import numba as nb
import numpy as np
import numpy.typing as npt

from peskit.common.constant import CONST_KB, TINY


@nb.njit(nogil=True, cache=True)
def bose_einstein(
    x: npt.NDArray[np.float64 | np.complex128],
    temp: float,
    center: float = 0.0,
) -> npt.NDArray[np.float64 | np.complex128]:
    """Calculate the Bose-Einstein distribution for a given energy array."""
    x = (x - center) / (max(TINY, temp * CONST_KB))
    return 1.0 / (np.exp(x) - 1.0)


@nb.njit(nogil=True, cache=True)
def fermi_dirac(
    x: npt.NDArray[np.float64 | np.complex128],
    temp: float,
    center: float = 0.0,
) -> npt.NDArray[np.float64 | np.complex128]:
    """Calculate the Fermi-Dirac distribution for a given energy array."""
    x = (x - center) / (max(TINY, temp * CONST_KB))
    return 1.0 / (np.exp(x) + 1.0)
