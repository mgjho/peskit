# import Callable
from collections.abc import Callable

import numba as nb
import numpy as np
import numpy.typing as npt
from dsfit.commons.constants import S2PI, TINY


def do_convolve(
    x: npt.NDArray[np.float64],
    func: Callable,
    resolution: float,
    pad: int = 3,  # 5
    **kwargs,
) -> npt.NDArray[np.float64]:
    r"""Convolves `func` with gaussian of FWHM `resolution` in `x`.

    Parameters
    ----------
    x
        A evenly spaced array specifing where to evaluate the convolution.
    func
        Function to convolve.
    resolution
        FWHM of the gaussian kernel.
    pad
        Multiples of the standard deviation :math:`\sigma` to pad with.
    **kwargs
        Additional keyword arguments to `func`.

    """
    xn, g = _gen_kernel(
        np.asarray(x, dtype=np.float64), float(resolution), pad=int(pad)
    )
    return np.convolve(func(xn, **kwargs), g, mode="valid")


@nb.njit(cache=True)
def _gen_kernel(
    x: npt.NDArray[np.float64], resolution: float, pad: int = 3
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Generate a Gaussian kernel for convolution.

    Parameters
    ----------
    x
        The input array of x values.
    resolution
        The resolution of the kernel given as FWHM.
    pad
        Multiples of the standard deviation :math:`\sigma` to truncate the kernel at.

    Returns
    -------
    extended
        The domain of the kernel.
    gauss
        The gaussian kernel defined on `extended`.

    """
    delta_x = x[1] - x[0]

    sigma = abs(resolution) / np.sqrt(8 * np.log(2))  # resolution given in FWHM
    n_pad = int(sigma * pad / delta_x + 0.5)
    x_pad = n_pad * delta_x

    extended = np.linspace(x[0] - x_pad, x[-1] + x_pad, 2 * n_pad + len(x))
    gauss = (
        delta_x
        * np.exp(
            -(np.linspace(-x_pad, x_pad, 2 * n_pad + 1) ** 2) / max(TINY, 2 * sigma**2)
        )
        / max(TINY, S2PI * sigma)
    )
    return extended, gauss
