import numpy as np
import numpy.typing as npt
from peskit.common.constant import TINY
def lorentzian(
    x: npt.NDArray[np.complex128],
    center_real: float,
    center_imag: float,
    amplitude_real: float,
) -> npt.NDArray[np.float64]:
    val = x.imag - center_imag
    val = np.where(np.abs(val) < TINY, TINY, val)
    return (-1 / np.pi) * ((amplitude_real) / (x - center_real + 1j * val)).imag