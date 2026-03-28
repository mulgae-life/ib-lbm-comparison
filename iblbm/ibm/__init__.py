"""IBM 방법론 서브패키지. DF / MDF / DFC.

하위 호환: from iblbm.ibm import ibm_direct_forcing 그대로 동작.
"""

from .common import (
    delta_hat,
    delta_peskin4pt,
    delta_function,
    get_delta,
    _DELTA_REGISTRY,
)
from .df import ibm_direct_forcing
from .mdf import ibm_multi_direct_forcing
from .dfc import apply_dfc_correction

__all__ = [
    "delta_hat",
    "delta_peskin4pt",
    "delta_function",
    "get_delta",
    "_DELTA_REGISTRY",
    "ibm_direct_forcing",
    "ibm_multi_direct_forcing",
    "apply_dfc_correction",
]
