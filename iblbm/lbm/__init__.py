"""LBM 핵심 모듈 서브패키지. D2Q9-BGK 충돌·스트리밍·평형분포.

하위 호환: from iblbm.lbm import collision_step 등으로 접근.
"""

from .lattice import D2Q9, make_d2q9
from .collision import collision_step, guo_forcing
from .streaming import streaming_step
from .equilibrium import compute_feq

__all__ = [
    "D2Q9", "make_d2q9",
    "collision_step", "guo_forcing",
    "streaming_step",
    "compute_feq",
]
