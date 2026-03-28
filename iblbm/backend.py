"""GPU/CPU 백엔드 선택.

기본값 → CuPy (GPU). GPU가 항상 기본.
IBLBM_GPU=0 명시 시에만 NumPy (CPU) 사용.
"""

import os

_use_gpu = os.environ.get("IBLBM_GPU", "1") != "0"

if _use_gpu:
    try:
        import cupy as xp
    except ImportError:
        import numpy as xp
        _use_gpu = False
else:
    import numpy as xp


def add_at(target, indices, values):
    """중복 인덱스 누적 (np.add.at 호환 래퍼).

    NumPy/CuPy 공통: xp.add.at(target, indices, values)
    CuPy 14+에서 cupy.add.at 공식 지원 (cupyx.scatter_add는 deprecated).

    ibm.py에서 Lagrangian → Eulerian 힘 분산에 사용.
    중복 인덱스가 존재할 때 일반 += 는 마지막 값만 반영하지만,
    이 함수는 모든 값을 누적한다.
    """
    xp.add.at(target, indices, values)
