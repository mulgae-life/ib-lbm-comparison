"""D2Q9 격자 상수 정의.

MATLAB 참조: IBLBM_steady_DF_CPU.m lines 72-89

D2Q9 방향 인덱스:
  0: 정지 (0,0)    5: 우상 (+1,+1)
  1: 우  (+1,0)    6: 좌상 (-1,+1)
  2: 상  (0,+1)    7: 좌하 (-1,-1)
  3: 좌  (-1,0)    8: 우하 (+1,-1)
  4: 하  (0,-1)

격자 속도 c = dx/dt = 1 (lattice 단위). cs^2 = c^2/3 = 1/3.
"""

from ..backend import xp as np
from dataclasses import dataclass


@dataclass(frozen=True)
class D2Q9:
    """D2Q9 격자 볼츠만 상수.

    c = 1 가정 (lattice 단위: dx = dt = 1).
    feq 수식에서 c^2, c^4 항은 1로 처리됨 (M2, M3 참조).
    """

    e: np.ndarray    # 격자 속도 벡터, (9, 2)
    w: np.ndarray    # 가중치, (9,)
    opp: np.ndarray  # 반대 방향 인덱스, (9,) 0-base
    cs2: float = 1 / 3  # 격자 음속 제곱 (= c^2 / 3)

    def __hash__(self):
        return id(self)


def make_d2q9() -> D2Q9:
    """D2Q9 격자 상수 생성.

    MATLAB: e = [0 0; c 0; 0 c; -c 0; 0 -c; c c; -c c; -c -c; c -c]
            w = [4/9 1/9 1/9 1/9 1/9 1/36 1/36 1/36 1/36]
            opp = [1 4 5 2 3 8 9 6 7]  (1-base)
    """
    e = np.array([
        [0, 0],    # 0: 정지
        [1, 0],    # 1: 우
        [0, 1],    # 2: 상
        [-1, 0],   # 3: 좌
        [0, -1],   # 4: 하
        [1, 1],    # 5: 우상
        [-1, 1],   # 6: 좌상
        [-1, -1],  # 7: 좌하
        [1, -1],   # 8: 우하
    ], dtype=np.float64)

    w = np.array([
        4 / 9,                           # 정지
        1 / 9, 1 / 9, 1 / 9, 1 / 9,     # 축 방향
        1 / 36, 1 / 36, 1 / 36, 1 / 36,  # 대각 방향
    ], dtype=np.float64)

    # 반대 방향: MATLAB [1 4 5 2 3 8 9 6 7] (1-base) → 0-base
    opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

    return D2Q9(e=e, w=w, opp=opp)
