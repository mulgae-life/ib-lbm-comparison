"""IBM 공통 인프라 — delta 함수, 레지스트리.

delta 함수:
  Peskin 4-point regularized delta (Peskin, Acta Numerica 2002):
    phi(r) = (1/8)(3 - 2|r| + sqrt(1 + 4|r| - 4r^2))    0 ≤ |r| ≤ 1
    phi(r) = (1/8)(5 - 2|r| - sqrt(-7 + 12|r| - 4r^2))   1 ≤ |r| ≤ 2
    phi(r) = 0                                              |r| > 2
  C1 연속, partition of unity + 1차 모멘트 조건 충족.
  support width: 4dx (격자 단위 4). 스텐실 a=2 (5점).
"""

from __future__ import annotations

from ..backend import xp as np


def delta_hat(r: np.ndarray) -> np.ndarray:
    """1차 hat function (C0, support=2dx).

    phi(r) = max(1 - |r|, 0)
    조홍주(2022) 논문 원본에서 사용.

    Args:
        r: 격자 간격으로 정규화된 거리, 배열

    Returns:
        delta 함수 값, 배열
    """
    return np.maximum(1.0 - np.abs(r), 0.0)


def delta_peskin4pt(r: np.ndarray) -> np.ndarray:
    """Peskin 4-point regularized delta function.

    Peskin, Acta Numerica (2002):
      phi(r) = (1/8)(3 - 2|r| + sqrt(1 + 4|r| - 4r^2))    0 ≤ |r| ≤ 1
      phi(r) = (1/8)(5 - 2|r| - sqrt(-7 + 12|r| - 4r^2))   1 ≤ |r| ≤ 2
      phi(r) = 0                                              |r| > 2

    C1 연속, partition of unity + 1차 모멘트 조건 충족.
    hat function 대비 보간 정확도 O(h) → O(h^2) 개선.

    Args:
        r: 격자 간격으로 정규화된 거리, 배열

    Returns:
        delta 함수 값, 배열
    """
    ar = np.abs(r)
    result = np.zeros_like(ar)

    mask1 = ar <= 1.0
    mask2 = (~mask1) & (ar <= 2.0)

    r1 = ar[mask1]
    result[mask1] = (1.0 / 8.0) * (
        3.0 - 2.0 * r1 + np.sqrt(np.maximum(1.0 + 4.0 * r1 - 4.0 * r1**2, 0.0))
    )

    r2 = ar[mask2]
    result[mask2] = (1.0 / 8.0) * (
        5.0 - 2.0 * r2 - np.sqrt(np.maximum(-7.0 + 12.0 * r2 - 4.0 * r2**2, 0.0))
    )

    return result


# 하위 호환 alias
delta_function = delta_peskin4pt

# 디스패치 레지스트리: (함수, stencil 반경)
_DELTA_REGISTRY = {
    "peskin4pt": (delta_peskin4pt, 2),  # 5점 스텐실
    "hat":       (delta_hat, 1),         # 3점 스텐실
}


def get_delta(delta_type: str):
    """delta_type에 해당하는 (delta_func, stencil_radius) 튜플 반환."""
    return _DELTA_REGISTRY[delta_type]
