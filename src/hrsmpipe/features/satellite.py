"""Satellite band features: indices and tasselled-cap transforms (notebook cells 72, 83-84).

Each function takes ``(time, site)`` band arrays and returns the satellite-specific feature
columns (table columns 24 onwards) in table order. Expressions are copied from the notebook.
"""

from __future__ import annotations

import numpy as np


def s1_features(vv, vh, angle, invalid_above: float = 1000.0) -> list[np.ndarray]:
    """Sentinel-1: VV, VH, angle, VV/VH, (VV^2+VH^2)/sqrt(2), (VV+VH)/(VV-VH).

    Values above ``invalid_above`` (inf-like artefacts, cell 70) are set to NaN first.
    """
    vv, vh, angle = (np.where(a > invalid_above, np.nan, a) for a in (vv, vh, angle))
    return [
        vv,
        vh,
        angle,
        vv / vh,
        (vv**2 + vh**2) / np.sqrt(2),
        (vv + vh) / (vv - vh),
    ]


def s2_features(b: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Sentinel-2: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 (x1e-4), NDVI, NDWI, greenness, brightness, wetness."""
    B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 = (
        b[k] * 0.0001 for k in ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
    )
    ndvi = (B8 - B4) / (B8 + B4)
    ndwi = (B3 - B8) / (B3 + B8)
    greenness = -0.3599 * B2 - 0.3533 * B3 - 0.4734 * B4 + 0.6633 * B8 - 0.0087 * B11 - 0.2856 * B12
    brightness = 0.3510 * B2 + 0.3813 * B3 + 0.3437 * B4 + 0.7196 * B8 + 0.2396 * B11 + 0.1949 * B12
    wetness = 0.2578 * B2 + 0.2305 * B3 + 0.0883 * B4 + 0.1071 * B8 - 0.7611 * B11 - 0.5308 * B12
    return [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, ndvi, ndwi, greenness, brightness, wetness]


def hls_features(b: dict[str, np.ndarray], angle_constant: float = 90.0) -> list[np.ndarray]:
    """HLSL30: B2..B7, B10, B11 (raw, no scale factor), 4 constant angle columns, NDVI, NDWI, tasselled cap.

    B9 is fetched but unused. The four angle columns (SZA, SAA, VZA, VAA in the training data)
    were the constant 90 in the notebook.
    """
    B2, B3, B4, B5, B6, B7, B10, B11 = (
        b[k] for k in ("B2", "B3", "B4", "B5", "B6", "B7", "B10", "B11")
    )
    angles = [np.full(B2.shape, angle_constant) for _ in range(4)]
    ndvi = (B5 - B4) / (B5 + B4)
    ndwi = (B3 - B5) / (B3 + B5)
    greenness = -0.2941 * B2 - 0.243 * B3 - 0.5424 * B4 + 0.7276 * B5 + 0.0713 * B6 - 0.1608 * B7
    brightness = 0.3510 * B2 + 0.3813 * B3 + 0.3437 * B4 + 0.7196 * B5 + 0.2396 * B6 + 0.1949 * B7
    wetness = 0.2578 * B2 + 0.2305 * B3 + 0.0883 * B4 + 0.1071 * B5 - 0.7611 * B6 - 0.5308 * B7
    return [B2, B3, B4, B5, B6, B7, B10, B11, *angles, ndvi, ndwi, greenness, brightness, wetness]
