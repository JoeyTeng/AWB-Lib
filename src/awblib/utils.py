from __future__ import annotations  # for compatibility with Python 3.8

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """Load an image from a path."""
    assert isinstance(path, str)

    return np.asarray(Image.open(path))[:, :, :3]  # discard alpha channel


def visualise_histogram(
    image: np.ndarray,
    bit_depth: int = 8,
    ax: Optional[plt.Axes] = None,
    color: list[str] = ["#FF0000", "#00FF00", "#0000FF"],
    cdf: bool = False,
    normalise: bool = True,
) -> float:
    """Visualise a histogram; return Overlap Area (OA)."""
    assert image.ndim == 3, f"image must has 3 channels, got {image.ndim}"

    channels: int = image.shape[2]
    assert channels == 3, f"image must be RGB, got {channels} channels"

    total_pixels: int = image.shape[0] * image.shape[1]

    assert bit_depth > 0, f"bit_depth must be positive, got {bit_depth}"
    max_value: int = 2**bit_depth

    histogram: np.ndarray = np.vstack([
        np.histogram(
            image[:, :, ch],
            # - 0.5 to make sure that the bins are centered around the values
            bins=np.arange(max_value + 1) - 0.5,
        )[0] for ch in range(channels)
    ])  # shape (channels, max_value)

    overlap_area: float = (histogram.min(axis=0) / total_pixels).sum()

    if cdf:
        histogram = histogram.cumsum(axis=1)

    # normalise
    if normalise:
        histogram = histogram / total_pixels

    assert isinstance(histogram, np.ndarray)

    if ax is None:
        fig, _ax = plt.subplots()
    else:
        _ax = ax

    assert isinstance(_ax, plt.Axes)

    for ch in range(channels):
        _ax.plot(
            np.arange(max_value),
            histogram[ch],
            color=color[ch],
            label=f"channel {ch}",
        )

    if ax is None:
        fig.show()

    return overlap_area
