# Implementing the AWB-HM algorithm (Auto White Balance using
# Histogram Matching), as specified in paper: C. Huang, Q. Zhang, H. Wang, and
# S.A. Feng, “Low Power and Low Complexity Automatic White Balance Algorithm
# for AMOLED Driving Using Histogram Matching,”

from __future__ import annotations  # for compatibility with Python 3.8

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer
from typing_extensions import TypeAlias

RGBImage: TypeAlias = Integer[Array, "height width 3"]
Index: TypeAlias = Integer[Array, ""]
Coef: TypeAlias = Float[Array, ""]


@partial(
    jax.jit,
    static_argnames=("bit_depth", ),
)
def balance(
    image: RGBImage,
    bit_depth: int = 8,
) -> RGBImage:
    """Apply automatic white balance of an image, using AWB-HM."""

    assert bit_depth > 0, f"bit_depth must be positive, got {bit_depth}"
    max_value: int = 2**bit_depth

    total_pixels: int = image.shape[0] * image.shape[1]

    channels: int = image.shape[2]

    # STAGE: Histogram Generation

    # number of bins == max_value
    bin_idx: Integer[Array, "3 height width"]
    histogram: Integer[Array, "3 bins"]
    values, _bin_idx, histogram = (
        jax.vmap(
            # compute histogram for one channel
            lambda img: jnp.unique(
                img.reshape((-1, )),
                return_inverse=True,
                return_counts=True,
                size=max_value,
                fill_value=max_value - 1,
            ),
            in_axes=(2, ),
        )(image))
    bin_idx = _bin_idx.reshape((channels, *image.shape[:2]))
    assert isinstance(values, Integer[Array, "3 bins"])
    assert isinstance(bin_idx, Integer[Array, "3 height width"])
    assert isinstance(histogram, Integer[Array, "3 bins"])

    # STAGE: Power Constraint

    channel_sums: Integer[Array, "3"] = image.sum(axis=(0, 1))
    assert isinstance(channel_sums, Integer[Array, "3"])

    k: Coef = channel_sums.mean() / channel_sums.max()
    assert isinstance(k, Coef)

    # selected channel to be matched against
    idx: Index = channel_sums.argmax()
    assert isinstance(idx, Index)
    target_values: Float[Array, "bins"] = values[idx] * k
    target_histogram: Integer[Array, "bins"] = histogram[idx]
    assert isinstance(target_values, Float[Array, "bins"])
    assert isinstance(target_histogram, Integer[Array, "bins"])

    # STAGE: Histogram Matching

    cdf: Integer[Array, "3 bins"] = jnp.cumsum(histogram, axis=1)
    assert isinstance(cdf, Integer[Array, "3 bins"])
    target_cdf: Integer[Array, "bins"] = jnp.cumsum(target_histogram)
    assert isinstance(target_cdf, Integer[Array, "bins"])

    result: RGBImage = (
        jax.vmap(
            # compute for one channel
            lambda source_cdf, source_idx: jnp.interp(
                source_cdf,
                target_cdf,
                target_values,
            )[source_idx],
            in_axes=0,
            out_axes=2,
        )(cdf, bin_idx)).astype(int)
    result = jnp.clip(result, 0, max_value - 1)
    assert isinstance(result, RGBImage)

    return result
