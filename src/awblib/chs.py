# Implementing the CHS algorithm (Color Histogram Stretching), as specified in paper: S. Wang, Y. Zhang, P. Deng and F. Zhou, "Fast automatic white balancing method by color histogram stretching," 2011 4th International Congress on Image and Signal Processing, Shanghai, China, 2011, pp. 979-983, doi: 10.1109/CISP.2011.6100338.
#
# This also implements a modified version of the GHS algorithm, as specified in
# another paper: Shen-Chuan Tai, Tzu-Wen Liao, Yi-Ying Chang and Chih - Pei
# Yeh, "Automatic White Balance algorithm through the average equalization and
# threshold," 2012 8th International Conference on Information Science and
# Digital Content Technology (ICIDT2012), Jeju, Korea (South), 2012,
# pp. 571-576.

from __future__ import annotations  # for compatibility with Python 3.8

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer
from typing_extensions import TypeAlias

RGBImage: TypeAlias = Integer[Array, "height width 3"]


@partial(
    jax.jit,
    inline=True,
    static_argnames=("bit_depth"),
)
def original_chs(
    image: RGBImage,
    bit_depth: int = 8,
    RGB_range: Optional[Integer[Array, "3 2"]] = None,
) -> RGBImage:
    """Apply automatic white balance of an image, using AWB-CHS
        (Colour Histogram Stretching).

    Parameters:
      - image: the image to be processed
      - bit_depth: the bit depth of the image
      - RGB_range: the range of the RGB values, as a 3x2 array. The rows
        corresponds to R, G, B channels respectively; the columns corresponds
        to the minimum and maximum possible values of that channel. If None,
        the range is assumed to be [0, 2**bit_depth - 1] for all channels.
    """

    assert bit_depth > 0, f"bit_depth must be positive, got {bit_depth}"
    max_value: int = 2**bit_depth
    total_pixels: int = image.shape[0] * image.shape[1]

    with jax.ensure_compile_time_eval():
        if RGB_range is None:
            RGB_range = jnp.array([[0, max_value - 1]] * 3)

        assert isinstance(RGB_range, Integer[Array, "3 2"]), RGB_range

    # STAGE: Detect Ilow and Ihigh

    # number of bins == max_value
    values: Integer[Array, "3 bins"]
    histogram: Integer[Array, "3 bins"]
    values, histogram = (
        jax.vmap(
            # compute histogram for one channel
            lambda img: jnp.unique(
                img.reshape((-1, )),
                return_counts=True,
                size=max_value,
                fill_value=max_value - 1,
            ),
            in_axes=(2, ),
        )(image))
    assert isinstance(values, Integer[Array, "3 bins"])
    assert isinstance(histogram, Integer[Array, "3 bins"])

    cdf: Integer[Array, "3 bins"] = histogram.cumsum(axis=1)
    assert isinstance(cdf, Integer[Array, "3 bins"])

    # find the first index where the cumulative sum is greater than .01 * total
    low = jax.vmap(lambda i, _values: _values[i])(
        jnp.argmax(cdf > 0.01 * total_pixels, axis=1),
        values,
    )
    assert isinstance(low, Integer[Array, "3"])
    # find the first index where the cumulative sum is greater than .99 * total
    high = jax.vmap(lambda i, _values: _values[i])(
        jnp.argmax(cdf > 0.99 * total_pixels, axis=1),
        values,
    )
    assert isinstance(high, Integer[Array, "3"])

    _max: Integer[Array, "3"] = RGB_range[:, 1].astype(int)
    _min: Integer[Array, "3"] = RGB_range[:, 0].astype(int)
    assert isinstance(_max, Integer[Array, "3"])
    assert isinstance(_min, Integer[Array, "3"])

    # STAGE: Histogram Stretching
    result: RGBImage
    result = ((image - low) * _max / (high - low) + _min).astype(int)
    result = jnp.clip(result, a_min=0, a_max=max_value - 1).astype(int)
    assert isinstance(result, RGBImage)

    return result


@partial(
    jax.jit,
    inline=True,
    static_argnames=("bit_depth"),
)
def modified_chs(
    image: RGBImage,
    bit_depth: int = 8,
) -> RGBImage:
    """Apply automatic white balance of an image, using AWB-CHS
        (Colour Histogram Stretching)

    with modification introduced in the paper: Automatic White Balance
    algorithm through the average equalization and threshold.

    Parameters:
      - image: the image to be processed
      - bit_depth: the bit depth of the image
    """

    assert bit_depth > 0, f"bit_depth must be positive, got {bit_depth}"
    max_value: int = 2**bit_depth

    total_pixels: int = image.shape[0] * image.shape[1]

    # STAGE: Detect Ilow and Ihigh

    # number of bins == max_value
    values: Integer[Array, "3 bins"]
    histogram: Integer[Array, "3 bins"]
    values, histogram = (
        jax.vmap(
            # compute histogram for one channel
            lambda img: jnp.unique(
                img.reshape((-1, )),
                return_counts=True,
                size=max_value,
                fill_value=max_value - 1,
            ),
            in_axes=(2, ),
        )(image))
    assert isinstance(values, Integer[Array, "3 bins"])
    assert isinstance(histogram, Integer[Array, "3 bins"])

    cdf: Integer[Array, "3 bins"] = histogram.cumsum(axis=1)
    assert isinstance(cdf, Integer[Array, "3 bins"])

    # find the first index where the cumulative sum is greater than .01 * total
    low = jax.vmap(lambda i, _values: _values[i])(
        jnp.argmax(cdf > 0.01 * total_pixels, axis=1),
        values,
    )
    assert isinstance(low, Integer[Array, "3"])
    # find the first index where the cumulative sum is greater than .99 * total
    high = jax.vmap(lambda i, _values: _values[i])(
        jnp.argmax(cdf > 0.99 * total_pixels, axis=1),
        values,
    )
    assert isinstance(high, Integer[Array, "3"])

    a_max: Integer[Array, ""] = high.max().astype(int)  # A_max
    a_min: Integer[Array, ""] = low.min().astype(int)  # A_min
    assert isinstance(a_max, Integer[Array, ""])
    assert isinstance(a_min, Integer[Array, ""])

    # STAGE: Histogram Stretching
    result: RGBImage
    result = ((image - low) * a_max / (high - low) + a_min).astype(int)
    result = jnp.clip(result, a_min=0, a_max=max_value - 1).astype(int)
    assert isinstance(result, RGBImage)

    return result


@partial(
    jax.jit,
    inline=True,
    static_argnames=("bit_depth", "modified"),
)
def balance(
    image: RGBImage,
    bit_depth: int = 8,
    RGB_range: Optional[Integer[Array, "3 2"]] = None,
    modified: bool = False,
) -> RGBImage:
    """Apply automatic white balance of an image, using CHS method."""

    result: RGBImage
    if modified:
        result = modified_chs(image, bit_depth=bit_depth)
    else:
        result = original_chs(image, bit_depth=bit_depth, RGB_range=RGB_range)
    assert isinstance(result, RGBImage)

    return result
