# Implementing the AWB-GW algorithm (Automatic White Balance with Gray World
# Assumption), as specified in paper: Shen-Chuan Tai, Tzu-Wen Liao, Yi-Ying
# Chang and Chih - Pei Yeh, "Automatic White Balance algorithm through the
# average equalization and threshold," 2012 8th International Conference on
# Information Science and Digital Content Technology (ICIDT2012), Jeju, Korea
# (South), 2012, pp. 571-576.
#
# This also implements a modified version of GW (Gray World).

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
)
def original_gw(image: RGBImage) -> RGBImage:
    """Apply automatic white balance of an image, using AWB-GW."""

    channel_sums: Integer[Array, "3"] = image.sum(axis=(0, 1))
    assert isinstance(channel_sums, Integer[Array, "3"])

    result: RGBImage
    # i * avg / c_avg = i * sum_avg / c_sum = i * sum / (c * c_sum)
    result = (image * channel_sums.sum() / (3 * channel_sums)).astype(int)
    assert isinstance(result, RGBImage)

    return result


@partial(
    jax.jit,
    inline=True,
)
def modified_gw(image: RGBImage) -> RGBImage:
    """Apply automatic white balance of an image, using AWB-GW, with
        modification introduced in the paper.

    Reference:
        Shen-Chuan Tai, Tzu-Wen Liao, Yi-Ying Chang and Chih - Pei Yeh,
        "Automatic White Balance algorithm through the average equalization and
        threshold," 2012 8th International Conference on Information Science
        and Digital Content Technology (ICIDT2012), Jeju, Korea (South), 2012,
        pp. 571-576.
    """

    channel_avgs: Float[Array, "3"] = image.mean(axis=(0, 1))
    assert isinstance(channel_avgs, Float[Array, "3"])

    global_avg: Float[Array, ""] = channel_avgs.mean()
    assert isinstance(global_avg, Float[Array, ""])

    result: RGBImage
    result = (image + (global_avg - channel_avgs)).astype(int)
    assert isinstance(result, RGBImage)

    return result


@partial(
    jax.jit,
    inline=True,
    static_argnames=("modified", ),
)
def balance(
    image: RGBImage,
    modified: bool = False,
) -> RGBImage:
    """Apply automatic white balance of an image, using AWB-GW."""

    result: RGBImage
    if modified:
        result = modified_gw(image)
    else:
        result = original_gw(image)
    assert isinstance(result, RGBImage)

    return result
