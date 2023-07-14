# Implementing the AWBAAET algorithm (Automatic White Balance through the
# Average Equalization and Threshold), as specified in paper: Shen-Chuan Tai,
# Tzu-Wen Liao, Yi-Ying Chang and Chih - Pei Yeh, "Automatic White Balance
# algorithm through the average equalization and threshold," 2012 8th
# International Conference on Information Science and Digital Content
# Technology (ICIDT2012), Jeju, Korea (South), 2012, pp. 571-576.
#
# This also implements the GW (Gray World) and CHS (Color Histogram Stretching)
# algorithms.

from __future__ import annotations  # for compatibility with Python 3.8

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer
from typing_extensions import TypeAlias
from awblib.chs import modified_chs

from awblib.gw import modified_gw

RGBImage: TypeAlias = Integer[Array, "height width 3"]


@partial(
    jax.jit,
    inline=True,
    static_argnames=("standard", ),
)
def _convert_to_YCbCr(
    image: RGBImage,
    standard="BT.709",
) -> Float[Array, "height width 3"]:
    if standard == "BT.709":
        # from Wikipedia
        matrix: Float[Array, "3 3"] = jnp.array([
            [0.2126, 0.7152, 0.0722],
            [-0.1146, -0.3854, 0.5],
            [0.5, -0.4542, -0.0458],
        ])
    else:
        raise ValueError(f"Unsupported standard: {standard}")

    assert isinstance(matrix, Float[Array, "3 3"])

    result: Float[Array, "height width 3"]
    result = jax.vmap(jax.vmap(lambda pixel: jnp.dot(matrix, pixel)))(image)
    assert isinstance(result, Float[Array, "height width 3"])

    return result


@partial(
    jax.jit,
    inline=True,
    static_argnames=("n", ),
)
def _weight(
    image: RGBImage,
    n: float = 200.,
) -> Float[Array, ""]:
    """Compute the weight of an image, as specified in the paper.

    Parameters:
      - image: the image to be processed
      - n: a empirical value, as specified in the paper. They use 200.
    """
    # NOTE: the paper's formula (7) is variance, not standard deviation!
    var_channels: Float[Array, "3"] = image.var(axis=(0, 1))
    assert isinstance(var_channels, Float[Array, "3"])
    """ NOTE: the paper did not specify what standard it is using, so we assume
        the raw input is in sRGB and the target YCbCr is in BT.709.
    """
    YCbCr: Float[Array, "height width 3"]
    YCbCr = _convert_to_YCbCr(image, standard="BT.709")
    assert isinstance(YCbCr, Float[Array, "height width 3"])
    mean_cbcr: Float[Array, "2"] = YCbCr[:, :, 1:].mean(axis=(0, 1))
    diff_mean_cbcr: Float[Array, ""] = jnp.abs(mean_cbcr[0] - mean_cbcr[1])
    assert isinstance(diff_mean_cbcr, Float[Array, ""])

    weight: Float[Array, ""]
    weight = (diff_mean_cbcr + (var_channels.max() - var_channels.min())) / n
    assert isinstance(weight, Float[Array, ""])

    return weight


@partial(
    jax.jit,
    inline=True,
    static_argnames=("bit_depth", "n"),
)
def balance(
    image: RGBImage,
    bit_depth: int = 8,
    n: float = 200.,
) -> RGBImage:
    gw: RGBImage = modified_gw(image)
    chs: RGBImage = modified_chs(image, bit_depth=bit_depth)
    weight: Float[Array, ""] = _weight(image, n=n)

    result: RGBImage
    result = ((1 - weight) * gw + weight * chs).astype(int)
    assert isinstance(result, RGBImage)

    return result
