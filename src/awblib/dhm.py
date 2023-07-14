# Implementing the AWB-DHM algorithm (Auto White Balance using Dynamic
# Histogram Matching), as specified in paper: T. Gollanapalli, V. R. Peddigari
# and P. S. Madineni, "Auto white balance using dynamic histogram matching for
# AMOLED panels," 2017 IEEE International Conference on Consumer
# Electronics-Asia (ICCE-Asia), Bengaluru, India, 2017, pp. 41-46,
# doi: 10.1109/ICCE-ASIA.2017.8307848.

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
    static_argnames=("bit_depth", "always_mid_fix"),
)
def balance(
    image: RGBImage,
    bit_depth: int = 8,
    always_mid_fix: bool = False,
) -> RGBImage:
    """Apply automatic white balance of an image, using AWB-DHM."""

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

    # STAGE: Histogram Analysis

    # multiple of gray_levels, divide this by channels to get the true value
    gray_levels: Integer[Array, "height width"] = image.sum(axis=2)
    assert isinstance(gray_levels, Integer[Array, "height width"])

    lower_half_count: Integer[Array, ""]
    # equivalent to: image.mean(axis=2) < (max_value // 2)
    lower_half_count = (gray_levels < (max_value // 2 * channels)).sum()
    assert isinstance(lower_half_count, Integer[Array, ""])

    channel_intensity: Integer[Array, "3"] = image.sum(axis=(0, 1))

    def find_best_fit_channel_if_majority_above_half() -> tuple[Index, Coef]:
        """CRITERIA I."""
        # Step 1:
        channel_idx: Index = channel_intensity.argmax()
        assert isinstance(channel_idx, Index)

        # Step 2:
        """ NOTE: the paper is not clear about whether `Imax` is the maximum,
        the average, or something else. An educated guess is that it is the
        average, which is used here.
        """
        # Imax * total_pixels
        i_max: Integer[Array, ""] = channel_intensity[channel_idx]
        # Iavg * total_pixels
        i_avg: Integer[Array, ""] = channel_intensity.sum()

        # Iavg / Imax = (Iavg * total_pixels) / (Imax * total_pixels)
        k: Coef = i_avg / i_max

        return channel_idx, k

    def find_best_fit_channel_if_majority_below_half() -> tuple[Index, Coef]:
        """CRITERIA II."""
        # Step 1:
        # NOTE: the paper has a false claim that the dynamic range is 0-127
        channel_idx: Index = channel_intensity.argmin()
        assert isinstance(channel_idx, Index)

        # Step 2:

        ch_: Integer[Array, "3"]
        # replace the omitted channel value with a big enough value.
        ch_ = channel_intensity.at[channel_idx].set(total_pixels * max_value)
        idx: Index = jnp.where(
            # Average gray level > 63 for 8-bit images. 64 = 256 // 4
            # (gray * channels) * total_pixels >= ch * tot * (max // 4) => 64
            gray_levels.sum() >= (channels * total_pixels * max_value // 4),
            channel_idx,
            ch_.argmin(),
        )

        if always_mid_fix:
            """ NOTE: from the results section (fig 6) in the paper, the
                selected channel is actually usually the one with mid level of
                intensity.
                See docs/dhm/*.png for the comparison of the histogram and cdf
                from the original paper.
            """
            return ch_.argmin(), jnp.ones(())
        else:
            return idx, jnp.ones(())

    """ NOTE: The paper never says specifically what "majority" means. It is
        assumed that it means "more than half".
    """
    idx, k = jax.lax.cond(
        lower_half_count < total_pixels // 2,
        find_best_fit_channel_if_majority_above_half,
        find_best_fit_channel_if_majority_below_half,
    )
    """ NOTE: the paper is not clear about this operation. I guess they want to
        reduce the value of the best-fit channel using the scaling factor `K`.
        This operation is only meaningful for CRITERIA I; in CRITERIA II, `K`
        is always 1.
    """
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
