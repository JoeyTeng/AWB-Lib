""" Generate comparison of tilted vs fixed (paper) vs fixed (ours) vs fixed
with always_mid_fix, using sample images extracted from the paper's PDF.
"""
import matplotlib.pyplot as plt
import numpy as np

from awblib.dhm import balance
from awblib.utils import load_image, visualise_histogram

tilted = [chr(ord('a') + i) for i in range(9)]
prefix = "data/"
suffix = ".png"
tilted_suffix = "-tilted"
fixed_suffix = "-paper"

for stem in tilted:
    fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(18, 24))

    def plot(img, axs):
        axs[0].imshow(img)
        oa = visualise_histogram(img, ax=axs[1])
        axs[1].grid()
        visualise_histogram(img, ax=axs[2], cdf=True)
        axs[2].grid()

        return oa

    oas: list[float] = []  # oa = overlap area

    tilted_img = load_image(f"{prefix}{stem}{tilted_suffix}{suffix}")
    oa = plot(tilted_img.copy(), axs[0])
    oas.append(oa)

    oa = plot(load_image(f"{prefix}{stem}{fixed_suffix}{suffix}"), axs[1])
    oas.append(oa)

    ours = balance(tilted_img, always_mid_fix=False)
    oa = plot(np.asarray(ours), axs[2])
    oas.append(oa)

    ours_with_fix = balance(tilted_img, always_mid_fix=True)
    oa = plot(np.asarray(ours_with_fix), axs[3])
    oas.append(oa)

    # make room for table
    fig.subplots_adjust(top=0.95, bottom=0.1)

    # table of Overlap Area (OA)
    table_axes = fig.add_axes((0.53, 0.03, 0.1, 0.05))
    table_axes.axis("tight")
    table_axes.axis("off")
    table = table_axes.table(
        cellText=[[f"{oa:.12f}" for oa in oas]],
        rowLabels=["Overlap Area"],
        colLabels=[
            "Tilted", "Fixed (paper)", "Fixed (ours)", "Fixed (always_mid_fix)"
        ],
        loc="center",
    )
    table.scale(8, 4)
    table.auto_set_font_size(False)
    table.set_fontsize(21)

    fig.suptitle("tilted vs fixed (paper)"
                 " vs fixed (ours) vs fixed with always_mid_fix;"
                 " image with histogram & cdf")
    fig.savefig(f"{stem}.png")
