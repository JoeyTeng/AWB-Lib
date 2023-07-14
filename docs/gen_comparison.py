""" Generate comparison of tilted vs fixed (paper) vs fixed (ours), using
sample images extracted from the paper's PDF.
"""
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from awblib.aaet import balance as aaet
from awblib.chs import balance as chs
from awblib.dhm import balance as dhm
from awblib.gw import balance as gw
from awblib.hm import balance as hm
from awblib.utils import load_image, visualise_histogram

methods = [
    gw,
    partial(gw, modified=True), chs,
    partial(chs, modified=True), aaet, hm, dhm
]
methods_str = ["GW", "mod GW", "CHS", "mod CHS", "AAET", "HM", "DHM"]
prefix = "assets/"
suffix = ".jpg"
tilted_suffix = "tilted"
fixed_suffix = "fixed"

nrows: int = len(methods) + 2
fig, axs = plt.subplots(ncols=3, nrows=nrows, figsize=(18, nrows * 6))


def plot(img, axs):
    axs[0].imshow(img)
    oa = visualise_histogram(img, ax=axs[1])
    axs[1].grid()
    visualise_histogram(img, ax=axs[2], cdf=True)
    axs[2].grid()

    return oa


oas: list[float] = []  # oa = overlap area
tilted_img = load_image(f"{prefix}{tilted_suffix}{suffix}")
oa = plot(tilted_img.copy(), axs[0])
oas.append(oa)
axs[0][0].set_title(f"Tilted (OA: {oa:.12f})")

oa = plot(load_image(f"{prefix}{fixed_suffix}{suffix}"), axs[1])
oas.append(oa)
axs[1][0].set_title(f"Manually Fixed (OA: {oa:.12f})")

for i, (f, f_str) in enumerate(zip(methods, methods_str), start=2):
    ours = f(tilted_img.copy())
    oa = plot(np.asarray(ours), axs[i])
    oas.append(oa)

    axs[i][0].set_title(f"{f_str} (OA: {oa:.12f})")

# make room for table
fig.subplots_adjust(top=0.95, bottom=0.1)

# table of Overlap Area (OA)
table_axes = fig.add_axes((0.53, 0.03, 0.1, 0.05))
table_axes.axis("tight")
table_axes.axis("off")
table = table_axes.table(
    cellText=[[f"{oa:.7f}" for oa in oas]],
    rowLabels=["Overlap Area"],
    colLabels=["Tilted", "Manual", *methods_str],
    loc="center",
)
table.scale(8, 4)
table.auto_set_font_size(False)
table.set_fontsize(20)

fig.suptitle(f"tilted vs fixed (manual) vs AWB {methods_str};"
             " image with histogram & cdf")
fig.savefig(f"{prefix}report.png")
