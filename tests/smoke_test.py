""" Generate comparison of tilted vs fixed (paper) vs fixed (ours), using
sample images extracted from the paper's PDF.
"""
from functools import partial

from awblib.aaet import balance as aaet
from awblib.chs import balance as chs
from awblib.dhm import balance as dhm
from awblib.gw import balance as gw
from awblib.hm import balance as hm
from awblib.utils import load_image

methods = [
    gw,
    partial(gw, modified=True), chs,
    partial(chs, modified=True), aaet, hm, dhm
]
prefix = "assets/"
suffix = ".jpg"
tilted_suffix = "tilted"

tilted_img = load_image(f"{prefix}{tilted_suffix}{suffix}")

for f in methods:
    ours = f(tilted_img.copy())
