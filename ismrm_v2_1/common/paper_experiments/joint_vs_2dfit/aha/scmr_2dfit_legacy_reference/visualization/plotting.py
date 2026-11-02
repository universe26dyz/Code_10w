"""Shared headless plotting helpers for evaluation figures."""

from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-2dfit-evaluation")

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_figure(fig, path: str | Path, dpi: int = 200, pdf: bool = False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=dpi,bbox_inches="tight",facecolor="white")
    if pdf: fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight",facecolor="white")
    plt.close(fig)


def show_image(ax, image, title="", cmap="magma", limits=None, mask=None):
    a=np.asarray(image,float)
    if mask is not None: a=np.where(mask,a,np.nan)
    kwargs={"cmap":cmap,"origin":"lower"}
    if limits: kwargs.update(vmin=limits[0],vmax=limits[1])
    im=ax.imshow(a,**kwargs); ax.set_title(title,fontsize=8); ax.axis("off"); return im


def parameter_limits(cfg, parameter):
    return tuple(cfg.get("figures",{}).get("display_ranges_ms",{}).get(parameter,[0,2500 if parameter=="T1" else 200]))


def residual_limit(cfg, parameter):
    return float(cfg.get("figures",{}).get("residual_ranges_ms",{}).get(parameter,300 if parameter=="T1" else 30))


def bullseye(ax, values, vmin, vmax, cmap="viridis", title="", *,
             show_segment_numbers=True, value_fontsize=6, title_fontsize=None,
             title_fontweight=None):
    from matplotlib.patches import Wedge
    norm=plt.Normalize(vmin,vmax); cm=plt.get_cmap(cmap)
    rings=[(0.66,1.0,range(1,7)),(0.34,0.66,range(7,13)),(0.08,0.34,range(13,17))]
    for inner,outer,segments in rings:
        segs=list(segments); width=360/len(segs)
        for i,s in enumerate(segs):
            val=values.get(s,np.nan); color=cm(norm(val)) if np.isfinite(val) else "lightgray"
            wedge=Wedge((0,0),outer,90-(i+1)*width,90-i*width,width=outer-inner,
                        facecolor=color,edgecolor="white",linewidth=1)
            ax.add_patch(wedge); theta=np.deg2rad(90-(i+0.5)*width); r=(inner+outer)/2
            label = (f"{s}\n{val:.0f}" if show_segment_numbers else f"{val:.0f}") if np.isfinite(val) \
                else (str(s) if show_segment_numbers else "")
            ax.text(r*np.cos(theta),r*np.sin(theta),label,
                    ha="center",va="center",fontsize=value_fontsize)
    ax.set(xlim=(-1.05,1.05),ylim=(-1.05,1.05),aspect="equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
    return plt.cm.ScalarMappable(norm=norm,cmap=cm)
