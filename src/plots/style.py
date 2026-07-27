"""
Canonical plot styling shared across the analysis / report notebooks.

Single source of truth for:
- ``DATASET_COLORS`` / ``DATASET_LABELS`` — dataset -> colour / display label
- ``apply_ppt_style_mpl`` / ``apply_ppt_style_plotly`` — 16:9 presentation styling

The colours are identical across notebooks; historically only the *key spelling*
differed for raster datasets:

    "wsf_tracker@10m"   (nb06/07, "@grid" suffix)
    "wsf_tracker_10m"   (nb90/91, "_grid" suffix)
    "wsf_tracker"       (base name, when metrics are aggregated across grids)

``DATASET_COLORS`` and ``DATASET_LABELS`` below include *every* spelling mapped
to the same value, so each notebook's existing lookups keep working unchanged.

Notebooks import from here after ``colab_bootstrap`` has put the repo on the path::

    from src.plots.style import DATASET_COLORS, DATASET_LABELS, apply_ppt_style_plotly as apply_ppt_style
"""
from __future__ import annotations

# --- canonical fonts / backdrop -------------------------------------------
FONT_FAMILY = "Barlow, Arial, sans-serif"   # Plotly notebooks (nb07/90/91)
FONT_FAMILY_MPL = "Barlow"                   # matplotlib notebook (nb06)
FONT_COLOR = "#1a1a1a"
BG_COLOR = "white"

# --- canonical colours ----------------------------------------------------
_OVERTURE = "#1B6CA8"
_GBA      = "#2BAE82"
_GLOBFP   = "#66C7E0"
_WSF_10   = "#C94A35"
_WSF_100  = "#EFB3A9"
_OBT_10   = "#E0882A"
_OBT_100  = "#F5C98A"
_GHSL     = "#CC2266"
_TEMPO    = "#7B4EBD"

DATASET_COLORS = {
    # vector
    "overture": _OVERTURE,
    "gba":      _GBA,
    "globfp":   _GLOBFP,
    # raster — "@grid" spelling (nb06/07)
    "wsf_tracker@10m":        _WSF_10,
    "wsf_tracker@100m":       _WSF_100,
    "obt_2023@10m":           _OBT_10,
    "obt_2023@100m":          _OBT_100,
    "ghsl_built_s_2025@100m": _GHSL,
    "tempo_2023q4@100m":      _TEMPO,
    # raster — "_grid" spelling (nb90/91)
    "wsf_tracker_10m":        _WSF_10,
    "wsf_tracker_100m":       _WSF_100,
    "obt_2023_10m":           _OBT_10,
    "obt_2023_100m":          _OBT_100,
    # raster — base names (metrics aggregated across grid; unique 100m datasets)
    "wsf_tracker":       _WSF_10,
    "obt_2023":          _OBT_10,
    "ghsl_built_s_2025": _GHSL,
    "tempo_2023q4":      _TEMPO,
}

DATASET_LABELS = {
    # vector
    "overture": "Overture Maps",
    "gba":      "Global Building Atlas",
    "globfp":   "GlobFP",
    # raster — "@grid" spelling
    "wsf_tracker@10m":        "WSF Tracker (10m)",
    "wsf_tracker@100m":       "WSF Tracker (100m)",
    "obt_2023@10m":           "OBT 2023 (10m)",
    "obt_2023@100m":          "OBT 2023 (100m)",
    "ghsl_built_s_2025@100m": "GHSL 2025 (100m)",
    "tempo_2023q4@100m":      "TEMPO Q4 2023 (100m)",
    # raster — "_grid" spelling
    "wsf_tracker_10m":        "WSF Tracker (10m)",
    "wsf_tracker_100m":       "WSF Tracker (100m)",
    "obt_2023_10m":           "OBT 2023 (10m)",
    "obt_2023_100m":          "OBT 2023 (100m)",
    # raster — base names
    "wsf_tracker":       "WSF Tracker",
    "obt_2023":          "OBT 2023",
    "ghsl_built_s_2025": "GHSL 2025 (100m)",
    "tempo_2023q4":      "TEMPO Q4 2023 (100m)",
}


def dataset_color(name, default: str = "#AAAAAA") -> str:
    """Look up a dataset's colour (case-insensitive, whitespace-stripped)."""
    return DATASET_COLORS.get(str(name).strip().lower(), default)


def apply_ppt_style_mpl(fig, ax_or_axes, font_family: str = FONT_FAMILY_MPL):
    """Apply 16:9 presentation styling to a matplotlib figure and its axes."""
    fig.set_size_inches(1280 / 96, 720 / 96)
    axes = ax_or_axes if hasattr(ax_or_axes, "__iter__") else [ax_or_axes]
    for ax in axes:
        ax.title.set_fontsize(20)
        ax.title.set_fontfamily(font_family)
        ax.xaxis.label.set_fontsize(15)
        ax.yaxis.label.set_fontsize(15)
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(13)
            label.set_fontfamily(font_family)
    fig.tight_layout()


def apply_ppt_style_plotly(fig, title=None, height=720, width=1280,
                           font_family: str = FONT_FAMILY,
                           font_color: str = FONT_COLOR,
                           bg_color: str = BG_COLOR):
    """Apply 16:9 presentation styling to a Plotly figure; returns the figure."""
    fig.update_layout(
        font=dict(family=font_family, color=font_color, size=15),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        height=height,
        width=width,
        title_font=dict(size=20, family=font_family, color=font_color),
        title_x=0.5,
        title_xanchor="center",
        legend=dict(
            font=dict(size=13, family=font_family),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#CCCCCC",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        title_font=dict(size=15, family=font_family),
        tickfont=dict(size=13, family=font_family),
        gridcolor="#E8E8E8",
    )
    fig.update_yaxes(
        title_font=dict(size=15, family=font_family),
        tickfont=dict(size=13, family=font_family),
        gridcolor="#E8E8E8",
    )
    if title is not None:
        fig.update_layout(title_text=title)
    return fig
