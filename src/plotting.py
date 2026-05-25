"""Matplotlib figures built from the results CSVs.

Presentation only: the data each figure plots is read straight from the
results/data CSVs and is not modified here. Output is static SVG with a single
cohesive visual system (see PALETTE and the style helpers below). Runs headless
on the Agg backend.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

# One cohesive palette (seaborn "deep" family). cache off is red, cache on is
# blue, and those two are held identical across every figure.
PALETTE = {
    "cache_off": "#C44E52",   # red
    "cache_on": "#4C72B0",    # blue
    "theoretical": "#55A868",  # green
    "measured": "#C44E52",    # red (same accent as cache off)
    "mha": "#DD8452",         # orange
    "gqa": "#4C72B0",         # blue
    "mha_equiv": "#BFBFBF",   # light gray reference
}
LABEL = {"cache_off": "cache off", "cache_on": "cache on"}

INK = "#2F2F2F"      # primary text and ticks
GRID = "#E4E4E4"     # thin, unobtrusive gridlines
SPINE = "#BBBBBB"     # light remaining spines


def _setup_style():
    """Global, cohesive look: light background, sans-serif, restrained type."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",          # keep text as crisp, light SVG text
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "axes.grid": False,
    })


_setup_style()


def _clean_axes(ax, grid_axis="y"):
    """Drop the top and right spines, lighten the rest, add thin gridlines."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(length=4, width=0.9, colors=INK)
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.9)


def _title(ax, text):
    ax.set_title(text, loc="left", fontsize=15, fontweight="bold", color=INK, pad=12)


def _end_label(ax, x, y, text, color):
    """Direct line label just past the last point, in the line's own color."""
    ax.annotate(text, xy=(x, y), xytext=(8, 0), textcoords="offset points",
                va="center", ha="left", color=color, fontsize=12,
                fontweight="bold", clip_on=False)


def _save(fig, out_path):
    fig.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def plot_per_step_latency(out_path=None):
    """Figure 1: median per-step decode latency vs sequence length, both modes."""
    out_path = out_path or (config.PLOTS_DIR / "fig1_per_step_latency.svg")
    df = pd.read_csv(config.LATENCY_CSV)
    agg = (df.groupby(["mode", "sequence_length"])["step_latency_ms"]
             .median().reset_index())

    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in ("cache_off", "cache_on"):
        sub = agg[agg["mode"] == mode].sort_values("sequence_length")
        x, y = sub["sequence_length"].values, sub["step_latency_ms"].values
        ax.plot(x, y, color=PALETTE[mode], linewidth=2.4, marker="o",
                markersize=7, markeredgecolor="white", markeredgewidth=1.1, zorder=3)
        _end_label(ax, x[-1], y[-1], LABEL[mode], PALETTE[mode])

    ax.set_xscale("log", base=2)
    ax.set_xticks(config.SEQUENCE_LENGTHS)
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlim(27, 2300)
    ax.set_xlabel("Starting sequence length (tokens)")
    ax.set_ylabel("Median latency per step (ms)")
    _title(ax, "Per-step decode latency vs sequence length")
    _clean_axes(ax)
    _save(fig, out_path)
    return out_path


def plot_per_step_by_position(out_path=None, start_length=1024):
    """Figure 2: measured per-step latency vs decode step at a fixed start.

    Direct within-run evidence. With the cache off, each successive step feeds a
    longer sequence, so per-step latency climbs with the decode step; with the
    cache on it stays flat. (True sequence length = ``start_length`` + step.)
    L=1024 is used because at shorter contexts the fixed per-forward overhead
    swamps the length-dependent term.
    """
    out_path = out_path or (config.PLOTS_DIR / "fig2_per_step_by_position.svg")
    df = pd.read_csv(config.LATENCY_CSV)
    df = df[df["sequence_length"] == start_length]

    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in ("cache_off", "cache_on"):
        sub = df[df["mode"] == mode].sort_values("step_index")
        x, y = sub["step_index"].values, sub["step_latency_ms"].values
        ax.plot(x, y, color=PALETTE[mode], linewidth=1.7, zorder=3)
        _end_label(ax, x[-1], y[-1], LABEL[mode], PALETTE[mode])

    ax.set_xlim(0, 145)
    ax.set_xlabel(f"Decode step  (sequence length = {start_length} + step)")
    ax.set_ylabel("Latency per step (ms)")
    _title(ax, f"Per-step decode latency by decode step ({start_length}-token start)")
    _clean_axes(ax)
    _save(fig, out_path)
    return out_path


def plot_cumulative_decode(out_path=None):
    """Figure 3: cumulative decode time vs tokens generated, both modes.

    MODELED curve. Generating the token at position n costs one decode step at
    sequence length n; we take the measured per-step latency as a function of
    length (the Figure 1 grid) and accumulate it over positions:
    cumulative(N) = sum over n<=N of per_step(n). This is interpolated from the
    grid, not a single recorded run, but it shows the scaling cleanly over the
    full length range: cache-off cost grows super-linearly (quadratic) because
    per-step rises with n, while cache-on cost is linear.
    """
    out_path = out_path or (config.PLOTS_DIR / "fig3_cumulative_decode.svg")
    df = pd.read_csv(config.LATENCY_CSV)
    agg = (df.groupby(["mode", "sequence_length"])["step_latency_ms"]
             .median().reset_index())

    positions = np.arange(1, max(config.SEQUENCE_LENGTHS) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in ("cache_off", "cache_on"):
        sub = agg[agg["mode"] == mode].sort_values("sequence_length")
        per_step = np.interp(positions, sub["sequence_length"], sub["step_latency_ms"])
        cumulative_s = np.cumsum(per_step) / 1000.0
        ax.plot(positions, cumulative_s, color=PALETTE[mode], linewidth=2.6, zorder=3)
        _end_label(ax, positions[-1], cumulative_s[-1], LABEL[mode], PALETTE[mode])

    ax.set_xlim(0, 1180)
    ax.set_xlabel("Tokens generated (decode position)")
    ax.set_ylabel("Cumulative decode time (s)")
    _title(ax, "Cumulative decode cost, modeled from the per-step grid")
    _clean_axes(ax)
    _save(fig, out_path)
    return out_path


def plot_memory(out_path=None):
    """Figure 4: measured vs theoretical KV cache memory vs sequence length."""
    out_path = out_path or (config.PLOTS_DIR / "fig4_kv_cache_memory.svg")
    df = pd.read_csv(config.MEMORY_CSV).sort_values("sequence_length")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["sequence_length"], df["theoretical_mb"], color=PALETTE["theoretical"],
            linewidth=2.6, label="theoretical", zorder=2)
    ax.scatter(df["sequence_length"], df["measured_mb"], color=PALETTE["measured"],
               s=70, edgecolor="white", linewidth=1.1, label="measured", zorder=4)
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("KV cache memory (MB)")
    _title(ax, "KV cache memory: measured vs theoretical")
    _clean_axes(ax)
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out_path)
    return out_path


def plot_attention_variants(out_path=None):
    """Figure 5: KV cache bytes/token across attention variants."""
    out_path = out_path or (config.PLOTS_DIR / "fig5_attention_variants.svg")
    df = pd.read_csv(config.ATTENTION_CSV)

    labels, values, colors = [], [], []
    for _, r in df.iterrows():
        short = r["model"].split("/")[-1]
        labels.append(f"{short}\n({r['attention_type']})")
        values.append(int(r["kv_bytes_per_token"]))
        colors.append(PALETTE["gqa"] if r["attention_type"] == "GQA" else PALETTE["mha"])
        if r["attention_type"] == "GQA":
            # Reference bar: same model under full MHA, to show the reduction.
            labels.append(f"{short}\n(MHA equiv.)")
            values.append(int(r["mha_equiv_bytes_per_token"]))
            colors.append(PALETTE["mha_equiv"])

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, width=0.62, zorder=3)
    ax.bar_label(bars, labels=[f"{v:,} B" for v in values], padding=4,
                 fontsize=11, color=INK, fontweight="bold")
    ax.set_ylabel("KV cache bytes per token")
    _title(ax, "KV cache footprint per token: MHA vs GQA")
    _clean_axes(ax)
    ax.margins(y=0.18)
    _save(fig, out_path)
    return out_path


def generate_all():
    return [
        plot_per_step_latency(),
        plot_per_step_by_position(),
        plot_cumulative_decode(),
        plot_memory(),
        plot_attention_variants(),
    ]
