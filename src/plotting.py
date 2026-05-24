"""Matplotlib figures built from the results CSVs (headless / Agg backend)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

_MODE_STYLE = {
    "cache_off": {"label": "cache off", "color": "#d62728", "marker": "o"},
    "cache_on": {"label": "cache on", "color": "#1f77b4", "marker": "s"},
}


def plot_per_step_latency(out_path=None):
    """Figure 1: median per-step decode latency vs sequence length, both modes."""
    out_path = out_path or (config.PLOTS_DIR / "fig1_per_step_latency.png")
    df = pd.read_csv(config.LATENCY_CSV)
    agg = (df.groupby(["mode", "sequence_length"])["step_latency_ms"]
             .median().reset_index())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode, style in _MODE_STYLE.items():
        sub = agg[agg["mode"] == mode].sort_values("sequence_length")
        ax.plot(sub["sequence_length"], sub["step_latency_ms"],
                marker=style["marker"], color=style["color"], label=style["label"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(config.SEQUENCE_LENGTHS)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Starting sequence length (tokens)")
    ax.set_ylabel("Median per-step decode latency (ms)")
    ax.set_title("Per-step decode latency vs sequence length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_cumulative_decode(out_path=None):
    """Figure 2: cumulative decode time vs tokens generated, both modes.

    Generating the token at position n costs one decode step at sequence length
    n. We take the measured per-step latency as a function of length (the
    Figure-1 curve) and accumulate it over positions: cumulative(N) =
    sum over n<=N of per_step(n). Cache-off per-step rises with n, so its
    cumulative cost grows super-linearly (quadratic); cache-on per-step is flat,
    so its cumulative cost is linear.
    """
    out_path = out_path or (config.PLOTS_DIR / "fig2_cumulative_decode.png")
    df = pd.read_csv(config.LATENCY_CSV)
    agg = (df.groupby(["mode", "sequence_length"])["step_latency_ms"]
             .median().reset_index())

    positions = np.arange(1, max(config.SEQUENCE_LENGTHS) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode, style in _MODE_STYLE.items():
        sub = agg[agg["mode"] == mode].sort_values("sequence_length")
        # per-step latency as a function of decode position (interp from grid).
        per_step = np.interp(positions, sub["sequence_length"], sub["step_latency_ms"])
        cumulative_s = np.cumsum(per_step) / 1000.0
        ax.plot(positions, cumulative_s, color=style["color"], label=style["label"])
    ax.set_xlabel("Tokens generated (decode position)")
    ax.set_ylabel("Cumulative decode time (s)")
    ax.set_title("Cumulative decode cost vs tokens generated\n"
                 "(accumulated from measured per-step latency)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_memory(out_path=None):
    """Figure 3: measured vs theoretical KV cache memory vs sequence length."""
    out_path = out_path or (config.PLOTS_DIR / "fig3_kv_cache_memory.png")
    df = pd.read_csv(config.MEMORY_CSV).sort_values("sequence_length")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["sequence_length"], df["theoretical_mb"],
            color="#2ca02c", linewidth=2, label="theoretical")
    ax.scatter(df["sequence_length"], df["measured_mb"],
               color="#d62728", zorder=5, label="measured")
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("KV cache memory (MB)")
    ax.set_title("KV cache memory grows linearly; measured matches theory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_attention_variants(out_path=None):
    """Figure 4: KV cache bytes/token across attention variants."""
    out_path = out_path or (config.PLOTS_DIR / "fig4_attention_variants.png")
    df = pd.read_csv(config.ATTENTION_CSV)

    labels, values, colors = [], [], []
    for _, r in df.iterrows():
        short = r["model"].split("/")[-1]
        labels.append(f"{short}\n({r['attention_type']})")
        values.append(r["kv_bytes_per_token"])
        colors.append("#1f77b4" if r["attention_type"] == "GQA" else "#ff7f0e")
        if r["attention_type"] == "GQA":
            # Reference bar: same model under full MHA, to show the reduction.
            labels.append(f"{short}\n(MHA-equiv.)")
            values.append(r["mha_equiv_bytes_per_token"])
            colors.append("#cccccc")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, fmt="%d B", padding=3)
    ax.set_ylabel("KV cache bytes per token")
    ax.set_title("KV cache footprint per token: MHA vs GQA")
    ax.margins(y=0.15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def generate_all():
    return [
        plot_per_step_latency(),
        plot_cumulative_decode(),
        plot_memory(),
        plot_attention_variants(),
    ]
