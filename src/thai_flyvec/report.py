"""Tables and figures that answer the three research questions, from the CSVs of
`scripts/run_experiments.py`.

1. Similarity of FlyVec vs word2vec / fastText -> `summary_table`, `plot_vs_k`
2. Sentiment accuracy vs memory and training time -> `plot_vs_resources`
3. Best hash length k -> `plot_vs_k`, `best_k`
"""

from __future__ import annotations

import pandas as pd

MODEL_ORDER = ("flyvec", "word2vec_binary", "word2vec", "fasttext", "fasttext_subword")
MODEL_LABELS = {
    "flyvec": "FlyVec",
    "word2vec_binary": "word2vec binary",
    "word2vec": "word2vec",
    "fasttext": "fastText",
    "fasttext_subword": "fastText + subword OOV (all pairs)",
}
SCORES = {
    ("th-wordsim-353", "rho"): "TH-WordSim-353 ρ",
    ("th-simlex-999", "rho"): "TH-SimLex-999 ρ",
    ("wisesight", "accuracy"): "wisesight accuracy",
    ("wisesight", "macro_f1"): "wisesight macro-F1",
}
BINARY_MODELS = ("flyvec", "word2vec_binary")
RESOURCE_KIND = {"flyvec": "flyvec", "word2vec_binary": "word2vec", "word2vec": "word2vec",
                 "fasttext": "fasttext", "fasttext_subword": "fasttext"}  # fmt: skip


def to_markdown(df: pd.DataFrame) -> str:
    """A GitHub markdown table (pandas' own needs the optional `tabulate`)."""
    cells = [[str(c) for c in df.columns]] + [["" if pd.isna(v) else str(v) for v in row] for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(row) + " |" for row in cells]
    lines.insert(1, "|" + "---|" * len(df.columns))
    return "\n".join(lines)


def load(results_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    from pathlib import Path

    results_dir = Path(results_dir)
    return pd.read_csv(results_dir / "raw_results.csv"), pd.read_csv(results_dir / "resources.csv")


def scores(raw: pd.DataFrame, pairs: str = "common") -> pd.DataFrame:
    """One row per (model, k, w, seed) and one column per score in SCORES.

    Similarity uses the pair set `pairs` ("common": the pairs every model
    knows, D7; "all": every pair, which only fastText + subword has). On
    "common" the fastText + subword row equals the fastText row, so
    `summary_table` reports it with `pairs="all"`.
    """
    raw = _key(raw)
    keep = raw[[(t, m) in SCORES for t, m in zip(raw.task, raw.metric)]]
    keep = keep[(keep.task == "wisesight") | (keep.pairs.fillna("") == pairs)]
    keep = keep.assign(score=[SCORES[(t, m)] for t, m in zip(keep.task, keep.metric)])
    table = keep.pivot_table(index=["model", "k", "w", "seed"], columns="score", values="value")
    return table.reindex(columns=[c for c in SCORES.values() if c in table.columns]).reset_index()


def _key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["k"] = df["k"].fillna(-1).astype(int)
    return df


def summary_table(raw: pd.DataFrame, w: int | None = None) -> pd.DataFrame:
    """Mean ± SD over seeds of every score, per model and k (and w unless fixed).

    Similarity is on the pairs every model knows, except the fastText + subword
    row, which is on all pairs: it shows how far subwords get on OOV words (D6).
    """
    table, on_all = scores(raw), scores(raw, pairs="all")
    subword = "fasttext_subword"
    table = pd.concat([table[table.model != subword], on_all[on_all.model == subword]])
    if w is not None:
        table = table[table.w == w]
    by = ["model", "k"] if w is not None else ["model", "k", "w"]
    table = table.drop(columns=[c for c in ("w", "seed") if c not in by])
    stats = table.groupby(by).agg(["mean", "std"])
    out = pd.DataFrame(index=stats.index)
    for score in SCORES.values():
        if (score, "mean") in stats.columns:
            out[score] = [
                f"{m:.3f} ± {s:.3f}" if pd.notna(s) else (f"{m:.3f}" if pd.notna(m) else "")
                for m, s in zip(stats[(score, "mean")], stats[(score, "std")])
            ]
    out = out.reset_index()
    out["order"] = out.model.map({m: i for i, m in enumerate(MODEL_ORDER)})
    out = out.sort_values(["order", "k"] + (["w"] if w is None else [])).drop(columns="order")
    out["model"] = out.model.map(MODEL_LABELS)
    out["k"] = out["k"].map(lambda k: "" if k < 0 else str(k))
    return out.reset_index(drop=True)


def oov_table(raw: pd.DataFrame) -> pd.DataFrame:
    """OOV pair ratio of the in-vocabulary models vs fastText with subwords."""
    df = raw[(raw.metric == "oov_ratio") & raw.task.isin(["th-wordsim-353", "th-simlex-999"])]
    df = df[(df.model == "word2vec") | ((df.model == "fasttext_subword") & (df.pairs == "all"))]
    return df.groupby(["task", "model"])["value"].mean().unstack()


def best_k(raw: pd.DataFrame, score: str = "TH-SimLex-999 ρ", w: int | None = None) -> pd.DataFrame:
    """For each binary model and w, the k with the highest mean score over seeds."""
    table = scores(raw)
    table = table[table.model.isin(BINARY_MODELS)]
    if w is not None:
        table = table[table.w == w]
    means = table.groupby(["model", "w", "k"])[score].mean().reset_index()
    best = means.loc[means.groupby(["model", "w"])[score].idxmax()].reset_index(drop=True)
    return best.rename(columns={score: "mean"}).assign(score=score)


def plot_vs_k(raw: pd.DataFrame, score: str, w: int = 11, ax=None):
    """Score against hash length k for FlyVec and binary word2vec, dense models as lines."""
    import matplotlib.pyplot as plt

    table = scores(raw)
    table = table[table.w == w]
    ax = ax or plt.subplots(figsize=(5, 3.5))[1]
    for model in BINARY_MODELS:
        stats = table[table.model == model].groupby("k")[score].agg(["mean", "std"])
        ax.errorbar(stats.index, stats["mean"], yerr=stats["std"], marker="o", capsize=3, label=MODEL_LABELS[model])
    for model, style in (("word2vec", "--"), ("fasttext", ":")):
        values = table[table.model == model][score]
        if len(values):
            ax.axhline(values.mean(), linestyle=style, color="gray", label=f"{MODEL_LABELS[model]} (300-d)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(table[table.model == "flyvec"].k.unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("hash length k")
    ax.set_ylabel(score)
    ax.set_title(f"{score} vs k (w = {w})")
    ax.legend(fontsize=8)
    return ax


def resource_table(resources: pd.DataFrame, w: int | None = None) -> pd.DataFrame:
    """Mean training minutes and model size per trained model."""
    df = resources if w is None else resources[resources.w == w]
    return df.groupby("model")[["train_minutes", "size_mb", "codes_mb"]].mean()


def plot_vs_resources(
    raw: pd.DataFrame, resources: pd.DataFrame, score: str, k: int = 32, w: int = 11, axes=None
):
    """Score against training time and against model size, one point per model.

    Binary models are shown at hash length `k`, with the size of their 0/1
    word codes (`codes_mb`: one bit per cell or dimension), which is what a
    user keeps after training; the size of the trained files is in
    `resource_table`.
    """
    import matplotlib.pyplot as plt

    table = scores(raw)
    table = table[(table.w == w) & ((table.k == k) | (table.k < 0))]
    res = resource_table(resources, w)
    if axes is None:
        axes = plt.subplots(1, 2, figsize=(9, 3.5))[1]
    for model in MODEL_ORDER[:-1]:  # fastText + subword costs the same as fastText
        values = table[table.model == model][score]
        if not len(values) or RESOURCE_KIND[model] not in res.index:
            continue
        row = res.loc[RESOURCE_KIND[model]]
        size = row["codes_mb"] if model in BINARY_MODELS else row["size_mb"]
        label = MODEL_LABELS[model] + (f" k={k}" if model in BINARY_MODELS else "")
        for ax, x in zip(axes, (row["train_minutes"], size)):
            ax.errorbar(x, values.mean(), yerr=values.std(), marker="o", capsize=3, label=label)
    axes[0].set_xlabel("training time (minutes)")
    axes[1].set_xlabel("model size (MB, log scale)")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.set_ylabel(score)
    axes[1].legend(fontsize=8)
    return axes
