"""What FlyVec is good at compared with word2vec and fastText (issue #58, docs/decisions.md D11).

    python scripts/run_analysis.py                      # w = 11, seeds 0 1 2
    python scripts/run_analysis.py --skip-topic         # H1 and H2 only (minutes)

Reuses the models of `scripts/run_experiments.py` (no training) and runs

- H1 rare words: Spearman rho on TH-SimLex-999 / TH-WordSim-353 separately for
  three equal-size groups of pairs, by the corpus count of the rarer word
- H2 OOV-heavy text: wisesight test macro-F1 separately for texts with 0%,
  0-10%, 10-20% and >20% of tokens outside the vocabulary
- H3 topic classification: prachathai-67k multi-label news topics (12 classes,
  title + body), macro-F1 and micro-F1

for FlyVec and binarized word2vec at k = 8 16 32, word2vec and fastText, plus
fastText with subword vectors for OOV words in H1 and H2. Writes
`results/analysis/{frequency,oov,topic}.csv` (one row per model, k, seed and
group) after every seed, then `results/analysis/summary.md` and figures in
`results/figures/analysis_*.png`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thai_flyvec import pipeline, use_utf8_output  # noqa: E402
from thai_flyvec.evaluate import (  # noqa: E402
    known_pairs,
    load_prachathai,
    load_similarity_pairs,
    load_wisesight,
    macro_f1_by_group,
    oov_ratios,
    pair_frequency,
    quantile_bins,
    sentiment,
    similarity_by_frequency,
    topic_classification,
    word_similarity,
)
from thai_flyvec.preprocess import Corpus  # noqa: E402
from thai_flyvec.report import MODEL_LABELS, to_markdown  # noqa: E402

SIMILARITY_SETS = {"th-wordsim-353": "th-wordsim-353.csv", "th-simlex-999": "th-simlex-999.csv"}
K_VALUES = (8, 16, 32)
OOV_GROUPS = ((0.0, 0.0, "0%"), (0.0, 0.1, "0-10%"), (0.1, 0.2, "10-20%"), (0.2, 1.0, ">20%"))
FREQ_LABELS = ("rare", "mid", "frequent")


def embeddings(corpus, models_dir, w, seed, k_values):
    """(model, k, Embedding) for every row; the fastText subword row comes last."""
    words = corpus.words
    flyvec = pipeline.load_flyvec(models_dir, w, seed)
    w2v = pipeline.load_keyed_vectors(models_dir, "word2vec", w, seed)
    ft = pipeline.load_keyed_vectors(models_dir, "fasttext", w, seed)
    for k in k_values:
        yield "flyvec", k, pipeline.flyvec_embedding(flyvec, words, k)
    for k in k_values:
        yield "word2vec_binary", k, pipeline.binary_embedding(w2v, words, k)
    yield "word2vec", None, pipeline.keyed_embedding(w2v, words, "word2vec")
    yield "fasttext", None, pipeline.keyed_embedding(ft, words, "fasttext")
    yield "fasttext_subword", None, pipeline.keyed_embedding(ft, words, "fasttext + subword", subword_oov=True)


def oov_group(ratio: float) -> str | None:
    if np.isnan(ratio):
        return None  # text with no tokens
    for low, high, label in OOV_GROUPS:
        if (ratio == 0 and high == 0) or low < ratio <= high:
            return label
    return None


def frequency_rows(embeds, similarity, counts, base) -> list[dict]:
    """H1: rho per frequency group on the pairs every in-vocabulary model knows."""
    rows = []
    in_vocab = [e for model, _, e in embeds if model != "fasttext_subword"]
    for task, pairs in similarity.items():
        common = known_pairs(pairs, *in_vocab)
        bins, edges = quantile_bins(pair_frequency(common, counts), len(FREQ_LABELS))
        unseen = [p for p in pairs if min(counts.get(p[0], 0), counts.get(p[1], 0)) == 0]
        for model, k, embed in embeds:
            for result in similarity_by_frequency(embed, common, bins):
                b = result["bin"]
                rows.append(
                    base | {"model": model, "k": k, "task": task, "group": FREQ_LABELS[b],
                            "min_count": int(edges[b]), "max_count": int(edges[b + 1]),
                            "n_pairs": result["n_pairs"], "rho": result["rho"]}
                )  # fmt: skip
            if model == "fasttext_subword":  # pairs with a word outside the vocabulary
                result = word_similarity(embed, unseen)
                rows.append(
                    base | {"model": model, "k": k, "task": task, "group": "oov", "min_count": 0,
                            "max_count": 0, "n_pairs": result["n_pairs"], "rho": result["rho"]}
                )  # fmt: skip
    return rows


def oov_rows(embeds, wisesight, vocab, seed, base) -> list[dict]:
    """H2: wisesight test macro-F1 per OOV group, with the same classifier as run_experiments."""
    tokens, labels = wisesight["test"]
    groups = [oov_group(r) for r in oov_ratios(tokens, vocab)]
    keep = np.array([g is not None for g in groups])
    rows = []
    for model, k, embed in embeds:
        result = sentiment(embed, wisesight, seed=seed, return_predictions=True)
        y_true = np.asarray(labels)[keep]
        y_pred = result["test_pred"][keep]
        kept_groups = np.array([g for g in groups if g is not None])
        for row in macro_f1_by_group(y_true, y_pred, kept_groups):
            rows.append(base | {"model": model, "k": k} | row)
        rows.append(base | {"model": model, "k": k, "group": "all", "n_texts": len(labels),
                            "macro_f1": result["macro_f1"]})  # fmt: skip
        print(f"    oov {embed.name}: done", flush=True)
    return rows


def topic_rows(embeds, prachathai, seed, base, n_jobs) -> list[dict]:
    """H3: prachathai-67k topics; the subword row is left out (articles are long and formal)."""
    rows = []
    for model, k, embed in embeds:
        if model == "fasttext_subword":
            continue
        result = topic_classification(embed, prachathai, seed=seed, n_jobs=n_jobs)
        rows.append(base | {"model": model, "k": k} | result)
        print(f"    topic {embed.name}: macro-F1 {result['macro_f1']:.3f}", flush=True)
    return rows


# --- summary ---------------------------------------------------------------------


def _label(model, k) -> str:
    name = MODEL_LABELS.get(model, model).replace(" (all pairs)", "")
    return f"{name} k = {int(k)}" if pd.notna(k) else name


def _mean_std(df, keys, value) -> pd.DataFrame:
    g = df.groupby(keys, dropna=False)[value].agg(["mean", "std"]).reset_index()
    g["text"] = [f"{m:.3f} ± {s:.3f}" if pd.notna(s) else f"{m:.3f}" for m, s in zip(g["mean"], g["std"])]
    return g


def _pivot(g, columns, order) -> pd.DataFrame:
    g = g.assign(row=[_label(m, k) for m, k in zip(g["model"], g["k"])])
    table = g.pivot(index="row", columns=columns, values="text")
    rows = [_label(m, k) for m, k in order if _label(m, k) in table.index]
    table = table.loc[rows]
    return table.reset_index().rename(columns={"row": "model"})


def _order(df):
    keys = df[["model", "k"]].drop_duplicates()
    rank = {m: i for i, m in enumerate(MODEL_LABELS)}
    keys = keys.assign(r=keys["model"].map(rank), kk=keys["k"].fillna(0)).sort_values(["r", "kk"])
    return list(zip(keys["model"], keys["k"]))


def summarize(out_dir: Path, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parts = ["# What FlyVec is good at (issue #58)", "",
             "Mean ± std over seeds, w = 11. Written by `scripts/run_analysis.py`.", ""]  # fmt: skip
    fig_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "frequency.csv"
    if path.exists():
        df = pd.read_csv(path)
        for task in SIMILARITY_SETS:
            d = df[df["task"] == task]
            g = _mean_std(d, ["model", "k", "group"], "rho")
            table = _pivot(g, "group", _order(d))
            table = table[["model"] + [c for c in (*FREQ_LABELS, "oov") if c in table.columns]]
            sizes = d[d["seed"] == d["seed"].min()].drop_duplicates("group").set_index("group")
            note = ", ".join(
                f"{grp} {int(sizes.loc[grp, 'n_pairs'])} pairs (count {int(sizes.loc[grp, 'min_count'])}"
                f"-{int(sizes.loc[grp, 'max_count'])})"
                for grp in FREQ_LABELS if grp in sizes.index
            )  # fmt: skip
            parts += [f"## H1 rare words: {task} ρ by frequency of the rarer word", "", note, "",
                      to_markdown(table), ""]  # fmt: skip
        _plot_groups(plt, df[df["group"].isin(FREQ_LABELS)], "rho", list(FREQ_LABELS), "task",
                     "Spearman ρ", fig_dir / "analysis_frequency.png")  # fmt: skip

    path = out_dir / "oov.csv"
    if path.exists():
        df = pd.read_csv(path)
        groups = [label for _, _, label in OOV_GROUPS] + ["all"]
        g = _mean_std(df, ["model", "k", "group"], "macro_f1")
        table = _pivot(g, "group", _order(df))[["model", *groups]]
        sizes = df[(df["seed"] == df["seed"].min())].drop_duplicates("group").set_index("group")["n_texts"]
        note = ", ".join(f"{grp} {int(sizes[grp])} texts" for grp in groups if grp in sizes.index)
        parts += ["## H2 OOV-heavy text: wisesight test macro-F1 by share of OOV tokens", "", note, "",
                  to_markdown(table), ""]  # fmt: skip
        _plot_groups(plt, df.assign(task="wisesight")[df["group"] != "all"], "macro_f1", groups[:-1], "task",
                     "macro-F1", fig_dir / "analysis_oov.png")  # fmt: skip

    path = out_dir / "topic.csv"
    if path.exists():
        df = pd.read_csv(path)
        order = _order(df)
        cols = []
        for metric in ("macro_f1", "micro_f1"):
            g = _mean_std(df, ["model", "k"], metric).assign(metric=metric)
            cols.append(g)
        g = pd.concat(cols)
        table = _pivot(g, "metric", order)
        parts += ["## H3 topic classification: prachathai-67k test (12 topics, multi-label)", "",
                  to_markdown(table), ""]  # fmt: skip
        _plot_topic(plt, df, order, fig_dir / "analysis_topic.png")

    (out_dir / "summary.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out_dir / 'summary.md'}")


def _plot_groups(plt, df, value, groups, panel_col, ylabel, path):
    panels = sorted(df[panel_col].unique())
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4), squeeze=False)
    x = np.arange(len(groups))
    styles = {"flyvec": "-", "word2vec_binary": "--", "word2vec": "-", "fasttext": "-", "fasttext_subword": ":"}
    for ax, panel in zip(axes[0], panels):
        d = df[df[panel_col] == panel]
        for model, k in _order(d):
            if model in ("flyvec", "word2vec_binary") and k != 16:
                continue  # one k per binary model keeps the figure readable
            if model == "fasttext_subword" and value == "rho":
                continue  # same vectors as fastText on in-vocabulary pairs
            s = d[(d["model"] == model) & ((d["k"] == k) if pd.notna(k) else d["k"].isna())]
            g = s.groupby("group")[value].agg(["mean", "std"]).reindex(groups)
            ax.errorbar(x, g["mean"], yerr=g["std"], marker="o", capsize=3,
                        linestyle=styles.get(model, "-"), label=_label(model, k))  # fmt: skip
        ax.set_xticks(x, groups)
        ax.set_title(panel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_topic(plt, df, order, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    labels, means, stds = [], [], []
    for model, k in order:
        s = df[(df["model"] == model) & ((df["k"] == k) if pd.notna(k) else df["k"].isna())]["macro_f1"]
        labels.append(_label(model, k))
        means.append(s.mean())
        stds.append(s.std())
    colors = ["#d95f02" if lab.startswith("FlyVec") else "#7570b3" if "binary" in lab else "#1b9e77" for lab in labels]
    y = np.arange(len(labels))[::-1]  # dots, not bars: the differences are ~0.01 on a 0.45-0.49 range
    for yi, m, s, c in zip(y, means, stds, colors):
        ax.errorbar(m, yi, xerr=s, fmt="o", color=c, capsize=3, markersize=7)
    ax.set_yticks(y, labels)
    ax.set_xlabel("prachathai-67k macro-F1 (test, mean ± std over seeds)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    use_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models" / "lr0.3")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--w", type=int, default=11)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--k", type=int, nargs="+", default=list(K_VALUES))
    parser.add_argument("--workers", type=int, default=8, help="processes for tokenizing and one-vs-rest")
    parser.add_argument("--skip-topic", action="store_true")
    parser.add_argument("--summary-only", action="store_true", help="only rebuild summary.md and figures")
    args = parser.parse_args()

    out_dir = args.results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.results_dir / "figures"
    if args.summary_only:
        summarize(out_dir, fig_dir)
        return

    corpus = Corpus.load(args.data_dir / "processed")
    counts = dict(zip(corpus.words, (int(c) for c in corpus.counts)))
    vocab = set(corpus.words)
    similarity = {
        task: load_similarity_pairs(args.data_dir / "eval" / name) for task, name in SIMILARITY_SETS.items()
    }
    wisesight = load_wisesight(
        args.data_dir / "raw" / "wisesight", cache=args.data_dir / "processed" / "wisesight_tokens.json"
    )
    prachathai = None
    if not args.skip_topic:
        prachathai = load_prachathai(
            args.data_dir / "raw" / "prachathai",
            cache=args.data_dir / "processed" / "prachathai_tokens.json",
            workers=args.workers,
        )

    frequency, oov, topic = [], [], []
    for seed in args.seeds:
        print(f"w={args.w} seed={seed}", flush=True)
        base = {"w": args.w, "seed": seed}
        embeds = list(embeddings(corpus, args.models_dir, args.w, seed, args.k))
        frequency += frequency_rows(embeds, similarity, counts, base)
        pd.DataFrame(frequency).to_csv(out_dir / "frequency.csv", index=False)
        oov += oov_rows(embeds, wisesight, vocab, seed, base)
        pd.DataFrame(oov).to_csv(out_dir / "oov.csv", index=False)
        if prachathai is not None:
            topic += topic_rows(embeds, prachathai, seed, base, args.workers)
            pd.DataFrame(topic).to_csv(out_dir / "topic.csv", index=False)
    summarize(out_dir, fig_dir)


if __name__ == "__main__":
    main()
