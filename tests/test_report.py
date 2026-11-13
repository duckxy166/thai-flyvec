import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from thai_flyvec import report  # noqa: E402


@pytest.fixture
def raw():
    """Fake results in the format of scripts/run_experiments.py."""
    rows = []
    rng = np.random.default_rng(0)
    for w in (5, 11):
        for seed in (0, 1, 2):
            models = [("flyvec", k) for k in (4, 8, 16)] + [("word2vec_binary", k) for k in (4, 8, 16)]
            models += [("word2vec", None), ("fasttext", None), ("fasttext_subword", None)]
            for model, k in models:
                base = {"model": model, "k": k, "w": w, "seed": seed}
                # FlyVec is best at k = 8 in this fake data
                quality = {"flyvec": 0.3 - abs((k or 8) - 8) / 100, "word2vec_binary": 0.2}.get(model, 0.4)
                for task in ("th-wordsim-353", "th-simlex-999"):
                    pair_sets = ["common", "all"] if model == "fasttext_subword" else ["common"]
                    for pairs in pair_sets:
                        # subword vectors for OOV words score lower on all pairs
                        rho = quality + rng.normal(0, 0.01) - (0.1 if pairs == "all" else 0)
                        oov = 0.05 if pairs == "all" else 0.3
                        rows += [
                            base | {"task": task, "pairs": pairs, "metric": "rho", "value": rho},
                            base | {"task": task, "pairs": pairs, "metric": "oov_ratio", "value": oov},
                        ]
                for metric in ("accuracy", "macro_f1"):
                    rows.append(base | {"task": "wisesight", "pairs": "", "metric": metric, "value": quality + 0.3})
    return pd.DataFrame(rows)


@pytest.fixture
def resources():
    rows = []
    for w in (5, 11):
        for seed in (0, 1, 2):
            rows += [
                {"model": "flyvec", "w": w, "seed": seed, "train_minutes": 30, "size_mb": 61, "codes_mb": 0.95},
                {"model": "word2vec", "w": w, "seed": seed, "train_minutes": 10, "size_mb": 23, "codes_mb": 0.7},
                {"model": "fasttext", "w": w, "seed": seed, "train_minutes": 20, "size_mb": 250, "codes_mb": None},
            ]
    return pd.DataFrame(rows)


def test_roundtrip_through_csv(raw, resources, tmp_path):
    raw.to_csv(tmp_path / "raw_results.csv", index=False)
    resources.to_csv(tmp_path / "resources.csv", index=False)
    loaded, _ = report.load(tmp_path)
    assert len(report.summary_table(loaded, w=11)) == len(report.summary_table(raw, w=11))


def test_scores_one_column_per_score(raw):
    table = report.scores(raw)
    assert list(table.columns[4:]) == list(report.SCORES.values())
    # 6 binary rows + 3 dense rows, per (w, seed)
    assert len(table) == 9 * 2 * 3


def test_summary_table(raw):
    table = report.summary_table(raw, w=11)
    assert list(table.model.unique()) == [report.MODEL_LABELS[m] for m in report.MODEL_ORDER]
    row = table[(table.model == "FlyVec") & (table.k == "8")].iloc[0]
    assert row["wisesight accuracy"].startswith("0.600 ± ")
    assert "w" in report.summary_table(raw).columns


def test_summary_table_subword_row_uses_all_pairs(raw):
    table = report.summary_table(raw, w=11).set_index("model")
    fasttext = float(table.loc["fastText", "TH-SimLex-999 ρ"].split()[0])
    subword = float(table.loc[report.MODEL_LABELS["fasttext_subword"], "TH-SimLex-999 ρ"].split()[0])
    assert fasttext == pytest.approx(0.4, abs=0.02)
    assert subword == pytest.approx(0.3, abs=0.02)
    # sentiment has one pair set, so the row keeps its own score
    assert table.loc[report.MODEL_LABELS["fasttext_subword"], "wisesight accuracy"].startswith("0.700")


def test_best_k(raw):
    best = report.best_k(raw, "TH-SimLex-999 ρ", w=11)
    assert best[best.model == "flyvec"].k.item() == 8


def test_oov_table(raw):
    table = report.oov_table(raw)
    assert table.loc["th-simlex-999", "word2vec"] == pytest.approx(0.3)
    assert table.loc["th-simlex-999", "fasttext_subword"] == pytest.approx(0.05)


def test_plots(raw, resources):
    ax = report.plot_vs_k(raw, "TH-WordSim-353 ρ", w=11)
    assert len(ax.lines) >= 2
    axes = report.plot_vs_resources(raw, resources, "wisesight macro-F1", k=8, w=11)
    assert len(axes) == 2


def test_to_markdown():
    md = report.to_markdown(pd.DataFrame({"a": [1, None], "b": ["x", "y"]}))
    assert md.splitlines() == ["| a | b |", "|---|---|", "| 1.0 | x |", "|  | y |"]
