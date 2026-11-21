"""Train and evaluate GloVe (300d, w = 11) on the Thai Wikipedia corpus.

    python scripts/run_glove.py --models-dir models/lr0.3 --w 11 --seeds 0 1 2

Trains (or reuses with `--reuse`) GloVe on the same corpus and 20,000-word
vocabulary as FlyVec/word2vec/fastText (`data/processed/`), evaluates on
TH-WordSim-353, TH-SimLex-999, wisesight_sentiment, and the #58 analyses
(H1 rare words, H2 OOV-heavy text, H3 prachathai-67k topics), and saves outputs
under `results/analysis/` without touching `results/raw_results.csv`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
import run_analysis  # noqa: E402

from thai_flyvec import baselines, pipeline, use_utf8_output  # noqa: E402
from thai_flyvec.evaluate import (  # noqa: E402
    known_pairs,
    load_prachathai,
    load_similarity_pairs,
    load_wisesight,
    machine_spec,
    sentiment,
    word_similarity,
)
from thai_flyvec.preprocess import Corpus  # noqa: E402
from thai_flyvec.report import to_markdown  # noqa: E402


def _merge_analysis_csv(path: Path, new_rows: list[dict], model_name: str = "glove") -> None:
    new_df = pd.DataFrame(new_rows)
    if path.exists():
        old_df = pd.read_csv(path)
        old_df = old_df[old_df["model"] != model_name]
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df
    merged.to_csv(path, index=False)


def _fmt_mean_sd(series: pd.Series) -> str:
    if len(series) > 1:
        return f"{series.mean():.3f} ± {series.std():.3f}"
    return f"{series.mean():.3f}"


def write_glove_summary(out_dir: Path, main_df: pd.DataFrame, res_df: pd.DataFrame, args) -> None:
    rows = []
    for w in sorted(main_df["w"].unique()):
        sub_m = main_df[main_df["w"] == w]
        sub_r = res_df[res_df["w"] == w]
        ws = sub_m[(sub_m["task"] == "th-wordsim-353") & (sub_m["metric"] == "rho")]["value"]
        sl = sub_m[(sub_m["task"] == "th-simlex-999") & (sub_m["metric"] == "rho")]["value"]
        acc = sub_m[(sub_m["task"] == "wisesight") & (sub_m["metric"] == "accuracy")]["value"]
        f1 = sub_m[(sub_m["task"] == "wisesight") & (sub_m["metric"] == "macro_f1")]["value"]
        rows.append(
            {
                "model": "GloVe",
                "w": w,
                "TH-WordSim-353 ρ": _fmt_mean_sd(ws),
                "TH-SimLex-999 ρ": _fmt_mean_sd(sl),
                "wisesight accuracy": _fmt_mean_sd(acc),
                "wisesight macro-F1": _fmt_mean_sd(f1),
                "train_minutes": _fmt_mean_sd(sub_r["train_minutes"]),
                "size_mb": f"{sub_r['size_mb'].mean():.2f}",
            }
        )
    summary_md = (
        f"# GloVe baseline (dim={args.dim}, epochs={args.epochs}, lr={args.lr}, "
        f"x_max={args.x_max}, alpha={args.alpha})\n\n"
        f"Seeds: {', '.join(str(s) for s in args.seeds)}. "
        "Evaluated on the same common pairs (WordSim 212, SimLex 743) and splits.\n\n"
        + to_markdown(pd.DataFrame(rows))
        + "\n"
    )
    (out_dir / "glove_summary.md").write_text(summary_md, encoding="utf-8")
    print(f"wrote {out_dir / 'glove_summary.md'}")


def main() -> None:
    use_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models" / "lr0.3")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--w", type=int, default=11)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dim", type=int, default=baselines.DIM)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--x-max", type=float, default=100.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--skip-topic", action="store_true")
    args = parser.parse_args()

    out_dir = args.results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.results_dir / "figures"
    (out_dir / "glove_machine.json").write_text(json.dumps(machine_spec(), indent=2), encoding="utf-8")

    corpus = Corpus.load(args.data_dir / "processed")
    counts = dict(zip(corpus.words, (int(c) for c in corpus.counts)))
    vocab = set(corpus.words)
    similarity = {
        task: load_similarity_pairs(args.data_dir / "eval" / name)
        for task, name in run_analysis.SIMILARITY_SETS.items()
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

    main_rows, res_rows, freq_rows, oov_rows_all, topic_rows_all = [], [], [], [], []
    for seed in args.seeds:
        print(f"GloVe w={args.w} seed={seed}", flush=True)
        path = pipeline.model_path(args.models_dir, "glove", args.w, seed)
        if args.reuse and path.exists() and path.with_suffix(".json").exists():
            print(f"  reuse {path.name}", flush=True)
            meta = pipeline.load_meta(path)
        else:
            print(f"  train {path.name}", flush=True)
            meta = pipeline.train_glove(
                corpus,
                args.models_dir,
                w=args.w,
                seed=seed,
                epochs=args.epochs,
                dim=args.dim,
                lr=args.lr,
                x_max=args.x_max,
                alpha=args.alpha,
                workers=args.workers,
            )
            print(f"  trained in {meta['train_minutes']:.2f} min, {meta['size_mb']:.2f} MB", flush=True)

        res_rows.append(
            {
                "model": "glove",
                "w": args.w,
                "seed": seed,
                "train_minutes": meta["train_minutes"],
                "size_mb": meta["size_mb"],
                "codes_mb": None,
            }
        )

        glove = pipeline.load_glove(args.models_dir, args.w, seed)
        embed = pipeline.glove_embedding(glove, corpus.words)
        base_main = {"model": "glove", "k": None, "w": args.w, "seed": seed}
        for task, pairs in similarity.items():
            common = known_pairs(pairs, embed)
            res = word_similarity(embed, common)
            res["oov_ratio"] = 1 - res["n_pairs"] / len(pairs)
            main_rows += [
                base_main | {"task": task, "pairs": "common", "metric": m, "value": res[m]}
                for m in ("rho", "n_pairs", "oov_ratio")
            ]
        sent_res = sentiment(embed, wisesight, seed=seed)
        main_rows += [
            base_main | {"task": "wisesight", "pairs": "", "metric": m, "value": v}
            for m, v in sent_res.items()
        ]
        print(
            f"  main: simlex={next(r['value'] for r in main_rows[-8:] if r['task']=='th-simlex-999' and r['metric']=='rho'):.3f} "
            f"wordsim={next(r['value'] for r in main_rows[-11:] if r['task']=='th-wordsim-353' and r['metric']=='rho'):.3f} "
            f"wisesight_f1={sent_res['macro_f1']:.3f}",
            flush=True,
        )

        base_ana = {"w": args.w, "seed": seed}
        embeds = [("glove", None, embed)]
        freq_rows += run_analysis.frequency_rows(embeds, similarity, counts, base_ana)
        oov_rows_all += run_analysis.oov_rows(embeds, wisesight, vocab, seed, base_ana)
        if prachathai is not None:
            topic_rows_all += run_analysis.topic_rows(embeds, prachathai, seed, base_ana, args.workers)

        pd.DataFrame(main_rows).to_csv(out_dir / "glove_main.csv", index=False)
        pd.DataFrame(res_rows).to_csv(out_dir / "glove_resources.csv", index=False)
        _merge_analysis_csv(out_dir / "frequency.csv", freq_rows)
        _merge_analysis_csv(out_dir / "oov.csv", oov_rows_all)
        if prachathai is not None:
            _merge_analysis_csv(out_dir / "topic.csv", topic_rows_all)

    write_glove_summary(out_dir, pd.DataFrame(main_rows), pd.DataFrame(res_rows), args)
    run_analysis.summarize(out_dir, fig_dir)


if __name__ == "__main__":
    main()
