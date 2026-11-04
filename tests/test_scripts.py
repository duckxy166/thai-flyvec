"""Run the pipeline scripts end to end on a tiny corpus (no downloads)."""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

TEXT = [
    "แมวกินปลา แมวนอนบนเตียง หมากินกระดูก หมาเห่าเสียงดัง",
    "รถยนต์วิ่งบนถนน รถไฟวิ่งบนราง ถนนในกรุงเทพมหานครมีรถมาก",
    "อาหารอร่อยมาก ร้านอาหารนี้อร่อย กาแฟร้อนหอม ชาเย็นหวาน",
]


def run(script: str, *args: str) -> None:
    spec = importlib.util.spec_from_file_location(script, SCRIPTS / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv, sys.argv = sys.argv, [script, *args]
    try:
        module.main()
    finally:
        sys.argv = argv


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    raw = root / "raw"
    (raw / "wisesight").mkdir(parents=True)
    with open(raw / "thwiki_subset.jsonl", "w", encoding="utf-8") as f:
        for i in range(60):
            article = {"id": str(i), "title": f"บทความ {i}", "text": "'''หัวข้อ''' " + TEXT[i % 3]}
            f.write(json.dumps(article, ensure_ascii=False) + "\n")
    (root / "eval").mkdir()
    pairs = "แมว,หมา,7\nรถยนต์,ถนน,6\nอาหาร,อร่อย,8\nแมว,ถนน,1\nกาแฟ,ชา,8\nปลา,รถไฟ,0.5\nแมว,ยีราฟ,3\n"
    for name in ("th-wordsim-353.csv", "th-simlex-999.csv"):
        (root / "eval" / name).write_text(pairs, encoding="utf-8")
    texts = [("อาหารอร่อยมาก", 0), ("ร้านนี้แย่", 2), ("กาแฟหอม", 0), ("รถติดแย่", 2), ("ไปไหนดี", 3)]
    for split in ("train", "validation", "test"):
        rows = [{"texts": t, "category": c} for t, c in texts * 4]
        (raw / "wisesight" / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
        )
    run("preprocess_corpus", "--input", str(raw / "thwiki_subset.jsonl"),
        "--out-dir", str(root / "processed"), "--max-tokens", "300", "--workers", "1")  # fmt: skip
    return root


def test_preprocess_stops_at_token_budget(data_dir):
    stats = json.loads((data_dir / "processed" / "stats.json").read_text())
    assert 300 <= stats["n_tokens"] < 400
    assert stats["n_articles"] == len((data_dir / "processed" / "articles_used.txt").read_text().splitlines())
    assert stats["n_articles"] < 60


def test_experiments_and_summary(data_dir, tmp_path):
    models, results = tmp_path / "models", tmp_path / "results"
    common = ["--data-dir", str(data_dir), "--models-dir", str(models)]
    run("run_experiments", *common, "--results-dir", str(results), "--w", "5", "--seeds", "0", "1",
        "--epochs", "1", "--workers", "1")  # fmt: skip

    raw = pd.read_csv(results / "raw_results.csv")
    assert set(raw.model) == {"flyvec", "word2vec_binary", "word2vec", "fasttext", "fasttext_subword"}
    assert set(raw.task) == {"th-wordsim-353", "th-simlex-999", "wisesight"}
    assert set(raw.k.dropna()) == {4, 8, 16, 32, 64}
    oov = raw[(raw.metric == "oov_ratio") & (raw.model == "word2vec")].value
    assert (oov > 0).all()  # "ยีราฟ" is not in the corpus
    resources = pd.read_csv(results / "resources.csv")
    assert len(resources) == 3 * 2
    assert (resources.size_mb > 0).all()
    assert (results / "machine.json").exists()

    run("summarize_results", "--results-dir", str(results), "--w", "5", "--k", "16")
    assert "FlyVec" in (results / "summary.md").read_text(encoding="utf-8")
    assert len(list((results / "figures").glob("*.png"))) == 8

    # the reuse path loads the saved models instead of training again
    run("run_experiments", *common, "--results-dir", str(tmp_path / "again"), "--w", "5", "--seeds", "0",
        "--epochs", "1", "--reuse", "--skip-sentiment")  # fmt: skip
    again = pd.read_csv(tmp_path / "again" / "raw_results.csv")
    assert "wisesight" not in set(again.task)


def test_train_scripts(data_dir, tmp_path, capsys):
    models = tmp_path / "models"
    common = ["--corpus", str(data_dir / "processed"), "--models-dir", str(models), "--epochs", "1"]
    run("train_flyvec", *common, "--cells", "40", "--k", "4", "--words", "แมว", "ยีราฟ")
    run("train_baselines", *common, "--model", "word2vec", "--workers", "1", "--words", "แมว")
    out = capsys.readouterr().out
    assert "ยีราฟ: (OOV)" in out
    assert (models / "flyvec_w11_s0.npz").exists()
    assert (models / "flyvec_w11_s0.json").exists()
    assert (models / "word2vec_w11_s0.kv").exists()
