"""A download that stops halfway must not leave files that the next run treats as finished."""

import importlib.util
import sys
import types
import urllib.request
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"


@pytest.fixture(scope="module")
def download_data():
    spec = importlib.util.spec_from_file_location("download_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_datasets(monkeypatch, rows, fail_after=None):
    """Install a `datasets` module whose Wikipedia articles are `rows`, optionally failing midway."""

    class Articles:
        def shuffle(self, seed):
            return self

        def __iter__(self):
            for i, row in enumerate(rows):
                if fail_after is not None and i == fail_after:
                    raise ConnectionError("connection lost")
                yield row

    module = types.ModuleType("datasets")
    module.load_dataset = lambda *args, **kwargs: Articles()
    monkeypatch.setitem(sys.modules, "datasets", module)


ARTICLES = [{"id": str(i), "title": f"บทความ {i}", "text": "ข้อความ" * 10} for i in range(5)]


def test_interrupted_wiki_download_leaves_no_files(download_data, tmp_path, monkeypatch):
    fake_datasets(monkeypatch, ARTICLES, fail_after=2)
    with pytest.raises(ConnectionError):
        download_data.download_wiki(tmp_path, max_chars=10**9, seed=0, force=False)
    assert not list((tmp_path / "raw").iterdir())

    fake_datasets(monkeypatch, ARTICLES)  # the next run starts over instead of skipping
    download_data.download_wiki(tmp_path, max_chars=10**9, seed=0, force=False)
    raw = tmp_path / "raw"
    assert len((raw / "thwiki_subset.jsonl").read_text(encoding="utf-8").splitlines()) == 5
    assert len((raw / "thwiki_articles.txt").read_text(encoding="utf-8").splitlines()) == 5
    assert sorted(p.name for p in raw.iterdir()) == ["thwiki_articles.txt", "thwiki_subset.jsonl"]


def test_force_keeps_the_old_files_if_the_new_download_fails(download_data, tmp_path, monkeypatch):
    fake_datasets(monkeypatch, ARTICLES)
    download_data.download_wiki(tmp_path, max_chars=10**9, seed=0, force=False)
    before = (tmp_path / "raw" / "thwiki_subset.jsonl").read_bytes()

    fake_datasets(monkeypatch, ARTICLES, fail_after=1)
    with pytest.raises(ConnectionError):
        download_data.download_wiki(tmp_path, max_chars=10**9, seed=0, force=True)
    assert (tmp_path / "raw" / "thwiki_subset.jsonl").read_bytes() == before
    assert not list((tmp_path / "raw").glob("*.part"))


def test_interrupted_eval_download_leaves_no_file(download_data, tmp_path, monkeypatch):
    def broken(url, filename):
        Path(filename).write_text("half a fi")
        raise ConnectionError("connection lost")

    monkeypatch.setattr(urllib.request, "urlretrieve", broken)
    with pytest.raises(ConnectionError):
        download_data.download_eval(tmp_path, force=False)
    assert not list((tmp_path / "eval").iterdir())
