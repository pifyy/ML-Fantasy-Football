"""Tests for the raw data pull, using fake loaders instead of the network."""

import polars as pl

from ml_fantasy_football.data.pull import pull_table


def fake_loader(season=None):
    return pl.DataFrame({"season": [season], "value": [1]})


def test_writes_one_file_per_season(tmp_path):
    assert pull_table("stats", fake_loader, [2015, 2016], raw_dir=tmp_path) == []
    assert pl.read_parquet(tmp_path / "stats" / "2016.parquet")["season"].to_list() == [2016]
    assert (tmp_path / "stats" / "2015.parquet").exists()


def test_writes_single_file_without_seasons(tmp_path):
    assert pull_table("players", fake_loader, None, raw_dir=tmp_path) == []
    assert (tmp_path / "players.parquet").exists()


def test_skips_cached_files_unless_forced(tmp_path):
    calls = []

    def counting_loader(season):
        calls.append(season)
        return fake_loader(season)

    pull_table("stats", counting_loader, [2015], raw_dir=tmp_path)
    pull_table("stats", counting_loader, [2015], raw_dir=tmp_path)
    assert calls == [2015]

    pull_table("stats", counting_loader, [2015], raw_dir=tmp_path, force=True)
    assert calls == [2015, 2015]


def test_failure_is_reported_and_later_seasons_still_run(tmp_path):
    def flaky_loader(season):
        if season == 2015:
            raise ConnectionError("boom")
        return fake_loader(season)

    failures = pull_table("stats", flaky_loader, [2015, 2016], raw_dir=tmp_path)
    assert len(failures) == 1 and "2015" in failures[0]
    assert not (tmp_path / "stats" / "2015.parquet").exists()
    assert (tmp_path / "stats" / "2016.parquet").exists()


def test_empty_table_counts_as_failure(tmp_path):
    empty = lambda season: pl.DataFrame({"season": []})
    assert len(pull_table("stats", empty, [2015], raw_dir=tmp_path)) == 1
    assert not (tmp_path / "stats" / "2015.parquet").exists()
