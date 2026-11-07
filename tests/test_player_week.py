"""Tests for the player-week table, using small hand-built tables."""

import polars as pl
import pytest

from ml_fantasy_football.data.player_week import (
    STAT_COLUMNS,
    build_player_week,
    team_week_games,
    validate_player_week,
)

TEAM_CODES = {"OAK": "LV", "BLT": "BAL"}
POSITIONS = ["QB", "RB", "WR", "TE"]
POSITION_MAP = {"FB": "RB"}


def stat_row(player_id, week, team="LV", position="WR", points=10.0, season=2015):
    row = dict.fromkeys(STAT_COLUMNS, 1.0)
    return row | {
        "player_id": player_id,
        "player_display_name": player_id,
        "season": season,
        "week": week,
        "team": team,
        "position": position,
        "fantasy_points_ppr": points,
    }


def game(week, home, away, season=2015, spread=3.0, total=40.0):
    return {
        "season": season,
        "week": week,
        "game_id": f"{season}_{week}_{home}",
        "gameday": "2015-09-13",
        "location": "Home",
        "total_line": total,
        "spread_line": spread,
        "home_team": home,
        "away_team": away,
    }


def make_tables(stats, snaps=None, rosters=None, injuries=None, schedules=None):
    """Assemble raw tables. Weeks 1-4 by default, with OAK on bye in week 2."""
    schedules = schedules or [
        game(1, "OAK", "DEN"),
        game(3, "OAK", "DEN"),
        game(4, "DEN", "OAK"),
    ]
    snap_schema = {
        "pfr_player_id": pl.String, "season": pl.Int32, "week": pl.Int32, "team": pl.String,
        "position": pl.String, "offense_snaps": pl.Float64, "offense_pct": pl.Float64,
    }
    roster_schema = {
        "gsis_id": pl.String, "pfr_id": pl.String, "season": pl.Int32, "week": pl.Int32,
        "team": pl.String, "position": pl.String, "status": pl.String,
    }
    injury_schema = {
        "gsis_id": pl.String, "season": pl.Int32, "week": pl.Int32, "team": pl.String,
        "report_status": pl.String, "practice_status": pl.String,
        "report_primary_injury": pl.String, "date_modified": pl.Datetime("us", "UTC"),
    }
    return {
        "stats": pl.DataFrame(stats).with_columns(pl.col("season", "week").cast(pl.Int32)),
        "snaps": pl.DataFrame(snaps or [], schema=snap_schema),
        "rosters": pl.DataFrame(rosters or [], schema=roster_schema),
        "injuries": pl.DataFrame(injuries or [], schema=injury_schema),
        "schedules": pl.DataFrame(schedules).with_columns(pl.col("season", "week").cast(pl.Int32)),
        "opportunity": pl.DataFrame(
            {"player_id": ["A"], "season": [2015], "week": [1], "total_fantasy_points_exp": [8.5]}
        ).with_columns(pl.col("season", "week").cast(pl.Int32)),
        "players": pl.DataFrame(
            {
                "gsis_id": ["A", "B"],
                "pfr_id": ["pfrA", "pfrB"],
                "display_name": ["Player A", "Player B"],
                "birth_date": ["1990-01-01", "1991-01-01"],
                "rookie_season": [2012, 2014],
                "draft_round": [1, 7],
                "draft_pick": [10, 250],
            }
        ),
    }


def build(tables):
    return build_player_week(tables, TEAM_CODES, POSITIONS, POSITION_MAP)


def week(table, number, player_id="A"):
    return table.filter((pl.col("player_id") == player_id) & (pl.col("week") == number)).to_dicts()[0]


def test_bye_is_null_but_missed_game_is_zero():
    table = build(make_tables([stat_row("A", 1, points=12.0), stat_row("A", 3)]))
    assert table.filter(pl.col("player_id") == "A").height == 4
    assert week(table, 1)["ppr_points"] == 12.0 and week(table, 1)["played"]

    bye = week(table, 2)
    assert bye["ppr_points"] is None and not bye["team_has_game"] and not bye["played"]

    missed = week(table, 4)
    assert missed["ppr_points"] == 0.0 and missed["team_has_game"] and not missed["played"]
    assert missed["targets"] is None and missed["offense_snaps"] == 0.0


def test_old_team_codes_match_the_schedule():
    snaps = [
        {"pfr_player_id": "pfrB", "season": 2015, "week": 1, "team": "OAK",
         "position": "WR", "offense_snaps": 30.0, "offense_pct": 0.5},
    ]
    table = build(make_tables([stat_row("A", 1, team="LV")], snaps=snaps))
    assert week(table, 1, "B")["team"] == "LV"
    assert week(table, 1, "B")["team_has_game"]


def test_snaps_without_a_stat_row_count_as_played_with_real_zeros():
    snaps = [
        {"pfr_player_id": "pfrB", "season": 2015, "week": 1, "team": "LV",
         "position": "TE", "offense_snaps": 30.0, "offense_pct": 0.5},
    ]
    row = week(build(make_tables([stat_row("A", 1)], snaps=snaps)), 1, "B")
    assert row["played"] and row["ppr_points"] == 0.0 and row["targets"] == 0
    assert row["target_share"] is None and row["position"] == "TE"


def test_stat_row_without_snaps_keeps_snaps_unknown():
    row = week(build(make_tables([stat_row("A", 1)])), 1)
    assert row["played"] and row["offense_snaps"] is None


def test_player_who_never_played_is_excluded():
    rosters = [
        {"gsis_id": "B", "pfr_id": "pfrB", "season": 2015, "week": 1, "team": "LV",
         "position": "WR", "status": "CUT"},
    ]
    table = build(make_tables([stat_row("A", 1)], rosters=rosters))
    assert table["player_id"].unique().to_list() == ["A"]


def test_non_skill_positions_are_excluded_and_fullbacks_become_rbs():
    stats = [stat_row("A", 1, position="LB"), stat_row("B", 1, position="FB")]
    table = build(make_tables(stats))
    assert table["player_id"].unique().to_list() == ["B"]
    assert table["position"].unique().to_list() == ["RB"]


def test_duplicate_roster_rows_prefer_active():
    rosters = [
        {"gsis_id": "A", "pfr_id": "pfrA", "season": 2015, "week": 1, "team": "BLT",
         "position": "WR", "status": "TRC"},
        {"gsis_id": "A", "pfr_id": "pfrA", "season": 2015, "week": 1, "team": "BLT",
         "position": "WR", "status": "ACT"},
    ]
    row = week(build(make_tables([stat_row("A", 1)], rosters=rosters)), 1)
    assert row["roster_status"] == "ACT" and row["roster_team"] == "BAL"


def test_latest_injury_report_wins():
    from datetime import datetime, timezone

    injuries = [
        {"gsis_id": "A", "season": 2015, "week": 1, "team": "LV", "report_status": "Questionable",
         "practice_status": None, "report_primary_injury": "Knee",
         "date_modified": datetime(2015, 9, 11, tzinfo=timezone.utc)},
        {"gsis_id": "A", "season": 2015, "week": 1, "team": "LV", "report_status": "Out",
         "practice_status": None, "report_primary_injury": "Knee",
         "date_modified": datetime(2015, 9, 13, tzinfo=timezone.utc)},
    ]
    row = week(build(make_tables([stat_row("A", 1)], injuries=injuries)), 1)
    assert row["report_status"] == "Out"


def test_implied_totals_sum_to_game_total_and_favor_the_favorite():
    games = team_week_games(
        pl.DataFrame([game(1, "OAK", "DEN", spread=3.0, total=40.0)]), TEAM_CODES
    )
    home = games.filter(pl.col("team") == "LV").to_dicts()[0]
    away = games.filter(pl.col("team") == "DEN").to_dicts()[0]
    assert home["implied_total"] == 21.5 and home["is_home"] and home["team_spread"] == 3.0
    assert away["implied_total"] == 18.5 and not away["is_home"] and away["team_spread"] == -3.0


def test_season_length_comes_from_the_schedule():
    schedules = [game(1, "OAK", "DEN", season=2021), game(18, "OAK", "DEN", season=2021)]
    table = build(make_tables([stat_row("A", 1, season=2021)], schedules=schedules))
    assert table["week"].max() == 18 and table.height == 18


def test_validation_passes_on_a_good_table_and_catches_problems():
    tables = make_tables([stat_row("A", 1), stat_row("A", 3)])
    table = build(tables)
    check = lambda t: validate_player_week(t, tables["stats"], tables["schedules"], POSITIONS, POSITION_MAP)
    assert check(table) == []

    assert any("duplicate" in p for p in check(pl.concat([table, table.head(1)])))
    assert any("missing" in p for p in check(table.filter(pl.col("week") != 1)))
    changed = table.with_columns(pl.col("ppr_points") + 1)
    assert any("changed points" in p for p in check(changed))


@pytest.mark.parametrize("points", [0.0, -1.5])
def test_low_or_negative_points_are_kept(points):
    assert week(build(make_tables([stat_row("A", 1, points=points)])), 1)["ppr_points"] == points
