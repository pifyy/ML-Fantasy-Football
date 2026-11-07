"""Build one clean table with a row for every player and regular-season week.

This table records what happened. It is not a feature table: columns such as
the week t team, snaps and injury report describe week t, so feature code must
lag them to respect the waiver-deadline rule.

Each player-season included has at least one offensive snap or stat row, and
gets a row for every regular-season week, so missed games appear as rows:

    team_has_game  played  ppr_points
    false          false   null        bye or cancelled game, not a missed game
    true           false   0           missed game
    true           true    from stats  played

Run from the repo root:
    uv run python -m ml_fantasy_football.data.player_week
"""

import sys
from pathlib import Path

import polars as pl
import yaml

CONFIG_PATH = Path("configs/data.yaml")
RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/player_week.parquet")

KEY = ["player_id", "season", "week"]

STAT_COLUMNS = [
    "attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "sacks_suffered",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_air_yards",
    "target_share",
    "air_yards_share",
    "wopr",
]
# Shares and WOPR stay null when a player has no stat row, since they aren't counts.
COUNT_COLUMNS = [c for c in STAT_COLUMNS if c not in ("target_share", "air_yards_share", "wopr")]

INFO_COLUMNS = ["birth_date", "rookie_season", "draft_round", "draft_pick"]


def load_raw(raw_dir: Path = RAW_DIR) -> dict[str, pl.DataFrame]:
    """Load the cached raw tables, with season and week as integers everywhere."""

    def load_seasons(name: str) -> pl.DataFrame:
        paths = sorted((raw_dir / name).glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No cached files for {name}. Run the data pull first.")
        frame = pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed")
        return frame.with_columns(
            pl.col("season").cast(pl.Float64).cast(pl.Int32),
            pl.col("week").cast(pl.Float64).cast(pl.Int32),
        )

    return {
        "stats": load_seasons("player_stats").filter(pl.col("season_type") == "REG"),
        "snaps": load_seasons("snap_counts").filter(pl.col("game_type") == "REG"),
        "rosters": load_seasons("rosters_weekly").filter(pl.col("game_type") == "REG"),
        "injuries": load_seasons("injuries").filter(pl.col("game_type") == "REG"),
        "schedules": load_seasons("schedules").filter(pl.col("game_type") == "REG"),
        "opportunity": load_seasons("ff_opportunity"),
        "players": pl.read_parquet(raw_dir / "players.parquet"),
    }


def team_week_games(schedules: pl.DataFrame, team_codes: dict[str, str]) -> pl.DataFrame:
    """Return one row per team and game, with betting lines from the team's side.

    spread_line is positive when the home team is favored, per the nflverse docs.
    The docs don't say whether lines are opening or closing.
    """
    base = schedules.select(
        "season", "week", "game_id", "gameday", "location", "total_line", "spread_line",
        "home_team", "away_team",
    ).with_columns(
        pl.col("home_team").replace(team_codes),
        pl.col("away_team").replace(team_codes),
    )
    home = base.with_columns(
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.lit(True).alias("is_home"),
        pl.col("spread_line").alias("team_spread"),
    )
    away = base.with_columns(
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.lit(False).alias("is_home"),
        (-pl.col("spread_line")).alias("team_spread"),
    )
    return (
        pl.concat([home, away])
        .with_columns(
            ((pl.col("total_line") + pl.col("team_spread")) / 2).alias("implied_total"),
            (pl.col("location") == "Neutral").alias("neutral_site"),
        )
        .select(
            "season", "week", "team", "game_id", "gameday", "opponent", "is_home",
            "neutral_site", "team_spread", "total_line", "implied_total",
        )
    )


def _most_common(frame: pl.DataFrame, value: str, alias: str) -> pl.DataFrame:
    """Return each player-season's most common non-null value, breaking ties by name."""
    return (
        frame.filter(pl.col(value).is_not_null())
        .group_by("player_id", "season", value)
        .agg(pl.len().alias("count"))
        .sort("count", value, descending=[True, False])
        .unique(["player_id", "season"], keep="first", maintain_order=True)
        .select("player_id", "season", pl.col(value).alias(alias))
    )


def build_player_week(
    tables: dict[str, pl.DataFrame],
    team_codes: dict[str, str],
    positions: list[str],
    position_map: dict[str, str],
) -> pl.DataFrame:
    """Join the raw tables into one row per player, season and regular-season week."""
    players = tables["players"]

    # Stats: the official record of who played. Null IDs can't be joined to anything.
    stats = (
        tables["stats"]
        .filter(pl.col("player_id").is_not_null())
        .with_columns(
            pl.col("team").replace(team_codes).alias("stats_team"),
            pl.col("position").replace(position_map).alias("stats_position"),
            pl.lit(True).alias("has_stats"),
            pl.col("fantasy_points_ppr").alias("stats_ppr"),
        )
    )

    # Snap counts use PFR IDs, so map them to the GSIS IDs used everywhere else.
    rosters = tables["rosters"].filter(pl.col("gsis_id").is_not_null())
    pfr_to_gsis = (
        pl.concat(
            [
                players.select(pl.col("gsis_id"), pl.col("pfr_id")),
                rosters.select(pl.col("gsis_id"), pl.col("pfr_id")),
            ],
            how="vertical_relaxed",
        )
        .drop_nulls()
        .unique("pfr_id", keep="first", maintain_order=True)
    )
    snaps = (
        tables["snaps"]
        .join(pfr_to_gsis, left_on="pfr_player_id", right_on="pfr_id", how="inner")
        .rename({"gsis_id": "player_id"})
        .with_columns(
            pl.col("team").replace(team_codes).alias("snap_team"),
            pl.col("position").replace(position_map).alias("snap_position"),
        )
        .sort("offense_snaps", descending=True, nulls_last=True)
        .unique(KEY, keep="first", maintain_order=True)
    )

    # Rosters list some 2015 players twice in a week, so prefer the active row.
    roster_rows = (
        rosters.rename({"gsis_id": "player_id"})
        .with_columns(
            pl.col("team").replace(team_codes).alias("roster_team"),
            pl.col("position").replace(position_map).alias("roster_position"),
            pl.col("status").alias("roster_status"),
        )
        .sort(pl.col("status") == "ACT", "roster_team", descending=[True, False], nulls_last=True)
        .unique(KEY, keep="first", maintain_order=True)
    )

    # One position per player-season: stats if it's a kept position, then rosters, then snaps.
    stat_position = _most_common(
        stats.filter(pl.col("stats_position").is_in(positions)), "stats_position", "pos_stats"
    )
    roster_position = _most_common(roster_rows, "roster_position", "pos_roster")
    snap_position = _most_common(snaps, "snap_position", "pos_snaps")

    # Keep player-seasons with a stat row or an offensive snap, at a kept position.
    appeared = pl.concat(
        [
            stats.select("player_id", "season"),
            snaps.filter(pl.col("offense_snaps") > 0).select("player_id", "season"),
        ]
    ).unique()
    player_seasons = (
        appeared.join(stat_position, on=["player_id", "season"], how="left")
        .join(roster_position, on=["player_id", "season"], how="left")
        .join(snap_position, on=["player_id", "season"], how="left")
        .with_columns(pl.coalesce("pos_stats", "pos_roster", "pos_snaps").alias("position"))
        .filter(pl.col("position").is_in(positions))
        .select("player_id", "season", "position")
    )

    # Season length comes from the schedule: 17 weeks through 2020, 18 from 2021.
    season_weeks = (
        tables["schedules"]
        .group_by("season")
        .agg(pl.col("week").max().alias("last_week"))
        .with_columns(pl.int_ranges(1, pl.col("last_week") + 1).alias("week"))
        .explode("week")
        .select("season", pl.col("week").cast(pl.Int32))
    )
    grid = player_seasons.join(season_weeks, on="season")

    # Latest injury report per player-week. 2025 has no timestamps, so nulls sort first.
    injuries = (
        tables["injuries"]
        .filter(pl.col("gsis_id").is_not_null())
        .rename({"gsis_id": "player_id"})
        .sort("date_modified", "team", descending=[True, False], nulls_last=True)
        .unique(KEY, keep="first", maintain_order=True)
        .select(
            *KEY,
            "report_status",
            "practice_status",
            pl.col("report_primary_injury").alias("report_injury"),
        )
    )

    opportunity = (
        tables["opportunity"]
        .filter(pl.col("player_id").is_not_null())
        .unique(KEY, keep="first", maintain_order=True)
        .select(*KEY, pl.col("total_fantasy_points_exp").alias("expected_ppr_points"))
    )

    info = players.select(pl.col("gsis_id").alias("player_id"), *INFO_COLUMNS, pl.col("display_name"))

    table = (
        grid.join(
            stats.select(*KEY, "has_stats", "stats_team", "stats_ppr", "player_display_name", *STAT_COLUMNS),
            on=KEY,
            how="left",
        )
        .join(snaps.select(*KEY, "snap_team", "offense_snaps", "offense_pct"), on=KEY, how="left")
        .join(roster_rows.select(*KEY, "roster_team", "roster_status"), on=KEY, how="left")
        .join(injuries, on=KEY, how="left")
        .join(opportunity, on=KEY, how="left")
        .join(info, on="player_id", how="left")
        .sort(KEY)
        .with_columns(
            pl.coalesce("display_name", "player_display_name").alias("player_name"),
            # Stats and snaps show where a player actually played, so they beat rosters.
            pl.coalesce("stats_team", "snap_team", "roster_team").alias("team"),
            pl.col("has_stats").fill_null(False),
        )
        .with_columns(
            # Weeks with no team listed use the nearest known team.
            pl.col("team").forward_fill().backward_fill().over("player_id", "season"),
            (pl.col("has_stats") | (pl.col("offense_snaps").fill_null(0) > 0)).alias("played"),
        )
        .join(team_week_games(tables["schedules"], team_codes), on=["season", "week", "team"], how="left")
        .with_columns(pl.col("game_id").is_not_null().alias("team_has_game"))
        .with_columns(
            pl.when(~pl.col("team_has_game"))
            .then(None)
            .otherwise(pl.col("stats_ppr").fill_null(0.0))
            .alias("ppr_points"),
            # Played with no stat row, such as a blocker with no targets, means real zeros.
            *[
                pl.when(pl.col("played")).then(pl.col(c).fill_null(0)).otherwise(pl.col(c)).alias(c)
                for c in COUNT_COLUMNS
            ],
            # A missing snap row is only a real zero if the player has no stats either.
            pl.when(pl.col("has_stats"))
            .then(pl.col("offense_snaps"))
            .otherwise(pl.col("offense_snaps").fill_null(0.0))
            .alias("offense_snaps"),
            pl.when(pl.col("has_stats"))
            .then(pl.col("offense_pct"))
            .otherwise(pl.col("offense_pct").fill_null(0.0))
            .alias("offense_pct"),
        )
    )

    return table.select(
        *KEY,
        "player_name",
        "position",
        "team",
        "roster_team",
        "roster_status",
        "team_has_game",
        "played",
        "has_stats",
        "ppr_points",
        "offense_snaps",
        "offense_pct",
        "expected_ppr_points",
        *STAT_COLUMNS,
        "report_status",
        "practice_status",
        "report_injury",
        "game_id",
        "gameday",
        "opponent",
        "is_home",
        "neutral_site",
        "team_spread",
        "total_line",
        "implied_total",
        *INFO_COLUMNS,
    ).sort(KEY)


def validate_player_week(
    table: pl.DataFrame,
    stats: pl.DataFrame,
    schedules: pl.DataFrame,
    positions: list[str],
    position_map: dict[str, str],
) -> list[str]:
    """Return a list of reconciliation failures. An empty list means the table is sound."""
    problems = []

    duplicates = table.height - table.unique(KEY).height
    if duplicates:
        problems.append(f"{duplicates} duplicate player-weeks")

    weeks_in_season = schedules.group_by("season").agg(pl.col("week").max().alias("weeks"))
    rows_per_player = (
        table.group_by("player_id", "season")
        .agg(pl.len().alias("rows"))
        .join(weeks_in_season, on="season")
        .filter(pl.col("rows") != pl.col("weeks"))
    )
    if rows_per_player.height:
        problems.append(f"{rows_per_player.height} player-seasons without a row for every week")

    # Every skill-position stat row must survive, with its points unchanged.
    source = stats.filter(
        pl.col("player_id").is_not_null()
        & pl.col("position").replace(position_map).is_in(positions)
    )
    lost = source.join(table, on=KEY, how="anti")
    if lost.height:
        problems.append(f"{lost.height} stat rows missing from the table")
    kept = source.join(table.select(*KEY, "ppr_points"), on=KEY)
    changed = kept.filter(
        (pl.col("fantasy_points_ppr").fill_null(0) - pl.col("ppr_points").fill_null(-999)).abs() > 1e-6
    )
    if changed.height:
        problems.append(f"{changed.height} stat rows with changed points")

    played_without_game = table.filter(pl.col("played") & ~pl.col("team_has_game"))
    if played_without_game.height:
        problems.append(f"{played_without_game.height} rows played in a week with no team game")

    bad_null_points = table.filter(pl.col("ppr_points").is_null() != ~pl.col("team_has_game"))
    if bad_null_points.height:
        problems.append(f"{bad_null_points.height} rows where null points don't match byes")

    return problems


def main() -> int:
    """Build, validate and save the table. Saves nothing if validation fails."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    tables = load_raw()
    table = build_player_week(
        tables, config["team_codes"], config["positions"], config["position_map"]
    )
    problems = validate_player_week(
        table, tables["stats"], tables["schedules"], config["positions"], config["position_map"]
    )
    if problems:
        print("Validation failed:", *problems, sep="\n  ")
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}: {table.height:,} rows, {table.width} columns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
