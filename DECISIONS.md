# Decisions

Choices made where the README's plan met the real data. Newest first.

## November 2026: player-week table (Milestone 1)

Based on an audit of the raw 2015–2025 tables.

| Problem found | Decision |
|---|---|
| Team codes differ by table. Stats use current codes for every season (`LV` in 2015), while other tables use historical ones (`OAK`, `SD`, `STL`), and 2015 rosters also use `ARZ`, `BLT`, `CLV`, `HST` and `SL`. | Map everything to the current code using `team_codes` in `configs/data.yaml`. |
| Snap counts use PFR IDs, but other tables use GSIS IDs. | Map through `players`, then rosters. About 0.1% of played rows still have no snap row. Those snaps stay null rather than 0. |
| Weekly stats have no row for missed games. | Give each included player-season a row for every regular-season week. |
| About 7,700 skill-position rows have offensive snaps but no stat row, such as blocking tight ends. | Count them as played, with real zeros for counting stats. Shares and WOPR stay null. |
| Roster status isn't comparable across seasons: `CUT` has about 12,000 rows a season in 2017–2019 and about 1,000 or fewer otherwise. | Don't use status to decide who's on a team. Include a player-season only if the player has a stat row or an offensive snap. |
| 2015 and 2019 rosters list some players twice in a week. | Prefer the `ACT` row. |
| Position can differ between tables and weeks, such as a player listed as QB on rosters and TE in stats. | Use one position per player-season: stats, then rosters, then snaps. Fullbacks count as RBs. |
| Team on a missed week can be unknown. | Use stats, then snaps, then rosters, then the nearest known team. About 2.5% of player-seasons have an unusual number of no-game weeks, mostly players who changed teams, so bye detection is less certain for them. |
| The 2022 Bills–Bengals game was cancelled. | Treat it like a bye: null points rather than a missed game. Season length comes from the schedule. |
| Some players have several injury reports in a week. | Keep the latest. 2025 has no timestamps, so ties there are broken by team name. |

The table records what happened and isn't leakage-safe by itself. Feature code must lag week t columns, including `team`, `offense_pct` and `report_status`.

## November 2026: injury report timing

80% of injury rows with a timestamp were last modified on a Friday and fewer than 1% on a Monday or Tuesday. This supports using only the week t-1 report in features, because the week t report usually doesn't exist at the waiver deadline. "Probable" appears only in 2015, so it can't be treated as a consistent category.

## November 2026: betting line timing is unresolved

The nflverse docs define the sign of `spread_line` but don't say whether lines are opening or closing. Historical lines are probably later than a Tuesday deadline. Features must compare validation results with and without them, as the README describes.
