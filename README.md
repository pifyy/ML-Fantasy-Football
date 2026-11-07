# Waiver Wire Predictor

A machine learning project that predicts which NFL fantasy football players on the waiver wire are likely to become valuable pickups, then ranks them for a specific fantasy team and league.

The project has two model tracks:

- **Live model:** gradient-boosted trees with calibrated probabilities: LightGBM, XGBoost or an average of the two, chosen on validation. It powers weekly picks during the 2026 season.
- **Learning model:** a neural network classifier written from scratch in NumPy and trained with gradient descent, benchmarked against the live model and simple baselines. It replaces the live model only if it wins on validation (see Section 6).

Later versions may add probabilistic projections and temporal difference (TD) reinforcement learning (see Roadmap).

## 1. The problem

Each week in season-long fantasy football, managers can add players nobody has rostered (the "waiver wire"). Most of these players stay irrelevant, but a few become reliable starters because of injuries, role changes or breakouts. Picking those few early is a big edge.

**v1 question:** given everything known about a low-ranked player at the waiver deadline before week t, will that player be a "waiver hit" over the following weeks?

### Label definition (binary)

For each player-week (player, season, week t):

**Candidate pool** (proxy for "on the waiver wire"). A player is in the pool only if all of these are true:

- Their season-to-date PPR points through week t-1 rank outside these position cutoffs, on both total points and points per game:
  - QB: top 15
  - RB: top 30
  - WR: top 36
  - TE: top 12
- Their prior-season points per game ranked outside the position's starter range (see the label thresholds below). Players with no prior season, such as rookies, pass this check. This keeps out established starters whose totals are low only because they missed games.
- They have played at least one snap so far this season.
- They play QB, RB, WR or TE. Kickers and team defenses are excluded in v1.

Historical roster percentages aren't freely available, so this rank-based pool stands in for "available on waivers" in training. Live picks use real league rosters instead (see Section 8). Keep the cutoffs and filters in a config file so they're easy to change, and report on validation how much each filter changes the results.

**Scored weeks:** week 1 has no prior data and weeks 2–3 are noisy. Start at week 4 by default, keep the first week in config, and compare alternatives on validation only.

**Label = 1 ("hit")** if the player's average PPR points over weeks t to t+2 would rank inside the position's weekly starter range:

- QB: top 12
- RB: top 24
- WR: top 30
- TE: top 12

How the label is calculated:

- The window is weeks t to t+2, leaving out the player's team's bye week.
- A game the player misses counts as 0. A bye does not count at all.
- The threshold is the PPR score of the Nth-best player at the position in each week, averaged over the same weeks as the player's window.

**Label = 0** otherwise.

Drop rows where the 3-week window runs past the end of the regular season (no label available). The regular season has 17 weeks through 2020 and 18 weeks from 2021, so the end week must come from the schedule rather than a fixed number.

The default scoring format is PPR. If the live league scores differently, change the label to match before training.

Expect a heavy class imbalance (likely well under 10% positives). This is normal and must be handled (see Section 5).

## 2. Data

**Source:** nflverse, via the nflreadpy Python package (fall back to nfl_data_py if needed). All free.

### NFL data for training and live picks

| Function | Used for | History | Live updates |
|---|---|---|---|
| `load_player_stats` | Points, targets, carries, receptions, air yards, pass attempts, sacks, labels | Check in Milestone 1 | Nightly after game days |
| `load_snap_counts` | Snap share and trends | 2012+ | 4 times a day |
| `load_injuries` | Injury report status | 2009+ | Daily |
| `load_rosters_weekly` | Player's team that week, roster moves and status | 2002+ | Daily |
| `load_schedules` | Opponent, home/away, byes, season length, betting lines | Check in Milestone 1 | Every 5 minutes |
| `load_ff_opportunity` | Expected fantasy points | 2015+ pulled | Has 2026 data through week 9 |
| `load_pbp` | Red-zone touches (optional, see Section 3) | 1999+ | Same as player stats |
| `load_players` | Age, experience, rookie flag | Not season-based | Not stated |
| `load_ff_playerids` | Links NFL IDs to fantasy platform IDs | Not season-based | Not stated |

### Data audit

Audit findings and how each one is handled are in `DECISIONS.md`. They include inconsistent team codes, different player IDs and missed games having no rows.

**Betting line timing is still unknown:** the nflverse docs don't say whether historical lines are opening or closing (see Section 3).

### Left out of v1

| Data | Why |
|---|---|
| `load_participation` (routes) | Doesn't update during the season, so it can't be used for live picks |
| `load_depth_charts` | Format changed after 2024 from weekly to timestamped, so training and live data wouldn't match |
| Next Gen Stats, PFR advanced stats, FTN charting | Start in 2016, 2018 and 2022, so they don't cover all the training seasons |

### League data for live picks only

- Rostered players, pulled just before each set of picks
- League settings: scoring, roster size, lineup slots, waiver type and day

### Snapshots saved by this project

Save timestamped weekly copies of injury reports, betting lines, rosters, picks and Sleeper trending adds. Historical tables may not show what was known at the waiver deadline, and trending adds have no saved history, so these copies are the only reliable record.

**Seasons:** 2015–2025 regular seasons. The 2026 season is in progress. Do not train on it. It is used for live picks only.

**Storage:** save raw pulls as Parquet in `data/raw/`, processed feature tables in `data/processed/` and live snapshots in `data/live/`. All three are gitignored. Use DuckDB or pandas/Polars to query them. Live picks are the exception: commit them to `reports/live/` so the timestamped record is public.

## 3. Features

**Hard rule: no leakage.** Every feature for week t must use only information available at the waiver deadline before week t. That means stats from weeks ≤ t-1, and information already published by that deadline. Never use week t stats or later.

Waiver claims usually process days before kickoff, often before the final injury report exists. Training on information the live model won't have would make test results look better than live results.

Suggested v1 features (per player-week). Season-to-date and rolling features reset each season.

### Usage (most important)

Use per-game rates rather than season totals, because games played barely carries over between seasons. Unless noted, calculate each stat for last week, as a rolling 3-week mean, and as a trend (change vs. the prior 3 weeks).

**All positions**

- Snap share

**WR and TE**

- Targets per game, receptions per game, target share, air yards share
- Average target depth (air yards per target)
- WOPR: 1.5 × target share + 0.7 × air yards share

**RB**

- Touches per game, carry share
- Targets per game, receptions per game, target share

**QB**

- Pass attempts per game, rush attempts per game
- Fantasy points per dropback (dropbacks = pass attempts + sacks taken)
- TD rate and yards per attempt over the last 5 games. These are a test: one analysis found they best predicted the next game for QBs, even though they are unstable between seasons.

**Low priority:** add these only if the feature group test below shows they help.

- Touchdowns and TD rate for RB, WR and TE, which are very unstable
- Yards per target or carry, which are weak predictors of future efficiency
- Red-zone touches, which are less stable than overall usage and need extra play-by-play processing

### Production

- PPR points: last week, rolling 3-week mean, season-to-date mean
- Expected PPR points from `load_ff_opportunity`: last week, rolling 3-week mean
- Season-to-date positional rank (the same one used for the candidate pool)
- Prior-season points per game

### Opportunity / context

- Teammate injury signal: did a higher-usage player at the same position on the same team miss week t-1, appear as Out or Doubtful on the week t-1 report, go on injured reserve by week t-1, or see their snap share collapse in week t-1?
- Vacated opportunity: the targets and carries per game left behind by that teammate
- Team implied point total for week t (from spread and total line), subject to the rule below
- Opponent's average PPR points allowed to this position, season to date
- Home/away, week number

**Betting lines:** these were the strongest features in the closest published study, so settle their timing in Milestone 1. If historical lines are closing lines, compare validation results with and without them. Keep them only if they clearly help, and record the timing mismatch in `DECISIONS.md`. Live picks use the line available at the deadline.

### Player info

- Position (one-hot), age, years of experience, rookie flag
- Team as of the latest roster data through week t-1

### Feature group test

Most published evidence on these stats measures season-to-season stability among established players, so test them on waiver candidates here:

1. Group the features by type: usage, production, betting lines, injuries and player info.
2. Remove one group at a time and measure the change in 2022 PR-AUC and Precision@5.
3. Keep a group only if removing it clearly hurts.

Don't rely on built-in importance scores for these decisions, since they can overstate a feature's effect. Run this test only on validation seasons.

### Leakage tests

Feature-building code lives in `src/ml_fantasy_football/features/` and must be covered by truncation tests: change or remove all data from week t onward, then assert that week t's features stay the same.

## 4. Train / validation / test split

Split by season, never randomly, so the model is always tested on the future:

| Seasons | Use |
|---|---|
| 2015–2021 | Train |
| 2022 | Tuning and early stopping |
| 2023 | Fit the calibration step |
| 2024–2025 | Test, touched only for final reported results |
| 2015–2025 | Retrain the live model after settings are frozen |
| 2026 | Live picks only |

Fit feature scaling (standardization) on training data only, then apply it to the other seasons. Retraining uses the settings and training length chosen on validation, because no season is left over for early stopping.

Reported test results always come from a model trained without the test seasons. Retraining the live model on 2015–2025 does not make those results available early: don't look at test metrics to decide anything about the live model.

## 5. Models

### Live model: gradient-boosted trees

General tabular benchmarks favor boosted trees at this data size, and feature work tends to matter more than the choice of model. No head-to-head study on NFL data was found, so compare the options here:

- Candidates: LightGBM, XGBoost and an average of the two, as binary classifiers on the same features as every other model
- Light tuning only, since the features should get most of the effort
- One model with position as a feature to start. Try one model per position on validation, since QB and TE have fewer rows.
- Calibrated on 2023 so probabilities can feed the team-fit calculation (see Section 8)
- Frozen and versioned when live picks start, so the live record measures one known model

### Learning model: neural network from scratch

Implement in NumPy only (no PyTorch or TensorFlow for this model). The point of this track is to implement gradient descent and backpropagation directly.

**Architecture**

- Input: standardized feature vector
- Hidden layers: configurable; start with 2 layers of 32 and 16 units
- Activation: ReLU in hidden layers
- Output: 1 unit with sigmoid → P(hit)
- Weight init: He initialization for ReLU layers

**Training**

- Loss: binary cross-entropy. Compare on validation with and without class weighting (positive class weight ≈ negatives / positives), since imbalance corrections can hurt calibration without improving ranking.
- Optimizer: mini-batch gradient descent. Implement plain SGD first, then add momentum. Adam is optional.
- Regularization: L2 weight decay; optionally dropout
- Early stopping on 2022 PR-AUC
- Calibration: fit a calibration step on 2023 before reading outputs as probabilities. This matters most with class weighting, which inflates predicted probabilities.
- All hyperparameters in a config file (`configs/nn.yaml`): learning rate, batch size, layer sizes, L2 strength, epochs, seed

**Required correctness checks (as tests in `tests/`)**

- **Gradient check:** compare backprop gradients against numerical finite-difference gradients on a tiny network. Relative error should be under ~1e-5.
- **Overfit a tiny batch:** the network must drive training loss near zero on ~20 examples. If it can't, there's a bug.
- **Shuffled-label check:** train on randomly shuffled labels. Validation performance should drop to roughly chance. If it doesn't, the pipeline has a bug, such as misaligned rows or labels. This does not detect feature leakage, because shuffling also breaks the link between a leaky feature and the label.
- **Leakage checks:** run the truncation tests in Section 3, and flag any single feature with a suspiciously high AUC.
- **Reproducibility:** fixed seed gives identical results.

## 6. Baselines and model selection

Every model is compared against:

- **Last week's points:** rank candidates by previous week's PPR points.
- **Usage trend:** rank by the change in snap share or target share.
- **Logistic regression** (scikit-learn) on the same features.
- **Crowd benchmark (live only):** Sleeper's trending adds, logged weekly. No saved history exists, so it can't be used on past seasons.

The live model is chosen on validation only:

- Compare ranking on 2022 with PR-AUC and Precision@5.
- Check calibration with out-of-fold Brier scores within 2023, because the calibration step is fit on that same season.
- Use LightGBM unless another option clearly beats it. A close result counts as a tie.

If logistic regression comes close to the boosted trees, prefer the simpler model. The neural net doesn't have to beat the boosted trees to count as a successful learning project; report honestly either way.

## 7. Evaluation

Report on the test seasons (2024–2025):

- PR-AUC (main metric, since positives are rare) and ROC-AUC
- Precision@5 per week: of the model's top 5 recommended pickups each week, how many were hits? Averaged across all weeks. This is the most intuitive, real-world metric.
- Calibration: reliability plot and Brier score. When the model says 30%, about 30% should hit.
- Breakdown by position, including the position mix of the top 5
- Bootstrap confidence intervals, resampling by week. The test set has roughly 150 top-5 picks, so don't claim a win on a small gap.

Produce a results table comparing all models and baselines, plus a few plots (PR curve, calibration plot, training/validation loss curves). Save them to `reports/`.

**Treat suspiciously good results as a bug (most likely leakage) until proven otherwise.** As rough guides, investigate a ROC-AUC above 0.9 or Precision@5 above 60%.

## 8. Live picks

### Weekly run

1. Pull the latest NFL data and save snapshots.
2. Build features using only information available at the deadline.
3. Score players with the frozen live model.
4. If a league is connected, pull its rosters and remove anyone already on a team. Without a league, use the rank-based candidate pool from Section 1.
5. Rank the remaining players: best overall and, with a connected league, best for your team.
6. Log the picks with a timestamp before waivers process.
7. Score earlier picks once their 3-week windows finish.

Available players who rank above the training pool's cutoffs are unlike anything the model trained on. Flag them separately instead of scoring them.

### League adapters

The model never needs to know which platform a league is on. Each platform gets a small adapter in `src/ml_fantasy_football/leagues/` that returns the same things:

```python
class LeagueAdapter(Protocol):
    def get_settings(self) -> LeagueSettings: ...         # scoring, roster size, lineup slots, waiver type and day
    def get_rostered_player_ids(self) -> set[str]: ...    # who's taken
    def get_my_roster(self) -> list[str]: ...             # the user's team
```

Build order:

| Order | Platform | Access | Why this position |
|---|---|---|---|
| 1 | ESPN | Unofficial API; private leagues use `espn_s2` and `SWID` cookies | Largest platform and the first league to support |
| 2 | Sleeper | Documented, read-only, no login | Quick to add and tests the interface |
| 3 | Yahoo | Official API with OAuth | Each user has to complete a login |

All platform IDs map to one player ID through `load_ff_playerids`. Check that it covers all three platforms.

League IDs and settings go in `configs/league.yaml`. Cookies and tokens go in `.env` and are never committed. Storing other people's login cookies is a security issue to solve before opening this up to other users.

ESPN's API is unofficial and could change without notice, so the first adapter task is a small script that pulls real rosters.

### Team-fit ranking

The model estimates how likely a player is to hit. This step estimates how much that player would help a specific team:

```text
team value = P(hit) × max(0, starter-level points − points from the team's current starter)
```

It uses:

- **Lineup rules:** starters per position, including FLEX and superflex spots
- **Weak spots:** the lowest-scoring starter at each position, based on recent points per game
- **The next three weeks:** bye weeks and injured starters
- **Who to drop:** the least useful bench player

This is a rule rather than a trained model, and it doesn't touch the final model test. Precision@5 can't measure it because no historical fantasy rosters are available. Test it on simulated rosters, and log live how many lineup points each suggested pickup would have added.

## 9. Repo structure

```text
ML-Fantasy-Football/
├── README.md
├── DECISIONS.md
├── pyproject.toml
├── .env                   # gitignored; league cookies and tokens
├── configs/
│   ├── data.yaml          # seasons, candidate-pool cutoffs, label thresholds
│   ├── trees.yaml         # LightGBM and XGBoost hyperparameters
│   ├── nn.yaml            # neural net hyperparameters
│   └── league.yaml        # league IDs and platform
├── data/                  # gitignored
│   ├── raw/
│   ├── processed/
│   └── live/              # timestamped weekly snapshots
├── src/ml_fantasy_football/
│   ├── data/              # pulling and caching nflverse data
│   ├── features/          # feature engineering + labels
│   ├── models/
│   │   ├── nn.py          # from-scratch NumPy neural network
│   │   ├── train.py       # training loop, early stopping
│   │   ├── trees.py       # LightGBM, XGBoost and their average
│   │   ├── calibrate.py   # calibration step
│   │   └── baselines.py   # heuristics, logistic regression
│   ├── leagues/           # ESPN, Sleeper and Yahoo adapters
│   ├── decision/          # team-fit ranking
│   ├── live/              # weekly run and pick logging
│   └── evaluate/          # metrics, plots, results table
├── tests/                 # gradient check, leakage tests, sanity checks
├── notebooks/             # exploration only, not source of truth
└── reports/               # generated figures and results
    └── live/              # committed record of timestamped picks
```

## 10. Tech stack

- Python 3.11+, managed with uv
- nflreadpy (or nfl_data_py), requests
- espn-api for the ESPN adapter, if it covers rosters and free agents for 2026
- pandas or polars, duckdb, pyarrow
- numpy (the neural net)
- scikit-learn, lightgbm, xgboost (live model, baselines and metrics)
- matplotlib (plots)
- pyyaml (configs)
- pytest (tests)
- Optional: mlflow for experiment tracking

## 11. Milestones

Live picks are time-sensitive, so they come before the neural net. About nine regular-season weeks remained as of November 6, 2026, and each week of delay means less live data.

No league is connected yet. Live picks can start without one using the rank-based pool, so Milestones 5 and 6 can wait until a league is available.

0. **Setup (done):** add dependencies, gitignore `data/`, and create `configs/`, `tests/` and `DECISIONS.md`. Done when `uv run pytest` runs in a fresh clone.
1. **Data pipeline (done):** pull and cache 2015–2025 data into one clean player-week table, and audit it (see `DECISIONS.md`). Done when row counts reconcile and missed games appear as rows.
2. **Labels and candidate pool:** implement them and check the hit rate per position and season. Done when the top "hits" look like real waiver pickups on manual review. This is the main checkpoint, because a bad label would invalidate everything after it.
3. **Features:** build features with truncation tests passing.
4. **Baselines and boosted trees:** heuristics, logistic regression and calibrated LightGBM, XGBoost and averaged results on validation, plus the feature group test. Any suspiciously good result must be explained.
5. **League adapter (needs a league):** pull real rosters and settings. Add the player ID mapping.
6. **Team-fit ranking (needs a league to use live):** implement it and test it on simulated rosters.
7. **Live picks:** retrain and freeze the chosen model, then start the weekly timestamped log.
8. **Neural net:** from-scratch implementation; gradient check, tiny-batch overfit and reproducibility tests passing.
9. **Train, tune and calibrate:** early stopping on 2022 and calibration on 2023; compare with the other models; commit the frozen configs before any test run.
10. **Final evaluation:** one run on test seasons, with the results table, confidence intervals and plots in `reports/`.
11. **Write-up:** short summary of test and live findings for LinkedIn, including if the boosted trees beat the neural net.

## 12. Open items

- No league is connected yet. A league would also provide scoring format, waiver day and lineup rules. Until then, assume PPR scoring and a Tuesday deadline.
- Whether historical betting lines are opening or closing
- Whether `espn-api` works as needed for 2026, and how well `load_ff_playerids` covers current players (it has ESPN, Sleeper and Yahoo ID columns, with gaps)

## 13. Notes for Claude Code

- Follow the milestones in order and keep each step runnable from the command line (e.g. `python -m ml_fantasy_football.data.pull`, `python -m ml_fantasy_football.models.train`).
- Never use data from after the waiver deadline in a feature for week t. Add a truncation test whenever a new feature is introduced.
- Never tune on the test seasons. Test results are computed once, at the end.
- Never change the frozen live model mid-season without logging a new model version alongside the picks it made.
- Keep the neural network in plain NumPy with clear, commented forward and backward passes. Readability matters more than speed.
- Put thresholds, cutoffs and hyperparameters in `configs/`, not hardcoded.
- Never commit cookies, tokens or `.env`.
- Write docstrings for public functions and keep notebooks out of the main pipeline.
- If a design choice in this README turns out to be impractical (e.g. a stat isn't available), choose a reasonable alternative and note it in `DECISIONS.md`.

## Roadmap (after v1)

- **More league platforms:** Sleeper, then Yahoo, plus secure handling of other people's logins.
- **Probabilistic projections:** predict a points distribution, not just hit/no-hit, to replace the rough points estimates in the team-fit ranking and account for upside and risk.
- **Full decision layer:** add risk preference based on matchup (favored vs. underdog), and use rival rosters to predict who will compete for a player, including waiver priority and FAAB bids.
- **Streamlit demo app.**
- **Season simulator + TD learning:** learn the long-term value of roster states, including FAAB bidding against simulated opponents.
