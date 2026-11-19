# fantasy football probability

A standalone fantasy football matchup engine. Type in your lineup on the left and your opponent's on the right
(ESPN layout: QB, RB, RB, WR, WR, TE, FLEX, D/ST, K, 7 bench, IR). The page gives each player's projection and
spread, and your win probability from 20,000 simulated weeks. It is not connected to ESPN or any league.

- **Win probability** (top). Click it for both teams' score distributions drawn over each other, and a table of
  mean, median, mode, SD, variance, skew and percentiles. It also shows margin odds (win by 10+, lose by 10+, tie)
  and how much of each team's variance comes from each starter.
- **⇄ next to a starter.** Shows every eligible bench player, or any player you type in, with the new team mean,
  team SD and win probability. Click a row for the full analysis with that change, then apply it.
- **i next to a player.** Shows his full outcome distribution and percentiles, chance he plays, and Vegas team
  total. It also shows his expert consensus rank (FantasyPros experts via Boris Chen: average rank, SD, best/worst,
  tier), DraftKings prop lines next to our expected stat line, and recent games.
- **final column.** When a game is over, final points fill in automatically once box scores publish. You can also
  type a score yourself. That player is then fixed, so late-week odds are right (one player left on Monday night).
- PPR / Half / Standard scoring uses ESPN's exact default rules. They were checked against ESPN's own point totals:
  99.6% of offense/kicker games match exactly, and D/ST has an average error of 0.3 points.

The page is plain HTML and JavaScript with no frameworks or libraries: about 3 KB of HTML, 10 KB of JS and
55 KB of data, gzipped. Your lineups are saved in your browser.

## How the engine works

1. **Projection.** One gradient-boosted model per stat and position predicts pass yards, pass TD, INT, rush yards,
   rush TD, receptions, rec yards, rec TD and fumbles. Fantasy points are built from the expected stat line, so
   every scoring format is exact. The inputs are:
   - recent and longer-term usage (targets, carries, air yards, snap share, team shares);
   - share of the usage among teammates actually expected to play, which catches injured-reserve moves;
   - opportunity left by injured teammates;
   - the player's own injury and practice status;
   - Vegas implied team total, spread and game total;
   - how many points the opponent allows to the position, plus home and dome.

   K and D/ST use ridge regression on a few features, mostly Vegas lines. Gradient boosting overfit them.
2. **Calibration.** A per-position isotonic map from raw projection to expected points. It corrects the usual
   problem that very high projections overshoot.
3. **Market blend.** Where DraftKings has a line (yards, receptions), the line is converted from a median to a
   mean and blended in at 60%. With an optional `ODDS_API_KEY`, anytime-TD prices from The Odds API are blended
   the same way. *The 60% is a prior, not fitted: there is no free historical prop data to fit it.*
4. **Outcome distribution.** Each player gets the empirical distribution of real outcomes for players at his
   position and projection level (2023–26). It is skewed, floored at zero, and lumpy for D/ST and K. It is then
   scaled by his own volatility factor, fit on absolute deviations and shrunk with k = 60 pseudo-games. That
   factor is backtested as calibrated (slope 0.97). The distribution is not a bell curve, and this is what lets
   a steady player beat a higher-projected volatile one when you're favored.
5. **Chance he plays.** Measured historically for fantasy contributors:

   | Situation | Chance he plays |
   |---|---|
   | Questionable | 72% |
   | Doubtful | 1% |
   | Out | 0.4% |
   | Did not practice on the last report | 38% |
   | Limited practice | 83% |
   | Not on the report | 97% |

   Mid-week, before game designations exist, practice status is shifted one level up. This is a heuristic.
6. **Correlation.** Players in the same game are drawn jointly, using pair correlations measured from 2023–26
   results:

   | Pair | Correlation |
   |---|---|
   | QB with his WR / TE | +0.26 / +0.23 |
   | Opposing QBs | +0.12 |
   | D/ST against the QB / RB it faces | −0.50 / −0.25 |
   | D/ST with its own kicker | +0.23 |
   | RBs on the same team | −0.07 |

7. **Simulation.** 20,000 weeks in the browser, using the same random numbers for every lineup you compare.
   That keeps swap comparisons stable.

## How good is it (out-of-sample, honest)

Everything below is walk-forward: each season is predicted by models trained only on earlier seasons. The
distribution settings used in the backtests were fit on 2022–24 only.

**Win probability calibration.** 4,200 simulated matchups from 2025–26 (`engine/backtest.json`), Brier 0.230:

| predicted | 30–40% | 40–50% | 50–60% | 60–70% | 70–80% |
|---|---|---|---|---|---|
| actual | 34.7% | 44.2% | 55.0% | 65.0% | 74.4% |

**Head-to-head vs ESPN** on identical player-weeks: 7,376 player-weeks, 2025–26, players who played.

| average error (PPR pts) | ours | ESPN |
|---|---|---|
| all positions | 5.12 | **5.04** |
| QB / RB / WR / TE | 6.28 / 5.33 / 5.31 / 5.02 | **6.17 / 5.19 / 5.23 / 4.84** |
| K | **3.79** | 3.83 |
| D/ST | **4.43** | 4.51 |

ESPN's point projections are still about 1.5% more accurate at the skill positions. That is without the
sportsbook blend, which can't be backtested. Matchup win probability on 7,000 identical matchups:

| | Brier, 2026 | Brier, 2025 | ESPN-style odds of 60%+ |
|---|---|---|---|
| ours | **0.229** | 0.237 | calibrated |
| ESPN "Chance to Win" | 0.232 | **0.234** | won 77% (underconfident) |

ESPN's formula is badly miscalibrated. When it shows 40–50%, that team wins 38% of the time; at 60%+ the team
wins 77%. Ours is calibrated in 2026. In 2025 ESPN's better projections outweighed its formula error. The build
saves every week's projections in `engine/history/` so the live model, with market lines, can be scored as the
season goes.

## Data (all public, refreshed by GitHub Actions)

- nflverse: play-by-play-derived player and team stats, schedules with Vegas lines, injury reports, weekly
  rosters, snap counts.
- Boris Chen weekly tier files: FantasyPros expert consensus average rank and SD.
- DraftKings player prop lines via ESPN's public odds feed. Lines only; prices need `ODDS_API_KEY`.

The workflow (`.github/workflows/update.yml`) runs daily at 8am ET, Sunday before and after the early games, and
Monday before the night game. It rebuilds `data.js` and commits it. Models refit once per NFL week (about 5 min);
other runs reuse them (about 1 min).

## Run locally

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python engine/build.py          # fetch data, fit, write data.js
.venv/bin/python engine/backtest.py       # calibration backtest
open index.html                           # works straight from the file
```

## Limits

- Pregame engine. A game in progress isn't modeled; enter a player's points as final once his game ends.
- Mid-week play probabilities use practice reports. They firm up when Friday designations post.
- Rookies with no NFL games yet aren't listed until they play.
- D/ST scoring matches ESPN exactly in 90% of games. Misses are mostly return-TD attribution.
