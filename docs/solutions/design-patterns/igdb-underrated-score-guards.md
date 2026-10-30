---
title: Why the underrated score dedupes, drops bundles, rolls up parent votes and uses a median prior
module: underrated
date: 2026-10-30
problem_type: design_pattern
component: service_layer
severity: medium
applies_when:
  - Changing the scoring in src/retro_analytics/underrated.py (score, standardized_residuals)
  - Changing how tracked games are matched to IGDB (igdb_match.py) or which IGDB entries are kept
  - Adding another rating or popularity source to the underrated list
tags:
  - igdb
  - underrated
  - scoring
  - bayesian-shrinkage
  - matching
---

# Why the underrated score dedupes, drops bundles, rolls up parent votes and uses a median prior

## Context

`ra underrated` ranks tracked games by how much better IGDB users rate them than their IGDB popularity predicts, within each console (README, "Underrated games"). The scoring in `src/retro_analytics/underrated.py` looks heavier than the one-line idea: it drops IGDB game types 3 and 13, keeps one row per console and IGDB game, adds the parent entry's votes to popularity, shrinks toward the console *median*, and needs 10 votes to list a game. Each of those exists because the first prototype, which lacked it, produced a top list dominated by artifacts. Taking any of them out "to simplify" brings that failure back. The code and its tests show what each guard does but not what goes wrong without it.

## Guidance

What each guard prevents, as seen on the real tracked set (about 3,000 games on 34 consoles):

- **Drop bundles and compilations (`game_type` 3 and 13).** IGDB users rate the individual games, so a compilation's own vote count stays tiny whatever its real popularity. Without this, every collection in the tracked set (MGS Legacy Collection, Metroid Prime Trilogy, The Orange Box) ranked as "underrated".
- **One row per (console, IGDB game).** PriceCharting lists the same game several times (`[Greatest Hits]`, `[Collector's Edition]`, plain). Each listing matches the same IGDB entry, so without the dedupe one game filled several top slots (Silent Hill 2 and Witcher 3 Complete each appeared twice). The plain listing wins, then the busiest one by cached CIB sales per day.
- **Popularity = own votes + parent votes.** Ports, remasters and editions (`parent_game` or `version_parent`) carry few votes of their own while the original carries the audience. Without the roll-up, *Witcher 3 Complete Edition* and *Silent Hill 2: Restless Dreams* looked obscure.
- **Shrink toward the console median, not a vote-weighted mean.** A vote-weighted prior is pulled toward the hits, so a low-vote stinker gets shrunk *up* (Ms. Pac-Man on SNES went from 57 on 5 votes to 76). The median keeps the prior at a typical game.
- **Fit on all rated games, list only those with 10 or more votes.** IGDB publishes a rating from a single vote. The 10-vote floor keeps noise off the list; fitting on every rated game keeps the slope from being estimated on the popular subset only.
- **Pooled slope below 20 rated games; score 0 at two or fewer, or with no spread.** Thin consoles (Virtual Boy has 2 eligible games) cannot estimate their own slope, and a one-game console would otherwise divide by a zero standard deviation and put `NaN` into `latest.json`, which the app's strict parser rejects as a whole file.

Two things that do *not* fix it and were rejected:

- **CIB price as the yardstick** ("quality given price"): within the tracked set price explains about 1% of rating variance, so the list is just the best-rated games.
- **PriceCharting sales per day as popularity**: the values take only about a dozen distinct levels and the tracked set is already chosen by that same volume, so it explains far less than IGDB votes.

Known residual artifacts that remain by design: rare, expensive carts (no affordability filter, deliberately), and ports that inherit the original platform's rating when IGDB has no separate port entry.

Matching notes that keep the list trustworthy: a match needs a score of 90 or more (below it, hand-checked matches were mostly wrong or hardware), numbers in the title must agree, and mods/forks/updates (`game_type` 5, 12, 14) are never candidates. The `score_cutoff` and precomputed-name optimisation in `igdb_match.py` was checked to give byte-identical `igdb_matches.csv` and `underrated.csv` on the real tracked set before it landed; re-run that comparison (clear the cache in a copy of the data directory, rerun `ra underrated`, diff both files) after any matcher change.

## When to Apply

Before removing or loosening any guard in `underrated.score()` or `standardized_residuals()`, or before swapping in a different rating or popularity source: re-run the stage on a copy of real data and read the top 25 per console, not only the tests. The tests pin each mechanism on synthetic frames; only the real list shows whether the artifacts above came back.
