---
title: Rising and Crashing Detection with Phone Export - Plan
type: feat
date: 2026-10-23
topic: rising-crashing-export
artifact_contract: ce-unified-plan/v1
product_contract_source: ce-brainstorm
execution: code
---

# Rising and Crashing Detection with Phone Export - Plan

## Goal Capsule

- **Objective:** Each weekly run tells the owner which tracked games are rising in price and which are crashing, and leaves that answer in a small file their phone can download from a private place.
- **Product authority:** The owner's intent as relayed by firstmate. This plan covers the pipeline half only (detection, export, publish). The Android app that reads the file is a separate plan and pull request, and is not active scope here.
- **Open blockers:** None.
- **Means:** a pure flagging module plus export and publish stages appended to `ra run` (KTD1-KTD6).
- **Stop conditions:** stop and ask firstmate if any requirement turns out to need a product change, or if publishing cannot be tested without touching the real results repository.
- **Execution profile:** test-first per unit; every file stays under 500 lines.
- **Who finishes:** the firstmate crewmate implements on branch `fm/ra-pipeline` and ships through `/no-mistakes`; the pull request merges before the Android app pull request.

---

## Product Contract

### Summary

The pipeline gains a flagging step that marks a game as rising or crashing from its forecast growth and from how much its price actually moved over the last 1, 3 and 6 months, using thresholds the owner can change.
A new `ra export` command writes the flagged games, with enough history and forecast for a phone to draw them, as one versioned JSON file.
An optional publish step commits that file to the owner's private results repository, and both run as part of `ra run` and therefore the weekly timer.

### Problem Frame

Today `ra run` ranks every tracked game by forecast growth into `results.csv` and `report.html`.
The ranking puts the best risers on top, but a game whose price just fell 30% only shows up as a low rank at the bottom of a long table, and the ranking ignores what the price has already done.
The owner also wants to check this on their phone, while the scrape and models stay on their PC.
The report is a local file on the PC, so the phone has nothing to read.

PriceCharting's terms allow personal use but not publishing its prices, so anything that leaves the PC must stay private to the owner.

### Key Decisions

- **Flags come from two signals, forecast and realized change, in both directions.** A forecast alone misses a crash that already happened; realized change alone misses a trend the model sees coming. Rising uses the same signals as crashing so the two tabs mean the same thing in opposite directions. Governs R1, R2, R3.
- **A game can carry both flags.** For example a game that fell 20% this month but whose model forecasts recovery is both crashing (realized) and rising (forecast). Each flag carries its reasons, so the reader sees why. Governs R4.
- **The PC decides what is flagged; the export holds only flagged games.** The phone only downloads and displays, so the thresholds live with the pipeline and the file stays small. Governs R6, R7.
- **One owner-edited settings file holds thresholds and the publish target.** The weekly timer runs `ra run` without arguments, so command-line flags would mean editing the systemd unit to change a threshold. The settings file lives in the gitignored data directory. Governs R5, R10.
- **Publishing refuses any target it cannot confirm is private.** Pushing PriceCharting prices to a public repository breaks its terms and cannot be undone once indexed, so the check must run before every push. Governs R12.

### Requirements

**Detection**

- R1. For each ranked game the pipeline computes the realized CIB price change over the last 1, 3 and 6 months, as a percentage of the price that many months before the latest month.
- R2. A game is flagged rising when its forecast growth is at or above the rising forecast threshold, or when any realized change from R1 is at or above that window's rising threshold.
- R3. A game is flagged crashing when its forecast growth is at or below the crashing forecast threshold, or when any realized change from R1 is at or below that window's crashing threshold.
- R4. Each flag records every condition that triggered it, as a machine-readable reason plus a short human sentence (for example "down 24% in 3 months").
- R5. Defaults apply when the owner sets nothing: rising at forecast growth of +10% or more, or realized +15% over 1 month, +25% over 3 months, +40% over 6 months; crashing at forecast growth of -10% or less, or realized -10% over 1 month, -15% over 3 months, -25% over 6 months. Each of the eight thresholds can be overridden individually.
- R6. A window the game has too little history for is skipped for that game, not treated as zero change.

**Export**

- R7. `ra export` writes one JSON file containing only flagged games, plus the generation date, the thresholds used and a schema version number.
- R8. Each exported game carries: console, game slug, display title, PriceCharting page link when known, latest month and CIB price, the realized changes from R1, the six-month forecast (monthly points and final growth), the last 24 monthly prices, and its flags with reasons.
- R9. `ra run` runs the export after the ranking, so the weekly timer produces a fresh file with no unit change.

**Publish**

- R10. When the owner has configured a results repository, `ra run` (and a standalone `ra publish`) commits the exported file to that repository at the configured path and branch and pushes it, using the PC's existing git credentials. The documented example target is `git@github.com:alutterb/retro-analytics-data.git`, file `results/latest.json` on `main`.
- R11. With no results repository configured, publishing does nothing and says so in one line; the run still succeeds.
- R12. Before pushing, publishing confirms the target repository is private and refuses, with an error naming the repository, when it is public or its visibility cannot be determined.
- R13. When the exported file is unchanged from what the repository already holds, publishing makes no commit.
- R14. A publish failure leaves the local export in place and makes `ra run` exit non-zero after the other stages have finished, matching how a partial scrape is reported today.

**Privacy**

- R15. The settings file, the export and any local copy of the results repository live under the gitignored data directory, so no price data, token or credential can be committed to retro-analytics.

### Acceptance Examples

- AE1. **Covers R2, R3, R4.** Given a game whose forecast growth is +4% and whose price fell from $100 to $80 over the last 3 months, when the run flags games, then the game is crashing with one reason "down 20% in 3 months" and is not rising.
- AE2. **Covers R2, R3, R4.** Given a game that fell 12% in the last month and is forecast to grow 15%, then it is both crashing (realized, 1 month) and rising (forecast), each with its own reason.
- AE3. **Covers R6.** Given a game with only 4 months of prices, then its 6-month change is absent, and only the 1- and 3-month windows can flag it.
- AE4. **Covers R11.** Given no results repository in the settings, when `ra run` finishes, then it prints that publishing is not configured and exits 0.
- AE5. **Covers R12.** Given a configured repository that the visibility check reports as public, then nothing is pushed and the command exits non-zero naming the repository.
- AE6. **Covers R13.** Given two runs on the same data, then the second publish creates no commit.

### Scope Boundaries

- The Android app, its notifications and its own settings: separate plan.
- Showing flags in `report.html` or new columns in `results.csv`: deferred; the existing ranking outputs stay as they are.
- Creating the results repository: firstmate created it after the owner confirmed; the pipeline never creates repositories.
- Publishing anywhere other than a git repository (cloud storage, a web server): out of scope.
- Keeping a history of past exports beyond what git history of the results repository already gives.

### Dependencies / Assumptions

- The PC can push to the results repository with its existing git credentials (SSH key or `gh` credential helper).
- The privacy check can use the GitHub CLI (`gh`), which is installed and authenticated on the owner's PC.

### Outstanding Questions

- Deferred to Planning: exact JSON field names and the settings file format and location.
- Deferred to Planning: whether the 24-month history window and default thresholds become named constants next to the existing report constants.

<!-- ce-section: work-relationships -->
### How This Work Fits Together

This plan covers the pipeline half of putting retro-analytics on the owner's phone. The breakdown below is the current understanding, not a committed roadmap.

- Android app (Rising and Crashing tabs, console filter, game detail with price history and forecast, notifications, offline cache): Depends on the export format from R7 and R8 and on the published file from R10. Can be built in parallel once the format is fixed.

### Sources / Research

- `src/retro_analytics/metrics.py`: current ranking and `growth_pct`.
- `src/retro_analytics/report.py`: 24-month sparkline window reused for R8.
- `src/retro_analytics/pipeline.py`: `run()` stage order and the partial-failure pattern R14 follows.
- `deploy/systemd/retro-analytics.service`: runs `ra run` with no arguments.

---

## Planning Contract

Product Contract preservation: Product Contract unchanged.

### Assumptions

- Firstmate confirmed scope and told the run to proceed through the loop without another scoping checkpoint, so plan-time bets are recorded here instead of being asked.
- Realized change reads the monthly series written by `ra prep` (`data/preprocessed/series.csv`), whose gaps are already interpolated; a window counts as available when the month that many months before the latest month is present in that game's series.
- Percentages in the export are rounded to one decimal and prices to cents; threshold comparisons use the unrounded values.

### Considered and Not Built

- A fallback git identity for publish commits: the owner's PC already has one, and a missing identity fails loudly as a `PublishError` (R14). Revisit if publishing from a fresh machine becomes a goal.

### Key Technical Decisions

- KTD1. **Settings live in `$RA_DATA_DIR/settings.toml`, parsed with the standard library `tomllib`.** Python 3.12+ ships it, so no dependency is added; a missing file means all defaults, an unknown table or key is an error naming the file, so a typo cannot silently fall back to a default. A commented example ships at `examples/settings.toml`. Instantiates the Key Decision on one settings file (R5, R10).
- KTD2. **Flagging is a pure function over the ranking and the monthly series.** It lives in its own module and returns plain records, so the export, tests and any later report change share one implementation (R1-R6).
- KTD3. **Export schema version 1 is a single JSON object, written to `data/outputs/latest.json`.** Shape below; the Android app pins to `schema_version` 1 and the field names are a shared interface, so later changes add fields or bump the version (R7, R8).
- KTD4. **Publish clones into a throwaway directory inside the data directory on every run.** A fresh shallow clone avoids stale local state; an empty remote (the case today) is handled by initialising the branch locally and pushing it. The weekly file is small, so clone cost is negligible (R10, R13, R15).
- KTD5. **The privacy check calls `gh repo view <owner>/<name> --json isPrivate`.** Only GitHub URLs (`git@github.com:o/n.git`, `ssh://git@github.com/o/n`, `https://github.com/o/n`) can be checked; anything else, a missing `gh`, or any `gh` failure refuses (R12).
- KTD6. **`ra run` appends export then publish after the report, and a publish failure is held like a scrape failure.** The run finishes every stage, then raises the first held error, so the CLI exits non-zero (R9, R14).

### High-Level Technical Design

Directional shape of the export file (KTD3); field names are the contract, values illustrative:

```text
{
  "schema_version": 1,
  "generated": "2026-10-25",            # ISO date of the run
  "horizon_months": 6,
  "thresholds": {
    "rising":   {"forecast_pct": 10,  "change_1m_pct": 15,  "change_3m_pct": 25,  "change_6m_pct": 40},
    "crashing": {"forecast_pct": -10, "change_1m_pct": -10, "change_3m_pct": -15, "change_6m_pct": -25}
  },
  "games": [{
    "console": "nes", "game": "legend-of-zelda", "title": "Legend of Zelda",
    "url": "https://www.pricecharting.com/game/nes/legend-of-zelda",   # or null
    "last_month": "2026-10-01", "last_price": 80.0,
    "changes_pct": {"1m": -12.0, "3m": -20.0, "6m": null},          # null = not enough history
    "forecast": {"growth_pct": 15.0, "months": ["2026-11-01", ...], "prices": [82.1, ...]},
    "history": {"months": ["2024-11-01", ...], "prices": [...]},        # last 24 months
    "flags": [
      {"direction": "crashing", "signal": "change_1m", "value_pct": -12.0, "threshold_pct": -10.0,
       "text": "down 12% in 1 month"},
      {"direction": "rising", "signal": "forecast", "value_pct": 15.0, "threshold_pct": 10.0,
       "text": "forecast up 15% in 6 months"}
    ]
  }]
}
```

Settings file shape (KTD1):

```text
[rising]            # any subset; omitted keys keep the defaults from R5
forecast_pct = 10
change_1m_pct = 15
change_3m_pct = 25
change_6m_pct = 40

[crashing]
forecast_pct = -10
change_1m_pct = -10
change_3m_pct = -15
change_6m_pct = -25

[publish]           # omit the table to disable publishing (R11)
repo = "git@github.com:alutterb/retro-analytics-data.git"
branch = "main"                 # default
path = "results/latest.json"    # default
```

```mermaid
flowchart LR
  rank[ra rank: results.csv] --> report[ra report]
  report --> export[ra export: flags + latest.json]
  export --> publish{publish configured?}
  publish -- no --> skip[print not configured]
  publish -- yes --> private{gh says private?}
  private -- no / unknown --> refuse[PublishError]
  private -- yes --> clone[fresh clone] --> diff{file changed?}
  diff -- no --> nocommit[no commit]
  diff -- yes --> push[commit and push]
```

---

## Implementation Units

### U1. Settings file

**Goal:** load thresholds and publish target from one owner-edited file with validated keys and defaults.
**Requirements:** R5, R10, R11, R15.
**Dependencies:** none.
**Files:** `src/retro_analytics/settings.py`, `src/retro_analytics/paths.py`, `tests/test_settings.py`, `examples/settings.toml`.
**Approach:**
1. Add `paths.settings()` (data dir root) and `paths.export()` (`outputs/latest.json`) next to the existing path helpers.
2. Default thresholds per R5 as module constants; loading merges file values over defaults per key.
3. Reject unknown tables or keys, non-numeric thresholds, and a `[publish]` table without `repo`, with a `PipelineError` naming the file and key.
4. Return publish settings as `None` when the table is absent.
**Patterns to follow:** `paths.py` helpers; `PipelineError` messages in `paths.require`.
**Test scenarios:**
- No settings file: all eight defaults, publish disabled.
- File overriding only `crashing.change_3m_pct`: that value changes, the other seven keep defaults.
- Unknown key `chnage_1m_pct`: error naming the key and file.
- Invalid TOML: error naming the file.
- `[publish]` with only `repo`: branch `main`, path `results/latest.json`.
- `[publish]` without `repo`: error.
- `examples/settings.toml` parses and equals the defaults plus the documented publish target.
**Verification:** settings tests pass; example file is valid.

### U2. Flagging

**Goal:** compute realized changes and rising/crashing flags with reasons for every ranked game.
**Requirements:** R1, R2, R3, R4, R6.
**Dependencies:** U1.
**Files:** `src/retro_analytics/flags.py`, `tests/test_flags.py`.
**Approach:**
1. Realized change for windows 1, 3, 6 from each game's monthly series: latest price over the price exactly k months earlier, minus one; `None` when that month is absent (R6).
2. Compare forecast growth (from `results.csv` `growth_pct`) and each available change against the thresholds; inclusive comparisons (`>=` rising, `<=` crashing).
3. One reason record per triggered condition with direction, signal, value, threshold and human text; a game with no reason is not flagged.
**Patterns to follow:** `metrics.rank` (frame in, frame or records out, no I/O).
**Test scenarios:**
- Covers AE1. Forecast +4%, price 100 to 80 over 3 months: one crashing reason "down 20% in 3 months", no rising reason.
- Covers AE2. Fell 12% last month and forecast +15%: crashing (change_1m) and rising (forecast).
- Covers AE3. Four months of history: `6m` is `None`, only 1m and 3m can flag.
- Exactly at a threshold (forecast -10.0%): flagged crashing.
- Game with no condition met: not returned as flagged.
- Custom thresholds passed in change the outcome versus defaults.
- Rising realized change (+30% over 3 months) flags rising with text "up 30% in 3 months".
**Verification:** flags tests pass; function has no file I/O.

### U3. Export stage

**Goal:** `ra export` writes the schema-1 JSON of flagged games.
**Requirements:** R7, R8.
**Dependencies:** U1, U2.
**Files:** `src/retro_analytics/export.py`, `src/retro_analytics/cli.py`, `tests/test_export.py`, `tests/test_cli.py`.
**Approach:**
1. Read `results.csv`, `series.csv`, `forecasts.csv` via `paths.require`, plus `games.csv` for titles and URLs when present.
2. Build the document per the HTD shape: only flagged games, in ranking order, 24 months of history (reuse `report.SPARK_MONTHS`), forecast points from `forecasts.csv`.
3. Write UTF-8 JSON atomically (temp file then replace) so a publish never reads a half-written file.
4. Print counts: flagged rising, flagged crashing, file path.
**Patterns to follow:** `report.main` (reads the same three inputs), `COMMANDS` table in `cli.py`.
**Test scenarios:**
- Three ranked games, one rising, one crashing, one neither: file has two games and `schema_version` 1.
- Exported game carries every R8 field; `url` null when `games.csv` is missing.
- History is capped at the last 24 months for a 60-month series.
- No flagged games: valid file with an empty `games` list.
- Missing `results.csv`: `PipelineError` saying to run `ra rank` first.
- `ra export` is registered in the CLI and dispatches.
**Verification:** export tests pass; output parses with `json.loads`.

### U4. Publish stage

**Goal:** `ra publish` pushes the export to the configured private repository, or does nothing when unset.
**Requirements:** R10, R11, R12, R13, R14, R15.
**Dependencies:** U1, U3.
**Files:** `src/retro_analytics/publish.py`, `src/retro_analytics/cli.py`, `tests/test_publish.py`.
**Approach:**
1. Unset publish settings: print one line and return (R11).
2. Parse `owner/name` from the repo URL (KTD5); refuse when it does not parse, when it carries credentials (`user:token@`), or when `gh` does not report `isPrivate: true` (R12). Refusals happen before any git call.
3. In a temporary directory under the data directory, shallow-clone the branch; when the remote branch does not exist, initialise an empty repo on that branch with the remote set (KTD4).
4. Copy the export to the configured path. "Unchanged" (R13) compares the new export with the repository's copy after dropping the top-level `generated` field from both; when they match, leave the repository file as it is, print that, and stop. The weekly series is monthly, so most weekly runs in a month see the same data.
5. Commit with message `Update results <generated date>` and push to the branch. Any git failure raises `PublishError(PipelineError)` with git's last stderr line.
**Patterns to follow:** `ScrapeError(PipelineError)` in `pricecharting.py`.
**Execution note:** tests use a local bare repository as the remote and monkeypatch the module-level visibility-check function (URL parsing included); they never contact GitHub. No environment variable, settings key or CLI flag can disable or skip that check.
**Test scenarios:**
- Covers AE4. No `[publish]` table: prints not configured, no git call made.
- Covers AE5. Visibility check reports public: raises naming the repository, remote untouched.
- Non-GitHub URL with the real check: refused before any git call.
- `gh` missing or failing: refused.
- Empty bare remote: first publish creates branch `main` with `results/latest.json`.
- Covers AE6. Second publish of identical file: no new commit on the remote.
- Covers AE6. Two exports with identical games but different `generated` dates: no second commit.
- Repo URL with embedded credentials (`https://user:tok@github.com/o/n`): refused before any git or gh call.
- Changed file: one new commit containing the new content.
- Push failure (remote removed): `PublishError` and the local export still exists.
- URL parsing accepts the three GitHub forms with and without `.git`.
**Verification:** publish tests pass against a local bare remote.

### U5. Pipeline wiring and docs

**Goal:** the weekly `ra run` exports and publishes, and the owner can configure it from the README.
**Requirements:** R9, R14; documents R5, R10-R12.
**Dependencies:** U3, U4.
**Files:** `src/retro_analytics/pipeline.py`, `tests/test_pipeline.py`, `README.md`, `examples/three-games.sh`.
**Approach:**
1. Append export and publish to `pipeline.run` after report; hold a `PublishError` and raise after the run, scrape error first when both happened (KTD6).
2. README: new "Rising and crashing" and "Phone export and publishing" sections with the settings example, the privacy rule, and the credential setup for the headless timer: a repo-scoped SSH deploy key with write access to `retro-analytics-data`, wired through `~/.ssh/config` so no agent is needed, plus `gh auth login` for the visibility check; add `ra export` and `ra publish` to the stage list; update the `ra run` description.
3. Three-game example prints the export path.
**Patterns to follow:** existing scrape-error holding in `pipeline.run`; README stage list format.
**Test scenarios:**
- `ra run` stage order ends with report, export, publish.
- Publish failure: all stages ran, then `PublishError` is raised.
- Scrape and publish both fail: scrape error raised after all stages.
**Verification:** full suite passes; README describes every new command and setting.

---

## Verification Contract

- `uv run pytest -q` passes on Python 3.12 and 3.13 (CI matrix in `.github/workflows/tests.yml`).
- No test touches the network or GitHub; publish tests use a local bare repository.
- `wc -l` shows every changed or new file under 500 lines.
- `git status` after a test run shows no files under `data/` tracked and no settings, tokens or price data added.

## Definition of Done

- U1-U5 land with their tests passing.
- `ra export` and `ra publish` appear in `ra --help`.
- README documents thresholds, the settings file with an example, publishing to `git@github.com:alutterb/retro-analytics-data.git`, and the privacy refusal.
- No dead-end or experimental code from abandoned approaches remains in the diff.
- Nothing is pushed to the real results repository during development.
