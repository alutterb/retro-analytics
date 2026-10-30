---
title: Underrated Games Tab - Plan
type: feat
date: 2026-10-29
topic: underrated-tab
artifact_contract: ce-unified-plan/v1
product_contract_source: ce-brainstorm
execution: code
---

# Underrated Games Tab - Plan

## Goal Capsule

- **Objective:** On their phone, the owner can browse a per-console list of tracked games that IGDB players rate higher than their popularity predicts, with enough evidence on each row (votes, CIB price) to judge how far to trust it.
- **Product authority:** The owner's intent as relayed by firstmate: variant A of the underrated-metric investigation, with no affordability filter. Critic-vs-user, underpriced-gem, sales-volume and Reddit early-discovery variants are not active scope, nor is filtering hardware out of upstream selection.
- **Open blockers:** None.
- **Means:** the weekly pipeline matches tracked games to IGDB, scores them and adds the list to the results file it already publishes; the Android app shows it as a fourth tab (KTD1-KTD8).
- **Stop conditions:** stop and ask firstmate if a requirement needs a product change, if IGDB access fails for a reason other than a code bug, or if a source file would pass 500 lines.
- **Who finishes:** the firstmate crewmate implements on `fm/ra-underrated-tab` and ships through `/no-mistakes`.

---

## Product Contract

### Summary

The weekly `ra run` fetches IGDB user ratings for the tracked games, scores each game by how much better it is rated than its IGDB popularity predicts within its console, and publishes the top of each console's list alongside the rising and crashing games.
The app gains an "Underrated" tab that lists those games with their rank, vote count and CIB price, opens a detail page on tap, and keeps its scroll position like the other tabs.

### Problem Frame

The app answers "what is moving in price" but not "what is worth playing that people overlook".
An investigation found that IGDB covers about 95% of the tracked games on the right platform (50 of 50 hand-checked matches correct) and that the shrunk user rating, compared against popularity within each console, yields a usable hidden-gem list.
Retro consoles have thin rating data, so the list is only trustworthy when the evidence behind each row is visible.

### Key Decisions

- **No affordability filter.** Rare and expensive gems (Golden Axe Warrior, Suikoden II, Windjammers) stay in the list; the price on each row lets the owner discount them. (session-settled: user-directed — chosen over the investigation's recommended filter to games at or below the console median in popularity and CIB price: the owner said "no filter needed".) Governs R6.
- **Thin consoles keep their short lists.** Virtual Boy, 7800, CD-i, Lynx, 32X, TG16 and Jaguar show whatever few eligible games they have rather than being hidden. (session-settled: user-directed — chosen over hiding those consoles from the tab: stated in the owner's intent.) Governs R7.
- **Each console publishes its top 25.** A console's list stops at 25 games; the bottom of a full ranking is "overrated", which does not belong on an Underrated tab, and the cap keeps the results file small. Governs R7.
- **The underrated list is separate from the flagged games.** It is its own list in the results file, so a game can be underrated without being flagged and older app versions ignore it. Governs R9, R10.
- **The detail page shows evidence, not a price chart.** It shows the score, rank, IGDB rating, votes, popularity and CIB price with links to PriceCharting and IGDB; carrying price history for every underrated game would double the results file for little gain. Governs R13.
- **The weekly run refreshes IGDB.** Refreshing ratings costs about ten requests once games are matched, so it piggybacks on the weekly run rather than adding a monthly schedule. Governs R1.

### Requirements

**Pipeline**

- R1. Each `ra run` refreshes IGDB ratings and vote counts for the tracked games and recomputes the underrated list before the export.
- R2. IGDB credentials come from the `TWITCH_CLIENT_ID` and `TWITCH_CLIENT_SECRET` environment variables and are never printed, logged, written to disk or committed.
- R3. When either credential is missing, the underrated step prints a warning and is skipped, and the rest of the weekly run, including export and publish, proceeds and exits as it would have without it.
- R4. Each tracked game's IGDB match is cached, so only newly tracked games are matched against IGDB, using the investigation's title normalization, digit-agreement rule and acceptance score of 90.
- R5. The owner can correct or suppress a game's IGDB match in a manual override file that the pipeline applies over the automatic match.
- R6. The score is the within-console standardized residual of a Bayesian-shrunk IGDB user rating (K = 10, prior = the console's median rating) regressed on log IGDB popularity (the entry's votes plus its parent game's votes), with no price or popularity filter.
- R7. A game is eligible with at least 10 IGDB votes; the list has one row per console and IGDB game, drops bundles and compilations (IGDB game_type 3 and 13) and hardware rows, and keeps each console's top 25 by score, fewer when fewer are eligible.
- R8. IGDB requests stay under 4 per second.

**Results file**

- R9. The published results file carries the underrated list, each row with its console, PriceCharting slug and title, PriceCharting link, IGDB link, score, rank within its console, IGDB user rating, vote count, popularity and latest CIB price.
- R10. The file's schema version does not change; a reader that predates the underrated list keeps working.

**App**

- R11. The app has a fourth tab, "Underrated", next to Rising, Crashing and Rebound, with the same console filter and sort control as the other tabs.
- R12. Each row shows the title, console, rank within its console, vote count and CIB price; the default order is by score, highest first.
- R13. Tapping a row opens a detail page with the game's score, rank, IGDB user rating, votes, popularity and CIB price, and buttons to open it on PriceCharting and IGDB.
- R14. Returning from the detail page restores the Underrated list's scroll position, as the other tabs do.
- R15. A results file without the underrated list loads normally, and the Underrated tab says no underrated games are available.

### Acceptance Examples

- AE1. **Covers R3.** With no `TWITCH_CLIENT_ID` in the environment, `ra run` prints a warning that the underrated step was skipped, still writes and publishes the results file, and exits zero when nothing else failed.
- AE2. **Covers R6, R7.** A console with three eligible games lists all three, ranked; a game with 9 votes is not listed; a compilation entry is not listed even with 500 votes.
- AE3. **Covers R5.** An override pointing `gameboy/bomberman` at a different IGDB id is used on the next run without rematching anything else; an override with no id removes the game from the list.
- AE4. **Covers R14.** Scrolled to the 30th Underrated row, the owner opens a game and goes back; the list shows the same row at the top.
- AE5. **Covers R15.** An app given last week's results file, which lacks the underrated list, shows Rising, Crashing and Rebound as before and an empty Underrated tab.

### Scope Boundaries

- Critic-vs-user gap, underpriced gem, sales-volume popularity and the Reddit early-discovery variant.
- Filtering hardware out of `ra seed` selection; this work only keeps hardware out of the underrated list.
- An affordability filter or "include rare/expensive" toggle.
- Notifications for games that newly become underrated.
- Flagging rows whose IGDB rating may come from another platform; deferred for later.
- Showing the underrated list in the HTML report.

### Dependencies / Assumptions

- A Twitch application's client id and secret, stored in `~/.config/retro-analytics/igdb.env`, which the systemd service loads with an `EnvironmentFile=` line the owner adds by hand.
- IGDB is free for this personal, non-commercial use.

### Sources / Research

- The underrated-metric investigation (scout report kept outside this repository): matching method, prototype scoring code and per-console coverage. The constants this plan needs from it (platform ids, override ids) are recorded under Assumptions and KTD6; its hardware title pattern is ported into `underrated.py`.
- `src/retro_analytics/export.py` and `android/app/src/main/kotlin/io/github/alutterb/retroanalytics/data/Results.kt`: the results file and its reader.
- `android/app/src/main/kotlin/io/github/alutterb/retroanalytics/ui/ScrollMemory.kt`: per-tab scroll memory from the scroll-position change.

---

## Planning Contract

Product Contract preservation: Product Contract unchanged.

### Assumptions

- IGDB platform ids, confirmed against the IGDB `platforms` endpoint: nes 18, super-nintendo 19, nintendo-64 4, gamecube 21, wii 5, nintendo-switch 130 + 508, gameboy 33, gameboy-color 22, gameboy-advance 24, nintendo-ds 20 + 159, sega-genesis 29, sega-cd 78 + 482, sega-dreamcast 23, playstation 7, playstation-2 8, playstation-3 9, playstation-4 48, psp 38, xbox 11, xbox-360 12, xbox-one 49, sega-saturn 32, turbografx-16 86 + 150, neo-geo-aes 80 + 79 + 136, 3do 50, jaguar 62 + 410, virtual-boy 87, atari-2600 59, atari-7800 60, atari-lynx 61, sega-master-system 64, sega-game-gear 35, sega-32x 30 + 482, cd-i 117.
- The `game_type` field (not the older `category`) carries the IGDB entry type, as in the investigation's pull.

### Key Technical Decisions

- KTD1. **Three small Python modules.** `igdb.py` is the HTTP client (token, throttle, retries, paging); `igdb_match.py` holds title normalization and matching; `underrated.py` holds the platform map, the match cache, scoring and the `ra underrated` stage. Splitting keeps every file well under 500 lines and lets matching and scoring be tested without HTTP.
- KTD2. **Add `rapidfuzz`.** The investigation's matcher scores fuzzy candidates with rapidfuzz `ratio` and `token_sort_ratio`; difflib would be far slower over tens of thousands of candidate names and would change scores. It is a small wheel with no runtime dependencies.
- KTD3. **Credentials go in a POST body, never a URL.** The token request sends `client_id`, `client_secret` and `grant_type` as form data, so a `requests` error message (which includes the URL) cannot contain the secret. The token stays in memory for one run and is never cached to disk. Error messages name the endpoint and status only.
- KTD4. **Match once; misses stay misses.** `data/raw/igdb_matches.csv` keeps one row per tracked game (console, game, igdb_id, match score); a miss is cached with a blank igdb_id. Only games without a row are matched, and only consoles with such games trigger a platform pull. A cached id that the rating refresh no longer returns (IGDB merged or removed it) loses its row, so the next run rematches it. The owner fixes a persistent miss with an override (KTD6). Governs R4.
- KTD5. **Missing credentials reuse last week's scores.** The stage warns and returns without touching `data/outputs/underrated.csv`; the export reads whatever file is there and joins this week's CIB prices. With no file, the results file carries an empty list. An IGDB failure with credentials present behaves like a failed publish: the export and publish still run with the previous scores, then `ra run` exits non-zero. Governs R3.
- KTD6. **Overrides: packaged defaults plus an owner file.** `src/retro_analytics/igdb_overrides.csv` ships the investigation's known wrong matches with ids looked up on the IGDB games endpoint: `psp/final-fantasy` to 397 (Final Fantasy: 20th Anniversary Edition) and `gameboy/bomberman` to no match (its IGDB candidates are ambiguous). Mega Man 8 on Saturn already matches the Saturn port entry of the right game, so it needs none; `data/igdb_overrides.csv` (gitignored, owner-written) wins over it. A blank igdb_id means "no match". Overrides apply after the cache, so editing one takes effect on the next run without rematching. Governs R5.
- KTD7. **The stage writes every eligible game; the export cuts to 25.** `underrated.csv` keeps the full ranked eligible list for inspection; `export.build` keeps rank 1-25 per console, restricted to games still in `games.csv`, and attaches `last_price`/`last_month` from `results.csv` (null when the game has no ranking). Governs R7, R9.
- KTD8. **A `Tab` enum in the app.** The app's tabs become `Tab { RISING, CRASHING, REBOUND, UNDERRATED }`, each with an optional `Direction`; `UiState.tab` and `ScrollMemory` key on `Tab`. Enum names match `Direction` names, so notification intents keep opening the right tab. Underrated rows are a separate `Underrated` type on `Results` with an empty-list default, so older files parse unchanged. Governs R10, R11, R14, R15.

### High-Level Technical Design

Directional only:

```
ra run: scrape -> prep -> train -> forecast -> rank -> report
        -> underrated (skip with warning if no TWITCH_*; error deferred to end)
        -> export (games + underrated[rank<=25] joined with results.csv prices)
        -> publish

underrated stage:
  games.csv --(cache hit?)--> igdb_matches.csv <-- match(platform pull) for misses
            --overrides--> igdb ids --(refresh ratings, 500/request; parent votes)-->
  dedupe per (console, igdb id), drop hardware + game_type 3/13
  -> shrunk = (n*rating + 10*median) / (n + 10) over rated games
  -> residual of shrunk on log1p(votes + parent votes), per-console slope (pooled below 20 games)
  -> standardize by console residual sd -> keep votes >= 10 -> rank within console
  -> data/outputs/underrated.csv
```

Results-file addition (schema_version stays 1):

```
"underrated": [{"console", "game", "title", "url", "igdb_url", "score", "rank",
                "rating", "votes", "popularity", "last_price", "last_month"}, ...]
```

`popularity` is the raw vote total (entry plus parent), not its log; `igdb_url` is the IGDB `url` field fetched with the ratings.

## Implementation Units

### U1. IGDB client

- **Goal:** authenticated, throttled IGDB queries that never leak credentials.
- **Requirements:** R2, R8.
- **Files:** `src/retro_analytics/igdb.py`, `tests/test_igdb.py`, `pyproject.toml`, `uv.lock`.
- **Approach:** `credentials()` reads `TWITCH_CLIENT_ID`/`TWITCH_CLIENT_SECRET` and returns None when either is blank. A `Client` gets a token with a form POST to Twitch, then `query(endpoint, body)` POSTs Apicalypse text with at least 0.26 s between requests, a timeout, and backoff retries on 429 and 5xx; `query_all` pages with `limit 500; offset n`. Failures raise an `IgdbError(PipelineError)` whose message has the endpoint and status only. Injectable session, sleep and clock, like `PoliteSession`.
- **Test scenarios:** missing or blank variables give None; requests are spaced 0.26 s apart on a fake clock; a 429 is retried then succeeds; a 401 raises `IgdbError` whose text contains neither the secret nor the token; paging stops on a short page.
- **Verification:** `uv run pytest tests/test_igdb.py`.

### U2. Title matching

- **Goal:** the investigation's matcher, as a pure function over candidate games.
- **Requirements:** R4.
- **Files:** `src/retro_analytics/igdb_match.py`, `tests/test_igdb_match.py`.
- **Approach:** port `norm`, candidate-name indexing (names, alternative and localized names, colon subtitles, excluding game_type 5, 12, 14), the scoring ladder (100 exact, 99.5 spaceless, digit agreement, 94 word-boundary prefix/suffix, rapidfuzz otherwise, penalties) and tie preference; accept at 90 or more.
- **Test scenarios:** "Super Mario Bros 3" matches "Super Mario Bros. 3"; "Mega Man X" does not match "Mega Man 10"; "Silent Hill 2" does not match "Silent Hill 4"; "Zelda Phantom Hourglass" matches "The Legend of Zelda: Phantom Hourglass"; "Luigi's Mansion [Player's Choice]" matches "Luigi's Mansion"; a mod entry is never chosen; the main game wins a tie with its port; nothing at or above 90 returns None.
- **Verification:** `uv run pytest tests/test_igdb_match.py`.

### U3. Underrated scoring and stage

- **Goal:** `ra underrated` writes the ranked eligible list, and `ra run` calls it.
- **Requirements:** R1, R3, R4, R5, R6, R7.
- **Dependencies:** U1, U2.
- **Files:** `src/retro_analytics/underrated.py`, `src/retro_analytics/igdb_overrides.csv`, `src/retro_analytics/paths.py`, `src/retro_analytics/pipeline.py`, `src/retro_analytics/cli.py`, `tests/test_underrated.py`, `tests/test_pipeline.py`.
- **Approach:** platform map for the 34 default consoles (consoles without one are skipped with a note); cache per KTD4; overrides per KTD6; rating refresh by id (fields include `url`) plus parent ids; `score(frame)` is pure pandas/numpy over (console, game, title, igdb_id, game_type, rating, votes, parent_votes, per_day) and returns ranked eligible rows per R6, R7. Hardware is dropped by the investigation's hardware title pattern (`ra seed` already applies `pricecharting.NON_GAME`, so every hardware row the investigation found got past it). As in the prototype, a console with two or fewer rated games, or zero residual spread, scores 0.0 rather than NaN. `pipeline.run` calls the stage after `report` and before `export`, catching `IgdbError` like `PublishError`.
- **Test scenarios:** shrinkage uses the console median and K = 10; a game with 9 votes is scored in the fit but not listed; game_type 3 and 13 rows are dropped; duplicate IGDB ids on one console keep the plain, most-traded listing; a console under 20 rated games uses the pooled slope; a console with three eligible games lists three; scores have unit sd within a console of more than two rated games; a console with one rated game scores 0.0, not NaN; a hardware title is dropped; with no credentials the stage prints a warning, leaves an existing `underrated.csv` untouched and returns; cached games (hits and misses) are not rematched; a cached id missing from the refresh loses its cache row; an owner override beats a packaged one and a blank id removes the game; `ra run` with an `IgdbError` still exports and publishes, then raises.
- **Verification:** `uv run pytest tests/test_underrated.py tests/test_pipeline.py`.

### U4. Export the underrated list

- **Goal:** `latest.json` carries the underrated rows.
- **Requirements:** R9, R10.
- **Dependencies:** U3.
- **Files:** `src/retro_analytics/export.py`, `tests/test_export.py`.
- **Approach:** `build` takes an optional underrated frame; `main` reads `underrated.csv` when present. Rows per KTD7; numbers rounded like the existing fields; rows with a non-finite number are dropped and the JSON is written with `allow_nan=False`, so a bad score can never make the whole file unreadable; `schema_version` stays 1; the print line adds the underrated count.
- **Test scenarios:** no underrated file gives `"underrated": []`; rank 26 is cut; a game dropped from `games.csv` is left out; a game without a ranking has null `last_price`; a NaN score row is left out; flagged games are unchanged.
- **Verification:** `uv run pytest tests/test_export.py`.

### U5. App data layer

- **Goal:** the app parses and browses underrated rows.
- **Requirements:** R10, R12, R15.
- **Dependencies:** U4 (shape only).
- **Files:** `android/app/src/main/kotlin/io/github/alutterb/retroanalytics/data/Results.kt`, `android/app/src/main/kotlin/io/github/alutterb/retroanalytics/data/Browse.kt`, `android/app/src/test/kotlin/io/github/alutterb/retroanalytics/data/Fixtures.kt`, `android/app/src/test/kotlin/io/github/alutterb/retroanalytics/data/ResultsTest.kt`, `android/app/src/test/kotlin/io/github/alutterb/retroanalytics/data/BrowseTest.kt`.
- **Approach:** `Underrated` serializable class with nullable price fields; `Results.underrated` defaults to empty; `browseUnderrated(results, console, sort)` (score, price with nulls last, title) and a tab-aware console list.
- **Test scenarios:** a file without `underrated` parses with an empty list; a file with it parses every field; a null `last_price` parses; browse orders by score by default, filters by console, puts missing prices last.
- **Verification:** `./gradlew test`.

### U6. Underrated tab, row and detail

- **Goal:** the fourth tab works like the other three.
- **Requirements:** R11, R12, R13, R14, R15.
- **Dependencies:** U5.
- **Files:** `android/app/src/main/kotlin/io/github/alutterb/retroanalytics/ui/Tab.kt` (new), `android/app/src/main/kotlin/io/github/alutterb/retroanalytics/ui/UnderratedScreens.kt` (new), `android/app/src/main/kotlin/io/github/alutterb/retroanalytics/ui/ListScreen.kt`, `android/app/src/main/kotlin/io/github/alutterb/retroanalytics/ui/AppViewModel.kt`, `android/app/src/main/kotlin/io/github/alutterb/retroanalytics/ui/MainActivity.kt`, `android/app/src/main/kotlin/io/github/alutterb/retroanalytics/ui/ScrollMemory.kt`, `android/app/src/test/kotlin/io/github/alutterb/retroanalytics/ui/ScrollMemoryTest.kt`.
- **Approach:** per KTD8; the list composable is generic over row keys so scroll memory works unchanged; the sort label for the first order reads "Score" on the Underrated tab; the empty state reads "No underrated games yet".
  - *Selection:* opening a game records the tab it was opened from; a row tapped on the Underrated tab opens the underrated detail page, resolved from `results.underrated`, even when the same game is also flagged. Back returns to the Underrated list.
  - *Console filter:* switching tabs resets the filter to All when the active console has no rows on the destination tab, and keeps it otherwise.
  - *Row:* its own composable, not `GameRow`. Line 1 the title, line 2 the console, line 3 in the neutral color "#<rank> in <console> - <N> votes"; on the right the CIB price, or "No price" in the muted color when it is null. No "New" badge.
  - *Detail:* title; "<console> - complete in box <price> (<month>)" or "no CIB price" ; facts: Score as a signed number with "standard deviations above what its popularity predicts", rank in console, IGDB user rating out of 100, votes, popularity (votes including the parent game's); buttons "Open on PriceCharting" and "Open on IGDB" when the links exist; a muted caveat "Ratings on retro games come from few voters and may reflect another platform's version."
- **Test scenarios:** scroll memory keeps a separate spot for UNDERRATED; every `Direction` name maps to a `Tab`; the console filter resets on a tab without that console and stays on one with it.
- **Verification:** `JAVA_HOME=~/.local/jdks/jdk-21 ANDROID_HOME=~/Android/Sdk ./gradlew test assembleDebug` in `android/`.

### U7. Docs and service template

- **Goal:** the owner can turn the feature on.
- **Requirements:** R2, R3.
- **Files:** `README.md`, `android/README.md`, `deploy/systemd/retro-analytics.service`, `CONCEPTS.md` if it names tabs.
- **Approach:** README section "Underrated games" (what the score means, credentials setup in `~/.config/retro-analytics/igdb.env` with `chmod 600`, the override file, what happens without credentials); pipeline-stage list and `latest.json` description updated; the service template gains a commented `#EnvironmentFile=-%h/.config/retro-analytics/igdb.env` line.
- **Verification:** read-through; no credential values anywhere.

## Verification Contract

- `uv run pytest` passes.
- `JAVA_HOME=~/.local/jdks/jdk-21 ANDROID_HOME=~/Android/Sdk ./gradlew test assembleDebug` passes in `android/`.
- Smoke: with credentials, `ra underrated` against a copy of a real data directory matches about 95% of non-hardware games and writes a list whose top rows resemble the investigation's variant A list; without credentials it warns and `ra export` still runs.
- `wc -l` shows no source file over 500 lines; `git grep` finds no credential values.

## Definition of Done

- Every unit's verification passes and the global checks above pass.
- No experimental or abandoned code is left in the diff.
- The README states the exact `EnvironmentFile=` line for the owner to add.
