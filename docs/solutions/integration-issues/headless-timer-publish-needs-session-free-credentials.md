---
title: Headless weekly timer cannot use gh keyring tokens and can hang on git
module: publish
date: 2026-10-23
problem_type: integration_issue
component: infrastructure
severity: high
symptoms:
  - "`ra publish` works from a shell, but the weekly systemd timer fails with `cannot confirm <repo> is private: gh failed`"
  - A stalled SSH connection during `git clone` or `git push` leaves the oneshot unit active forever, so later weekly runs never start
root_cause: incomplete_setup
resolution_type: config_change
related_components:
  - background_job
  - authentication
tags:
  - systemd
  - gh-cli
  - keyring
  - git-timeout
  - oneshot
  - publish
---

# Headless weekly timer cannot use gh keyring tokens and can hang on git

## Problem

`ra run` publishes `latest.json` to the owner's private results repository from a systemd user timer. Two things that work fine in an interactive shell break when the timer fires with nobody logged in: the `gh` privacy check has no token, and a git network call can block with no timeout.

## Symptoms

- From a terminal, `ra publish` succeeds. The timer run exits non-zero with `cannot confirm alutterb/retro-analytics-data is private: gh failed (...)`, and the phone keeps showing old data. Nothing visible fails, because interactive runs still work.
- If the network stalls during `git ls-remote`, `clone` or `push`, the `Type=oneshot` service stays `activating`. The timer cannot start a unit that is still running, so later weekly refreshes are silently skipped.

## What Didn't Work

- The first README told the owner to run `gh auth login` once "so the privacy check works". On this machine `gh auth status` reports `Logged in to github.com account alutterb (keyring)`. The token lives in the desktop keyring (gnome-keyring), and `~/.config/gh/hosts.yml` holds no token. The README also recommends `loginctl enable-linger`, and the timer has `Persistent=true`. A run at boot or under linger can therefore start before anyone unlocks the keyring.
- An earlier draft of the README pointed `Host github.com` in `~/.ssh/config` at the deploy key with `IdentitiesOnly yes`. That would have broken the owner's normal GitHub SSH access, because a deploy key only opens one repository. Scoping the key to the service with `GIT_SSH_COMMAND` avoids that.
- The `gh` call already had `timeout=60`, but the git calls had none. Code review found the gap (two reviewers independently), and an independent validator confirmed it. The code permitted the hang; how often git actually stalls was not measured.

## Solution

Give the timer credentials that need no session, and make every network call fail instead of waiting.

- **Privacy-check token.** Use a fine-grained, read-only token for the results repository (the same kind the phone app uses) in a `chmod 600` env file. The unit loads it with a commented-out opt-in, `#EnvironmentFile=%h/.config/retro-analytics/env` (`deploy/systemd/retro-analytics.service:17`). `gh` reads `GH_TOKEN` from the environment before it looks at the keyring.
- **Push credential.** A repo-scoped deploy key, passed only to the service through `Environment="GIT_SSH_COMMAND=ssh -i %h/.ssh/retro_analytics_data -o IdentitiesOnly=yes"` (same unit, commented opt-in).
- **No hangs.** `_git` runs every git command with `timeout=GIT_TIMEOUT` (300 s) and `GIT_TERMINAL_PROMPT=0`. It turns `TimeoutExpired` and `OSError` (git missing) into `PublishError` (`src/retro_analytics/publish.py:63-72`), which `ra run` reports after every other stage has run. The unit also sets `TimeoutStartSec=2h` (`deploy/systemd/retro-analytics.service:14`) as a backstop for the scrape itself.
- **Same host for check and push.** `gh` is asked about `github.com/<owner>/<name>` explicitly (`src/retro_analytics/publish.py:51`), so a `GH_HOST` setting cannot make it vouch for a different host than the one git pushes to.

## Why This Works

A systemd user service started by a timer gets the user manager's environment, with no graphical session and no unlocked Secret Service keyring. A `GH_TOKEN` in the service environment takes precedence in gh's token lookup, so the check no longer depends on a login. systemd turns off the start timeout for `Type=oneshot` units by default, so nothing ends a blocked child unless the code or the unit sets a limit. With per-call timeouts, a stall becomes an ordinary publish failure: the local export stays in place, the run exits non-zero, and the next week starts normally.

## Prevention

- When a job meant to run unattended calls `gh`, `git` or anything else that authenticates, check where its credential is stored (`gh auth status` prints `(keyring)` or the hosts file). Assume the keyring is locked.
- Every `subprocess.run` that touches the network gets a `timeout=`. Git also gets `GIT_TERMINAL_PROMPT=0`. Any `Type=oneshot` unit gets an explicit `TimeoutStartSec=`.
- Keep job-only credentials out of global config (`~/.ssh/config`, global `gh` auth). Pass them in the unit, so interactive use keeps the owner's normal identity.
- Tests cover both failure modes without a network: `tests/test_publish.py` replaces `subprocess.run` with one that raises `TimeoutExpired`, and empties `PATH` so git cannot be found. Both must raise `PublishError`.

## Related

- `docs/plans/2026-10-23-1930-feat-rising-crashing-export-plan.md` (R12, R14, U4, U5).
- README sections "Phone export and publishing" and "Weekly refresh".
