---
title: Android app build setup pitfalls (compileSdk, gitignore, SDK install)
module: android
date: 2026-10-24
problem_type: developer_experience
component: tooling
severity: medium
applies_when:
  - Changing Android library versions in android/gradle/libs.versions.toml
  - Adding a Kotlin package or source directory under android/
  - Scripting the Android SDK install on a machine without Android Studio
tags:
  - android
  - gradle
  - compose
  - compilesdk
  - gitignore
  - sdkmanager
---

# Android app build setup pitfalls (compileSdk, gitignore, SDK install)

## Context

The Android app under `android/` was added to a Python repository and is built from the command line with a user-local JDK 21 and Android SDK, without Android Studio. Three problems came up during setup. None of them is visible from the final build files, and each one either breaks the build or silently drops source files from commits.

## Guidance

**Compose libraries set the minimum compileSdk.** Compose BOM 2026.09.00 (Compose UI and Foundation 1.12.x) and lifecycle 2.11.0 declare `minCompileSdk=37` in their AAR metadata. With `compileSdk = 36`, AGP's `checkAarMetadata` task fails the build. The app therefore uses `compileSdk = 37` (`android/app/build.gradle.kts`), and the SDK needs `platforms;android-37.0`, not just 36. Plan review caught this before the first build. When you bump the Compose BOM or AndroidX versions, check the new AARs' `aar-metadata.properties` (or just run `./gradlew assembleDebug`) and install the platform they ask for. The CI job installs platform 37 explicitly for the same reason (`.github/workflows/tests.yml`).

**The root `.gitignore` ignores every directory named `data`.** The Python pipeline ignores price data with a bare `data/` pattern. That pattern also matches the Kotlin package directory `android/app/src/main/kotlin/.../data/`, so `git add` refused the app's data layer. The fix is a scoped negation after it: `!android/app/src/**/data/`. Narrowing the original rule to `/data/` would also work (the pipeline's data, including `data/example/`, lives under the root `data/`), but the negation leaves the pipeline's privacy rule exactly as it was. Any new directory named `data` outside `android/app/src/` is still ignored.

**`yes | sdkmanager --licenses` under `set -o pipefail` aborts the script.** Newer command-line tools print "The --licenses option is no longer needed" and exit. `yes` then dies of SIGPIPE (exit 141), and with `pipefail` that fails the whole install script before any package is installed. Install packages directly, with no license step: `sdkmanager --sdk_root="$HOME/Android/Sdk" "platform-tools" "platforms;android-37.0" "build-tools;36.0.0" < /dev/null`. `android/README.md` documents that form.

## Why This Matters

Each problem costs a confusing round trip. The compileSdk error is reported far from the library bump that caused it. The gitignore problem is worse, because files that were never staged produce no error at all, only a commit missing its data layer. A future change to `.gitignore` that drops the negation would silently untrack the app's core code.

## When to Apply

- Before bumping `composeBom`, `lifecycle` or other AndroidX versions in `android/gradle/libs.versions.toml`.
- When editing the root `.gitignore`.
- When writing or changing the SDK install steps in `android/README.md` or CI.

## Examples

```text
# .gitignore (root)
data/
...
# The top-level data/ rule is for price data, not Kotlin packages.
!android/app/src/**/data/
```

Check that a Kotlin data file is not ignored: `git check-ignore -v android/app/src/main/kotlin/io/github/alutterb/retroanalytics/data/Results.kt` should print nothing.

## Related

- `android/README.md` (toolchain install and build).
- `docs/plans/2026-10-23-1931-feat-android-app-plan.md` (KTD1).
- The Android emulator segfaulted on the owner's PC (exit 139 with both `-gpu swiftshader_indirect` and `-gpu off`), so app verification there is JVM tests, `assembleDebug` and lint; the first real run is on the phone.
