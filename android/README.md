# Retro Prices (Android)

A personal Android app that shows which of your tracked retro games are
rising and which are crashing in price. Your PC does the work: the weekly
`ra run` scrapes PriceCharting, forecasts, flags games and publishes a small
results file to your private results repository. The phone downloads that
file and shows it.

- Rising, Crashing and Rebound tabs (rebound: recently crashed, forecast up), a console filter, and sorting by biggest move,
  price or title.
- A detail page per game with the last 24 months of complete-in-box prices,
  the six-month forecast (dashed), the 1-, 3- and 6-month changes and why
  the game was flagged.
- A notification when a download flags a game that was not flagged that way
  before, as one summary. While notifications are not allowed, those games
  wait and are announced at the first download after you allow them. The
  first download after setup notifies nothing.
- Works offline with the last file it downloaded.

It is for your own phone only: no Play Store, no other users, and the prices
never leave your private repository.

## Build it

Building needs JDK 21 and the Android SDK. Neither needs root; both go in
your home directory.

```sh
# JDK 21 (Eclipse Temurin)
mkdir -p ~/.local/jdks
curl -L -o /tmp/jdk21.tar.gz \
  'https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse'
tar -xzf /tmp/jdk21.tar.gz -C ~/.local/jdks
ln -sfn "$(ls -d ~/.local/jdks/jdk-21.* | tail -1)" ~/.local/jdks/jdk-21

# Android command-line tools and SDK packages
mkdir -p ~/Android/Sdk/cmdline-tools
curl -L -o /tmp/cmdline-tools.zip \
  https://dl.google.com/android/repository/commandlinetools-linux-16111833_latest.zip
unzip -q /tmp/cmdline-tools.zip -d /tmp/cmdline-tools
mv /tmp/cmdline-tools/cmdline-tools ~/Android/Sdk/cmdline-tools/latest
export JAVA_HOME=~/.local/jdks/jdk-21
~/Android/Sdk/cmdline-tools/latest/bin/sdkmanager --sdk_root="$HOME/Android/Sdk" \
  "platform-tools" "platforms;android-37.0" "build-tools;36.0.0"
```

Newer command-line tools releases are listed at
<https://developer.android.com/studio#command-tools>; any recent one works.

Then, from this directory:

```sh
export JAVA_HOME=~/.local/jdks/jdk-21 ANDROID_HOME=~/Android/Sdk
./gradlew test assembleDebug
```

The first build downloads Gradle and the libraries (a few minutes). The APK
is `app/build/outputs/apk/debug/app-debug.apk`.

## Install it on your phone

Either way works:

- **USB with adb.** On the phone, enable Developer options (tap Build number
  seven times in Settings > About phone), turn on USB debugging, connect the
  cable and accept the prompt. Then:

  ```sh
  ~/Android/Sdk/platform-tools/adb install -r app/build/outputs/apk/debug/app-debug.apk
  ```

- **File transfer.** Copy `app-debug.apk` to the phone (USB, cloud drive or
  email to yourself), open it in the Files app and allow that app to install
  unknown apps when asked.

The APK is signed with this machine's debug key (`~/.android/debug.keystore`).
Updates install over the old version only when built on the same machine;
otherwise uninstall the old version first.

## Set it up

1. On GitHub, create a fine-grained personal access token
   (Settings > Developer settings > Fine-grained tokens):
   - Repository access: only `alutterb/retro-analytics-data`.
   - Permissions: Contents read-only (Metadata read-only is added
     automatically).
   - Pick an expiry you are comfortable with. When it expires, the app shows
     "GitHub rejected the token" and the daily refresh posts one warning.
2. Open the app. Paste the token and keep the prefilled results location
   (`https://api.github.com/repos/alutterb/retro-analytics-data/contents/results/latest.json?ref=main`)
   unless you publish elsewhere.
3. Tap **Test connection**, then **Save**, and allow notifications.

The token is encrypted with a key kept in the phone's Android Keystore and is
only ever sent to `api.github.com` over https. Backups exclude the app's data,
so after moving to a new phone you paste the token again.

## How it updates

The app downloads the results file when you open it (or tap Refresh) and in
the background about once a day when the phone is online. The PC publishes a
new file once a week, so most downloads find nothing new. Thresholds for
rising and crashing are set on the PC in `data/settings.toml`; the app shows
whatever the PC flagged.

## Development

- `app/src/main/kotlin/.../data/`: the results-file model and parser, the
  newly-flagged comparison, list sorting, the GitHub client, the encrypted
  token store, the cache and the refresh path.
- `app/src/main/kotlin/.../work/`: the daily WorkManager job and notifications.
- `app/src/main/kotlin/.../ui/`: the Compose screens.
- `app/src/test/`: JVM unit tests for parsing, newly-flagged detection and
  pending announcements, the location check and list sorting (`./gradlew test`).

The app reads results-file schema version 1 and ignores fields it does not
use. A file with any other version shows "update the app" and keeps the
last good data.
