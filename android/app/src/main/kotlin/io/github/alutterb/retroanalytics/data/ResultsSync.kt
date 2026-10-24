package io.github.alutterb.retroanalytics.data

import java.io.IOException

sealed interface Refreshed {
    data class Updated(val results: Results, val newFlags: List<NewFlag>) : Refreshed
    data class Failed(val failure: Failure) : Refreshed
}

/**
 * Download, parse and cache, without Android types so it runs in JVM tests.
 *
 * The cache is replaced only by a file that parsed, so a bad download never
 * hides the last good data. A download identical to the cache leaves the
 * "New" markers alone: the PC publishes weekly while the app downloads daily.
 */
class ResultsSync(
    private val store: ResultsStore,
    private val fetch: (url: String, token: String) -> Fetched = GitHubClient::fetch,
) {
    fun refresh(url: String, token: String): Refreshed {
        val (text, results) = when (val downloaded = download(url, token)) {
            is Downloaded.Failed -> return Refreshed.Failed(downloaded.failure)
            is Downloaded.Ok -> downloaded
        }
        return try {
            if (text == store.readText()) return Refreshed.Updated(results, emptyList())
            val newFlags = newlyFlagged(store.read(), results)
            store.write(text, newFlags)
            Refreshed.Updated(results, newFlags)
        } catch (e: IOException) {
            Refreshed.Failed(Failure.STORAGE)
        } catch (e: IllegalStateException) {
            Refreshed.Failed(Failure.STORAGE)
        }
    }

    /** Tries a location and token without touching the cache. */
    fun test(url: String, token: String): Refreshed = when (val downloaded = download(url, token)) {
        is Downloaded.Failed -> Refreshed.Failed(downloaded.failure)
        is Downloaded.Ok -> Refreshed.Updated(downloaded.results, emptyList())
    }

    private sealed interface Downloaded {
        data class Ok(val text: String, val results: Results) : Downloaded
        data class Failed(val failure: Failure) : Downloaded
    }

    private fun download(url: String, token: String): Downloaded =
        when (val fetched = fetch(url, token)) {
            is Fetched.Failed -> Downloaded.Failed(fetched.failure)
            is Fetched.Ok -> when (val parsed = parseResults(fetched.text)) {
                is ParseResult.Ok -> Downloaded.Ok(fetched.text, parsed.results)
                is ParseResult.Unsupported -> Downloaded.Failed(Failure.UNSUPPORTED)
                is ParseResult.Invalid -> Downloaded.Failed(Failure.NOT_UNDERSTOOD)
            }
        }
}

fun isNewLocation(saved: String, entered: String): Boolean = entered.trim() != saved.trim()

enum class TokenWarning { POST, RESET, NONE }

/** The one "token rejected" warning: posted once per run of rejections, re-armed by a success. */
fun tokenWarning(refreshed: Refreshed, alreadyWarned: Boolean): TokenWarning = when {
    refreshed is Refreshed.Updated && alreadyWarned -> TokenWarning.RESET
    refreshed is Refreshed.Failed && refreshed.failure == Failure.BAD_TOKEN && !alreadyWarned -> TokenWarning.POST
    else -> TokenWarning.NONE
}
