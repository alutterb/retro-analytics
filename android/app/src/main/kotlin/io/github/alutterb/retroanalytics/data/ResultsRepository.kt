package io.github.alutterb.retroanalytics.data

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

sealed interface Refreshed {
    data class Updated(val results: Results, val newFlags: List<NewFlag>) : Refreshed
    data class Failed(val failure: Failure) : Refreshed
}

/**
 * The one refresh path, used when the app opens and by the daily worker:
 * download, parse, compare with the cached file, and replace the cache only
 * when the new file parsed.
 */
class ResultsRepository(context: Context) {
    val settings = TokenStore(context.applicationContext)
    val store = ResultsStore(context.applicationContext)

    suspend fun refresh(): Refreshed = lock.withLock {
        withContext(Dispatchers.IO) {
            val token = settings.token() ?: return@withContext Refreshed.Failed(Failure.NO_TOKEN)
            when (val fetched = GitHubClient.fetch(settings.resultsUrl, token)) {
                is Fetched.Failed -> Refreshed.Failed(fetched.failure)
                is Fetched.Ok -> when (val parsed = parseResults(fetched.text)) {
                    is ParseResult.Unsupported -> Refreshed.Failed(Failure.UNSUPPORTED)
                    is ParseResult.Invalid -> Refreshed.Failed(Failure.NOT_UNDERSTOOD)
                    is ParseResult.Ok -> {
                        val newFlags = newlyFlagged(store.read(), parsed.results)
                        store.write(fetched.text, newFlags)
                        Refreshed.Updated(parsed.results, newFlags)
                    }
                }
            }
        }
    }

    /** Tries a location and token without saving them. */
    suspend fun test(url: String, token: String): Pair<Results?, Failure?> = withContext(Dispatchers.IO) {
        when (val fetched = GitHubClient.fetch(url, token)) {
            is Fetched.Failed -> null to fetched.failure
            is Fetched.Ok -> when (val parsed = parseResults(fetched.text)) {
                is ParseResult.Ok -> parsed.results to null
                is ParseResult.Unsupported -> null to Failure.UNSUPPORTED
                is ParseResult.Invalid -> null to Failure.NOT_UNDERSTOOD
            }
        }
    }

    /** Saves the settings; a new location starts over, so its first download notifies nothing. */
    fun save(url: String, token: String?) {
        if (url.trim() != settings.resultsUrl) store.clear()
        settings.resultsUrl = url
        if (!token.isNullOrBlank()) {
            settings.saveToken(token)
            settings.tokenWarningShown = false
        }
    }

    private companion object {
        val lock = Mutex()
    }
}
