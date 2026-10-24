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
 * The one refresh path, used when the app opens and by the daily worker. The
 * cache is replaced only by a file that parsed, so a bad download never hides
 * the last good data.
 */
class ResultsRepository(context: Context) {
    private val settings = TokenStore(context.applicationContext)
    private val store = ResultsStore(context.applicationContext)

    val hasToken: Boolean get() = settings.hasToken
    val resultsUrl: String get() = settings.resultsUrl
    fun cached(): Results? = store.read()
    fun newKeys(): Set<Pair<String, Direction>> = store.newlyFlaggedKeys()
    fun savedToken(): String? = settings.token()

    var rejectedTokenNotified: Boolean
        get() = settings.tokenWarningShown
        set(value) {
            settings.tokenWarningShown = value
        }

    suspend fun refresh(): Refreshed = lock.withLock {
        withContext(Dispatchers.IO) {
            val token = settings.token() ?: return@withContext Refreshed.Failed(Failure.NO_TOKEN)
            when (val downloaded = download(settings.resultsUrl, token)) {
                is Downloaded.Failed -> Refreshed.Failed(downloaded.failure)
                is Downloaded.Ok -> {
                    val newFlags = newlyFlagged(store.read(), downloaded.results)
                    store.write(downloaded.text, newFlags)
                    Refreshed.Updated(downloaded.results, newFlags)
                }
            }
        }
    }

    /** Tries a location and token without saving them. */
    suspend fun test(url: String, token: String): Refreshed = withContext(Dispatchers.IO) {
        when (val downloaded = download(url, token)) {
            is Downloaded.Failed -> Refreshed.Failed(downloaded.failure)
            is Downloaded.Ok -> Refreshed.Updated(downloaded.results, emptyList())
        }
    }

    /**
     * Saves the settings and returns whether the location changed. A new
     * location starts with an empty cache, so its first download notifies nothing.
     */
    fun save(url: String, token: String?): Boolean {
        val locationChanged = url.trim() != settings.resultsUrl
        if (locationChanged) store.clear()
        settings.resultsUrl = url
        if (!token.isNullOrBlank()) {
            settings.saveToken(token)
            settings.tokenWarningShown = false
        }
        return locationChanged
    }

    private sealed interface Downloaded {
        data class Ok(val text: String, val results: Results) : Downloaded
        data class Failed(val failure: Failure) : Downloaded
    }

    private fun download(url: String, token: String): Downloaded =
        when (val fetched = GitHubClient.fetch(url, token)) {
            is Fetched.Failed -> Downloaded.Failed(fetched.failure)
            is Fetched.Ok -> when (val parsed = parseResults(fetched.text)) {
                is ParseResult.Ok -> Downloaded.Ok(fetched.text, parsed.results)
                is ParseResult.Unsupported -> Downloaded.Failed(Failure.UNSUPPORTED)
                is ParseResult.Invalid -> Downloaded.Failed(Failure.NOT_UNDERSTOOD)
            }
        }

    private companion object {
        val lock = Mutex()
    }
}
