package io.github.alutterb.retroanalytics.data

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * The one refresh path, used when the app opens and by the daily worker.
 * Refreshing and saving settings share one lock, so a download of the old
 * location can never land in the cache after the location changed.
 */
class ResultsRepository(context: Context) {
    private val settings = TokenStore(context.applicationContext)
    private val store = ResultsStore(context.applicationContext.filesDir)
    private val sync = ResultsSync(store)

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
            sync.refresh(settings.resultsUrl, token)
        }
    }

    /** Tries a location and token without saving them. */
    suspend fun test(url: String, token: String): Refreshed = withContext(Dispatchers.IO) { sync.test(url, token) }

    /**
     * Saves the settings and returns whether the location changed. A new
     * location starts with an empty cache, so its first download notifies nothing.
     */
    suspend fun save(url: String, token: String?): Boolean = lock.withLock {
        withContext(Dispatchers.IO) {
            val locationChanged = isNewLocation(settings.resultsUrl, url)
            if (locationChanged) store.clear()
            settings.resultsUrl = url
            if (!token.isNullOrBlank()) {
                settings.saveToken(token)
                settings.tokenWarningShown = false
            }
            locationChanged
        }
    }

    private companion object {
        val lock = Mutex()
    }
}
