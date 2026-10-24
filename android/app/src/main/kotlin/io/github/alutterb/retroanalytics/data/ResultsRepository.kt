package io.github.alutterb.retroanalytics.data

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.IOException
import java.security.GeneralSecurityException
import java.security.ProviderException

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
    fun newKeys(): Set<FlagKey> = store.newlyFlaggedKeys()
    fun pendingKeys(): Set<FlagKey> = store.pendingKeys()

    fun setPendingKeys(keys: Set<FlagKey>) {
        try {
            store.setPendingKeys(keys)
        } catch (e: IOException) {
            // The flags stay marked "New" in the lists; only the notification is lost.
        }
    }
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
     * Saves the settings. A new location starts with an empty cache, so its
     * first download notifies nothing. If the token cannot be stored, nothing changes.
     */
    suspend fun save(url: String, token: String?): SaveResult = lock.withLock {
        withContext(Dispatchers.IO) {
            if (!token.isNullOrBlank()) {
                try {
                    settings.saveToken(token)
                } catch (e: GeneralSecurityException) {
                    return@withContext SaveResult.TokenNotStored
                } catch (e: ProviderException) {
                    return@withContext SaveResult.TokenNotStored
                }
                settings.tokenWarningShown = false
            }
            val locationChanged = isNewLocation(settings.resultsUrl, url)
            if (locationChanged) store.clear()
            settings.resultsUrl = url
            SaveResult.Saved(locationChanged)
        }
    }

    private companion object {
        val lock = Mutex()
    }
}

sealed interface SaveResult {
    data class Saved(val locationChanged: Boolean) : SaveResult
    data object TokenNotStored : SaveResult
}
