package io.github.alutterb.retroanalytics.data

import java.io.IOException
import java.net.HttpURLConnection
import java.net.URI
import java.net.URISyntaxException
import java.net.URL

const val DEFAULT_RESULTS_URL =
    "https://api.github.com/repos/alutterb/retro-analytics-data/contents/results/latest.json?ref=main"

/** Why a download or connection test failed, in words for the owner. */
enum class Failure(val message: String, val fixInSettings: Boolean) {
    NO_TOKEN("No GitHub token saved yet", true),
    BAD_LOCATION("The results location must be an https://api.github.com/ address", true),
    BAD_TOKEN("GitHub rejected the token; it may have expired or lack access", true),
    MISSING("No results file at that location, or the token cannot see it", true),
    NO_NETWORK("No connection to GitHub", false),
    SERVER("GitHub returned an error; try again later", false),
    NOT_UNDERSTOOD("The results file could not be read", false),
    UNSUPPORTED("The results file is newer than this app; update the app", false),
}

/** The token is only ever sent to the GitHub API over https. */
fun isAllowedResultsUrl(url: String): Boolean = try {
    val uri = URI(url.trim())
    uri.scheme == "https" && uri.host == "api.github.com" && uri.userInfo == null && uri.port == -1
} catch (e: URISyntaxException) {
    false
}

sealed interface Fetched {
    data class Ok(val text: String) : Fetched
    data class Failed(val failure: Failure) : Fetched
}

/** Downloads the raw results file through the GitHub contents API. */
object GitHubClient {
    fun fetch(url: String, token: String): Fetched {
        val target = url.trim()
        if (!isAllowedResultsUrl(target)) return Fetched.Failed(Failure.BAD_LOCATION)
        val connection = URL(target).openConnection() as HttpURLConnection
        return try {
            connection.connectTimeout = 15_000
            connection.readTimeout = 30_000
            connection.setRequestProperty("Authorization", "Bearer $token")
            connection.setRequestProperty("Accept", "application/vnd.github.raw+json")
            connection.setRequestProperty("X-GitHub-Api-Version", "2022-11-28")
            when (connection.responseCode) {
                200 -> Fetched.Ok(connection.inputStream.bufferedReader().use { it.readText() })
                401, 403 -> Fetched.Failed(Failure.BAD_TOKEN)
                404 -> Fetched.Failed(Failure.MISSING)
                else -> Fetched.Failed(Failure.SERVER)
            }
        } catch (e: IOException) {
            Fetched.Failed(Failure.NO_NETWORK)
        } finally {
            connection.disconnect()
        }
    }
}
