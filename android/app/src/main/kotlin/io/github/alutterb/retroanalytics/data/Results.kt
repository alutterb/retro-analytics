package io.github.alutterb.retroanalytics.data

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.decodeFromJsonElement
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonPrimitive

/** The results-file version this app reads; `ra export` on the PC writes it. */
const val SUPPORTED_SCHEMA = 1

/**
 * The weekly results file: only games the PC flagged rising or crashing.
 * A game that recently crashed while its forecast points up also carries a
 * [Game.rebound] flag; files from before that field simply lack it. The
 * [underrated] list is separate from the flagged games and missing from older
 * files, which then parse with an empty list.
 *
 * Only the fields a screen or rule reads are modelled. Everything else in the
 * file (thresholds, flag signals, horizon) is ignored like any unknown key, so
 * the PC can change those without breaking the app. Prices are
 * complete-in-box, in US dollars.
 */
@Serializable
data class Results(
    @SerialName("schema_version") val schemaVersion: Int,
    val generated: String,
    val games: List<Game>,
    val underrated: List<Underrated> = emptyList(),
)

@Serializable
data class Game(
    val console: String,
    val game: String,
    val title: String,
    val url: String? = null,
    @SerialName("last_month") val lastMonth: String,
    @SerialName("last_price") val lastPrice: Double,
    @SerialName("changes_pct") val changesPct: Changes,
    val forecast: Forecast,
    val history: Series,
    val flags: List<Flag>,
    val rebound: Flag? = null,
) {
    /** The same slug can exist on several consoles. */
    val key: String get() = "$console/$game"

    /** Every flag the game carries, the rebound included. */
    val allFlags: List<Flag> get() = flags + listOfNotNull(rebound)
}

/**
 * A game IGDB players rate higher than its popularity predicts within its
 * console. [score] is in standard deviations; [popularity] is the raw vote
 * count including the parent game's. The price is null when the game has no
 * ranking this week.
 */
@Serializable
data class Underrated(
    val console: String,
    val game: String,
    val title: String,
    val url: String? = null,
    @SerialName("igdb_url") val igdbUrl: String? = null,
    val score: Double,
    val rank: Int,
    val rating: Double,
    val votes: Int,
    val popularity: Int,
    @SerialName("last_price") val lastPrice: Double? = null,
    @SerialName("last_month") val lastMonth: String? = null,
) {
    val key: String get() = "$console/$game"
}

@Serializable
data class Changes(
    @SerialName("1m") val oneMonth: Double? = null,
    @SerialName("3m") val threeMonths: Double? = null,
    @SerialName("6m") val sixMonths: Double? = null,
)

@Serializable
data class Forecast(
    @SerialName("growth_pct") val growthPct: Double? = null,
    val months: List<String>,
    val prices: List<Double>,
)

@Serializable
data class Series(val months: List<String>, val prices: List<Double>)

@Serializable
data class Flag(
    val direction: Direction,
    @SerialName("value_pct") val valuePct: Double,
    val text: String,
)

@Serializable
enum class Direction(val label: String) {
    @SerialName("rising") RISING("rising"),
    @SerialName("crashing") CRASHING("crashing"),
    @SerialName("rebound") REBOUND("rebound"),
}

sealed interface ParseResult {
    data class Ok(val results: Results) : ParseResult
    data class Unsupported(val version: Int) : ParseResult
    data class Invalid(val reason: String) : ParseResult
}

private val json = Json { ignoreUnknownKeys = true }

fun parseResults(text: String): ParseResult {
    val element = try {
        json.parseToJsonElement(text)
    } catch (e: SerializationException) {
        return ParseResult.Invalid("not valid JSON")
    }
    val version = try {
        (element as? JsonObject)?.get("schema_version")?.jsonPrimitive?.int
    } catch (e: IllegalArgumentException) {
        return ParseResult.Invalid("schema_version is not a number")
    } ?: return ParseResult.Invalid("no schema_version")
    if (version != SUPPORTED_SCHEMA) return ParseResult.Unsupported(version)
    return try {
        ParseResult.Ok(json.decodeFromJsonElement<Results>(element))
    } catch (e: IllegalArgumentException) {
        // SerializationException is an IllegalArgumentException.
        ParseResult.Invalid(e.message?.lineSequence()?.firstOrNull() ?: "unexpected shape")
    }
}
