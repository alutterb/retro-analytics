package io.github.alutterb.retroanalytics.data

import kotlin.math.abs

/** List orders; the first is the default. On the Underrated tab the first orders by score. */
enum class SortOrder(val label: String) {
    MOVE("Biggest move"),
    PRICE("Price"),
    TITLE("Title"),
}

/**
 * The flag shown on a game's row: its largest move in that direction. For a
 * rebound the value is the forecast growth, so the biggest rebound sorts first.
 */
fun Game.headline(direction: Direction): Flag? =
    allFlags.filter { it.direction == direction }.maxByOrNull { abs(it.valuePct) }

/** Games flagged in `direction`, optionally on one console, in `sort` order. */
fun browse(results: Results, direction: Direction, console: String?, sort: SortOrder): List<Game> {
    val moves = results.games
        .filter { console == null || it.console == console }
        .mapNotNull { game -> game.headline(direction)?.let { game to abs(it.valuePct) } }
    val sorted = when (sort) {
        SortOrder.MOVE -> moves.sortedByDescending { it.second }
        SortOrder.PRICE -> moves.sortedByDescending { it.first.lastPrice }
        SortOrder.TITLE -> moves.sortedBy { it.first.title.lowercase() }
    }
    return sorted.map { it.first }
}

/** Underrated games, optionally on one console, in `sort` order; games without a price sort last by price. */
fun browseUnderrated(results: Results, console: String?, sort: SortOrder): List<Underrated> {
    // A repeated key would crash the list, so keep the first row of each game.
    val rows = results.underrated.filter { console == null || it.console == console }.distinctBy { it.key }
    return when (sort) {
        SortOrder.MOVE -> rows.sortedByDescending { it.score }
        SortOrder.PRICE -> rows.sortedWith(compareBy<Underrated> { it.lastPrice == null }.thenByDescending { it.lastPrice })
        SortOrder.TITLE -> rows.sortedBy { it.title.lowercase() }
    }
}

/** Consoles with a flagged game, or with an underrated game when `underrated`. */
fun consoles(results: Results, underrated: Boolean = false): List<String> =
    (if (underrated) results.underrated.map { it.console } else results.games.map { it.console }).distinct().sorted()
