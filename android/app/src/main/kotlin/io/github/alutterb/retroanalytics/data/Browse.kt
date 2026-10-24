package io.github.alutterb.retroanalytics.data

import kotlin.math.abs

/** List orders; the first is the default. */
enum class SortOrder(val label: String) {
    MOVE("Biggest move"),
    PRICE("Price"),
    TITLE("Title"),
}

/** The flag shown on a game's row: its largest move in that direction. */
fun Game.headline(direction: Direction): Flag? =
    flags.filter { it.direction == direction }.maxByOrNull { abs(it.valuePct) }

/** Games flagged in `direction`, optionally on one console, in `sort` order. */
fun browse(results: Results, direction: Direction, console: String?, sort: SortOrder): List<Game> {
    val games = results.games.filter { game ->
        game.headline(direction) != null && (console == null || game.console == console)
    }
    return when (sort) {
        SortOrder.MOVE -> games.sortedByDescending { abs(it.headline(direction)!!.valuePct) }
        SortOrder.PRICE -> games.sortedByDescending { it.lastPrice }
        SortOrder.TITLE -> games.sortedBy { it.title.lowercase() }
    }
}

fun consoles(results: Results): List<String> = results.games.map { it.console }.distinct().sorted()
