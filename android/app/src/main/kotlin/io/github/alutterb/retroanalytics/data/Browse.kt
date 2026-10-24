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

fun consoles(results: Results): List<String> = results.games.map { it.console }.distinct().sorted()
