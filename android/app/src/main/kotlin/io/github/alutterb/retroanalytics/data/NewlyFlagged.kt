package io.github.alutterb.retroanalytics.data

data class NewFlag(val game: Game, val direction: Direction) {
    val id: Pair<String, Direction> get() = game.key to direction
}

/**
 * Games flagged in a direction now that were not flagged that way last time.
 *
 * With no previous download nothing is new, so the first download after
 * install (or after changing the results location) notifies nothing.
 */
fun newlyFlagged(previous: Results?, current: Results): List<NewFlag> {
    if (previous == null) return emptyList()
    val before = previous.games.flatMap { game -> game.flags.map { game.key to it.direction } }.toSet()
    return current.games.flatMap { game ->
        game.flags.map { it.direction }.distinct()
            .filter { (game.key to it) !in before }
            .map { NewFlag(game, it) }
    }
}

fun summaryText(newFlags: List<NewFlag>): String {
    val single = newFlags.singleOrNull()
    if (single != null) return "${single.game.title} is newly ${single.direction.label}"
    val names = newFlags.joinToString(", ") { "${it.game.title} ${it.direction.label}" }
    return "${newFlags.size} newly flagged: $names"
}
