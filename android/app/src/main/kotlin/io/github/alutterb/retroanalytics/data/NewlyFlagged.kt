package io.github.alutterb.retroanalytics.data

typealias FlagKey = Pair<String, Direction>

data class NewFlag(val game: Game, val direction: Direction) {
    val id: FlagKey get() = game.key to direction
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

/** What to tell the owner now, and which flags still wait for a notification that could be posted. */
data class Announcement(val announce: List<NewFlag>, val pending: Set<FlagKey>)

/**
 * New flags plus earlier ones that could not be announced, still flagged in
 * [current], as one summary. While notifications are not permitted nothing is
 * announced and everything waits; once permitted, the wait ends.
 */
fun announcement(pending: Set<FlagKey>, current: Results, newFlags: List<NewFlag>, canPost: Boolean): Announcement {
    val wanted = pending + newFlags.map { it.id }
    val due = current.games.flatMap { game ->
        game.flags.map { it.direction }.distinct().map { NewFlag(game, it) }
    }.filter { it.id in wanted }
    return if (canPost) Announcement(due, emptySet()) else Announcement(emptyList(), due.map { it.id }.toSet())
}
