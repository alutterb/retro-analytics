package io.github.alutterb.retroanalytics.ui

import io.github.alutterb.retroanalytics.data.Direction
import io.github.alutterb.retroanalytics.data.Results
import io.github.alutterb.retroanalytics.data.SortOrder
import io.github.alutterb.retroanalytics.data.browse
import io.github.alutterb.retroanalytics.data.browseUnderrated

/**
 * The list tabs. The flagged tabs carry their [Direction] and share its name, so a
 * notification's tab extra (a direction name) opens the matching tab.
 */
enum class Tab(val label: String, val direction: Direction?) {
    RISING("Rising", Direction.RISING),
    CRASHING("Crashing", Direction.CRASHING),
    REBOUND("Rebound", Direction.REBOUND),
    UNDERRATED("Underrated", null);

    /** The first order sorts by score here rather than by move. */
    fun sortLabel(sort: SortOrder): String =
        if (this == UNDERRATED && sort == SortOrder.MOVE) "Score" else sort.label

    companion object {
        fun named(name: String): Tab? = entries.firstOrNull { it.name == name }
    }
}

/** Whether `tab` lists any game on `console`. */
fun hasRows(results: Results, tab: Tab, console: String): Boolean {
    val direction = tab.direction ?: return browseUnderrated(results, console, SortOrder.MOVE).isNotEmpty()
    return browse(results, direction, console, SortOrder.MOVE).isNotEmpty()
}

/** The console filter after switching to `tab`: back to All when that tab has nothing on the active console. */
fun consoleOnTab(results: Results?, tab: Tab, console: String?): String? =
    if (results == null || console == null || hasRows(results, tab, console)) console else null
