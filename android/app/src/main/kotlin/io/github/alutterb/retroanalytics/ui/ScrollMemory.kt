package io.github.alutterb.retroanalytics.ui

import io.github.alutterb.retroanalytics.data.Direction

/** Where a list was scrolled to: the first visible game and how far its top is scrolled off. */
data class ScrollSpot(val key: String, val offset: Int)

/**
 * Remembers one scroll spot per tab. Spots are stored by game key, not index, so a refresh
 * that reorders or changes the list still lands on the same game.
 */
class ScrollMemory {
    private val spots = mutableMapOf<Direction, ScrollSpot>()

    fun save(tab: Direction, key: String, offset: Int) {
        spots[tab] = ScrollSpot(key, offset)
    }

    /** Index and offset to restore in [keys], or the top when nothing is remembered or the game is gone. */
    fun restore(tab: Direction, keys: List<String>): Pair<Int, Int> {
        val spot = spots[tab] ?: return 0 to 0
        val index = keys.indexOf(spot.key)
        return if (index < 0) 0 to 0 else index to spot.offset
    }

    fun clear() = spots.clear()
}
