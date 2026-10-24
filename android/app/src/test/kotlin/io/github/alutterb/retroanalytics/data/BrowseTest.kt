package io.github.alutterb.retroanalytics.data

import io.github.alutterb.retroanalytics.data.Fixtures.game
import io.github.alutterb.retroanalytics.data.Fixtures.parsed
import org.junit.Assert.assertEquals
import org.junit.Test

class BrowseTest {
    private val results = parsed(
        game("zelda", "Zelda", console = "nes", price = 80.0, flags = listOf("rising" to 12.0)),
        game("chrono", "Chrono Trigger", console = "snes", price = 300.0, flags = listOf("rising" to 30.0, "crashing" to -11.0)),
        game("sonic", "Sonic", console = "sega-genesis", price = 25.0, flags = listOf("crashing" to -12.0, "crashing" to -40.0)),
        game("metroid", "Metroid", console = "nes", price = 50.0, flags = listOf("rising" to 20.0)),
    )

    private fun slugs(direction: Direction, console: String? = null, sort: SortOrder = SortOrder.MOVE) =
        browse(results, direction, console, sort).map { it.game }

    @Test
    fun gameFlaggedBothWaysIsInBothTabs() {
        assertEquals(listOf("chrono", "metroid", "zelda"), slugs(Direction.RISING))
        assertEquals(listOf("sonic", "chrono"), slugs(Direction.CRASHING))
    }

    @Test
    fun defaultSortIsLargestMoveFirst() {
        assertEquals(SortOrder.MOVE, SortOrder.entries.first())
        assertEquals(listOf("chrono", "metroid", "zelda"), slugs(Direction.RISING))
    }

    @Test
    fun headlineIsTheLargestFlagInThatDirection() {
        val sonic = results.games.first { it.game == "sonic" }
        assertEquals(-40.0, sonic.headline(Direction.CRASHING)!!.valuePct, 0.0)
        assertEquals(null, sonic.headline(Direction.RISING))
    }

    @Test
    fun sortByPriceIsHighestFirst() {
        assertEquals(listOf("chrono", "zelda", "metroid"), slugs(Direction.RISING, sort = SortOrder.PRICE))
        assertEquals(listOf("chrono", "sonic"), slugs(Direction.CRASHING, sort = SortOrder.PRICE))
    }

    @Test
    fun sortByTitleIsAlphabetical() {
        assertEquals(listOf("chrono", "metroid", "zelda"), slugs(Direction.RISING, sort = SortOrder.TITLE))
        assertEquals(listOf("metroid", "zelda"), slugs(Direction.RISING, "nes", SortOrder.TITLE))
    }

    @Test
    fun consoleFilterKeepsOnlyThatConsole() {
        assertEquals(listOf("metroid", "zelda"), slugs(Direction.RISING, console = "nes"))
        assertEquals(emptyList<String>(), slugs(Direction.CRASHING, console = "nes"))
    }

    @Test
    fun consolesAreListedOnceInOrder() {
        assertEquals(listOf("nes", "sega-genesis", "snes"), consoles(results))
    }
}
