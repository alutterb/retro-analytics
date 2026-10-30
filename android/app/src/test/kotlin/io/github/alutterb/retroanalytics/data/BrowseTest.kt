package io.github.alutterb.retroanalytics.data

import io.github.alutterb.retroanalytics.data.Fixtures.game
import io.github.alutterb.retroanalytics.data.Fixtures.parsed
import io.github.alutterb.retroanalytics.data.Fixtures.underrated
import org.junit.Assert.assertEquals
import org.junit.Test

class BrowseTest {
    private val results = parsed(
        game("zelda", "Zelda", console = "nes", price = 80.0, flags = listOf("rising" to 12.0)),
        game("chrono", "Chrono Trigger", console = "snes", price = 300.0, flags = listOf("rising" to 30.0, "crashing" to -11.0)),
        game("sonic", "Sonic", console = "sega-genesis", price = 25.0, flags = listOf("crashing" to -12.0, "crashing" to -40.0)),
        game("metroid", "Metroid", console = "nes", price = 50.0, flags = listOf("rising" to 20.0)),
        game("vmu", "VMU", console = "dreamcast", price = 40.0, flags = listOf("crashing" to -25.0), rebound = 81.0),
        game("sapphire", "Pokemon Sapphire", console = "gba", price = 90.0, flags = listOf("crashing" to -27.0), rebound = 30.0),
    )

    private fun slugs(direction: Direction, console: String? = null, sort: SortOrder = SortOrder.MOVE) =
        browse(results, direction, console, sort).map { it.game }

    @Test
    fun gameFlaggedBothWaysIsInBothTabs() {
        assertEquals(listOf("chrono", "metroid", "zelda"), slugs(Direction.RISING))
        assertEquals(listOf("sonic", "sapphire", "vmu", "chrono"), slugs(Direction.CRASHING))
    }

    @Test
    fun reboundTabListsOnlyReboundGamesBiggestForecastFirst() {
        assertEquals(listOf("vmu", "sapphire"), slugs(Direction.REBOUND))
        assertEquals(listOf("sapphire"), slugs(Direction.REBOUND, console = "gba"))
        assertEquals(listOf("sapphire", "vmu"), slugs(Direction.REBOUND, sort = SortOrder.PRICE))
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
        assertEquals(listOf("chrono", "sapphire", "vmu", "sonic"), slugs(Direction.CRASHING, sort = SortOrder.PRICE))
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
        assertEquals(listOf("dreamcast", "gba", "nes", "sega-genesis", "snes"), consoles(results))
    }

    private val withUnderrated = parsed(
        game("zelda", console = "nes", flags = listOf("rising" to 12.0)),
        underrated = listOf(
            underrated("kirby", "Kirby", console = "nes", score = 1.2, price = 20.0),
            underrated("gunstar", "Gunstar Heroes", console = "sega-genesis", score = 2.4, price = null),
            underrated("crystalis", "Crystalis", console = "nes", score = 2.1, price = 45.0),
            underrated("ristar", "Ristar", console = "sega-genesis", score = -0.3, price = 60.0),
        ),
    )

    private fun underratedSlugs(console: String? = null, sort: SortOrder = SortOrder.MOVE) =
        browseUnderrated(withUnderrated, console, sort).map { it.game }

    @Test
    fun underratedDefaultsToHighestScoreFirst() {
        assertEquals(listOf("gunstar", "crystalis", "kirby", "ristar"), underratedSlugs())
    }

    @Test
    fun underratedConsoleFilterKeepsOnlyThatConsole() {
        assertEquals(listOf("crystalis", "kirby"), underratedSlugs(console = "nes"))
        assertEquals(emptyList<String>(), underratedSlugs(console = "snes"))
    }

    @Test
    fun underratedByPriceIsHighestFirstWithMissingPricesLast() {
        assertEquals(listOf("ristar", "crystalis", "kirby", "gunstar"), underratedSlugs(sort = SortOrder.PRICE))
        assertEquals(listOf("crystalis", "gunstar", "kirby", "ristar"), underratedSlugs(sort = SortOrder.TITLE))
    }

    @Test
    fun underratedConsolesComeFromTheUnderratedList() {
        assertEquals(listOf("nes"), consoles(withUnderrated))
        assertEquals(listOf("nes", "sega-genesis"), consoles(withUnderrated, underrated = true))
    }
}
