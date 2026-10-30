package io.github.alutterb.retroanalytics.ui

import io.github.alutterb.retroanalytics.data.Direction
import io.github.alutterb.retroanalytics.data.Fixtures.game
import io.github.alutterb.retroanalytics.data.Fixtures.parsed
import io.github.alutterb.retroanalytics.data.Fixtures.underrated
import io.github.alutterb.retroanalytics.data.SortOrder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class TabTest {
    private val results = parsed(
        game("zelda", console = "nes", flags = listOf("rising" to 12.0)),
        game("sonic", console = "sega-genesis", flags = listOf("crashing" to -20.0)),
        underrated = listOf(underrated("gunstar", console = "sega-genesis")),
    )

    @Test
    fun everyDirectionNameOpensItsTab() {
        Direction.entries.forEach { direction ->
            assertEquals(direction, Tab.named(direction.name)!!.direction)
        }
        assertEquals(Tab.UNDERRATED, Tab.named("UNDERRATED"))
        assertNull(Tab.named("rising"))
    }

    @Test
    fun firstSortReadsScoreOnlyOnUnderrated() {
        assertEquals("Score", Tab.UNDERRATED.sortLabel(SortOrder.MOVE))
        assertEquals("Price", Tab.UNDERRATED.sortLabel(SortOrder.PRICE))
        assertEquals("Biggest move", Tab.RISING.sortLabel(SortOrder.MOVE))
    }

    @Test
    fun consoleResetsToAllOnATabWithoutIt() {
        assertNull(consoleOnTab(results, Tab.UNDERRATED, "nes"))
        assertNull(consoleOnTab(results, Tab.CRASHING, "nes"))
        assertNull(consoleOnTab(results, Tab.RISING, "sega-genesis"))
    }

    @Test
    fun consoleStaysOnATabWithIt() {
        assertEquals("sega-genesis", consoleOnTab(results, Tab.UNDERRATED, "sega-genesis"))
        assertEquals("sega-genesis", consoleOnTab(results, Tab.CRASHING, "sega-genesis"))
        assertEquals("nes", consoleOnTab(results, Tab.RISING, "nes"))
    }

    @Test
    fun allStaysAllAndNoResultsKeepsTheFilter() {
        assertNull(consoleOnTab(results, Tab.UNDERRATED, null))
        assertEquals("nes", consoleOnTab(null, Tab.UNDERRATED, "nes"))
    }
}
