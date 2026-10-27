package io.github.alutterb.retroanalytics.data

import io.github.alutterb.retroanalytics.data.Fixtures.game
import io.github.alutterb.retroanalytics.data.Fixtures.parsed
import org.junit.Assert.assertEquals
import org.junit.Test

class NewlyFlaggedTest {
    private fun pairs(flags: List<NewFlag>) = flags.map { it.game.game to it.direction }

    @Test
    fun firstDownloadFlagsNothingAsNew() {
        // Covers AE1.
        val current = parsed(game("zelda", flags = listOf("rising" to 15.0)))
        assertEquals(emptyList<NewFlag>(), newlyFlagged(null, current))
    }

    @Test
    fun onlyFlagsAbsentLastTimeAreNew() {
        // Covers AE2.
        val previous = parsed(game("chrono", flags = listOf("rising" to 12.0)))
        val current = parsed(
            game("chrono", flags = listOf("rising" to 14.0)),
            game("earthbound", flags = listOf("crashing" to -18.0)),
            game("zelda", flags = listOf("rising" to 11.0)),
        )
        assertEquals(
            listOf("earthbound" to Direction.CRASHING, "zelda" to Direction.RISING),
            pairs(newlyFlagged(previous, current)),
        )
    }

    @Test
    fun aCrashedGameThatStartsReboundingIsNewlyRebound() {
        val previous = parsed(game("vmu", flags = listOf("crashing" to -25.0)))
        val current = parsed(game("vmu", flags = listOf("crashing" to -25.0), rebound = 81.0))
        assertEquals(listOf("vmu" to Direction.REBOUND), pairs(newlyFlagged(previous, current)))
        assertEquals("vmu is newly rebound", summaryText(newlyFlagged(previous, current)))
    }

    @Test
    fun switchingDirectionCountsAsNew() {
        // Covers AE3.
        val previous = parsed(game("zelda", flags = listOf("rising" to 12.0)))
        val current = parsed(game("zelda", flags = listOf("crashing" to -15.0)))
        assertEquals(listOf("zelda" to Direction.CRASHING), pairs(newlyFlagged(previous, current)))
    }

    @Test
    fun sameSlugOnAnotherConsoleIsADifferentGame() {
        val previous = parsed(game("tetris", console = "nes", flags = listOf("rising" to 12.0)))
        val current = parsed(game("tetris", console = "gameboy", flags = listOf("rising" to 12.0)))
        assertEquals(listOf("tetris" to Direction.RISING), pairs(newlyFlagged(previous, current)))
    }

    @Test
    fun twoFlagsInOneDirectionAreOneNewEntry() {
        val current = parsed(game("zelda", flags = listOf("crashing" to -12.0, "crashing" to -20.0)))
        assertEquals(1, newlyFlagged(parsed(), current).size)
    }

    @Test
    fun summaryNamesEveryNewGameWithItsDirection() {
        val current = parsed(
            game("earthbound", "EarthBound", flags = listOf("crashing" to -18.0)),
            game("zelda", "Zelda", flags = listOf("rising" to 11.0)),
        )
        assertEquals(
            "2 newly flagged: EarthBound crashing, Zelda rising",
            summaryText(newlyFlagged(parsed(), current)),
        )
    }

    @Test
    fun summaryForOneGame() {
        val current = parsed(game("zelda", "Zelda", flags = listOf("rising" to 11.0)))
        assertEquals("Zelda is newly rising", summaryText(newlyFlagged(parsed(), current)))
    }
}

class AnnouncementTest {
    private val current = parsed(
        game("earthbound", "EarthBound", flags = listOf("crashing" to -18.0)),
        game("zelda", "Zelda", flags = listOf("rising" to 11.0)),
    )
    private val fresh = newlyFlagged(parsed(), current)

    private fun titles(a: Announcement) = a.announce.map { it.game.title }

    @Test
    fun deniedRefreshKeepsPendingAndPostsNothing() {
        val denied = announcement(emptySet(), current, fresh, canPost = false)
        assertEquals(emptyList<String>(), titles(denied))
        assertEquals(fresh.map { it.id }.toSet(), denied.pending)
    }

    @Test
    fun grantedRefreshPostsOneSummaryNamingPendingAndClearsIt() {
        val denied = announcement(emptySet(), current, fresh, canPost = false)
        val granted = announcement(denied.pending, current, emptyList(), canPost = true)
        assertEquals(
            "2 newly flagged: EarthBound crashing, Zelda rising",
            summaryText(granted.announce),
        )
        assertEquals(emptySet<FlagKey>(), granted.pending)
    }

    @Test
    fun laterIdenticalRefreshPostsNothing() {
        val after = announcement(emptySet(), current, emptyList(), canPost = true)
        assertEquals(emptyList<String>(), titles(after))
        assertEquals(emptySet<FlagKey>(), after.pending)
    }

    @Test
    fun pendingFlagNoLongerFlaggedIsDropped() {
        val later = parsed(game("zelda", "Zelda", flags = listOf("rising" to 11.0)))
        val granted = announcement(fresh.map { it.id }.toSet(), later, emptyList(), canPost = true)
        assertEquals(listOf("Zelda"), titles(granted))
    }
}
