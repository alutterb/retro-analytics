package io.github.alutterb.retroanalytics.data

import io.github.alutterb.retroanalytics.data.Fixtures.game
import io.github.alutterb.retroanalytics.data.Fixtures.results
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ResultsSyncTest {
    @get:Rule
    val folder = TemporaryFolder()

    private var served: Fetched = Fetched.Failed(Failure.NO_NETWORK)
    private val store by lazy { ResultsStore(folder.root) }
    private val sync by lazy { ResultsSync(store) { _, _ -> served } }

    private fun serve(vararg games: String) {
        served = Fetched.Ok(results(*games))
    }

    private fun refresh() = sync.refresh("https://api.github.com/x", "token")

    @Test
    fun firstDownloadIsCachedAndNotifiesNothing() {
        serve(game("zelda", flags = listOf("rising" to 15.0)))
        val refreshed = refresh() as Refreshed.Updated
        assertEquals(emptyList<NewFlag>(), refreshed.newFlags)
        assertEquals(listOf("zelda"), store.read()!!.games.map { it.game })
    }

    @Test
    fun aNewFlagInTheNextFileIsReportedAndMarked() {
        serve(game("zelda", flags = listOf("rising" to 15.0)))
        refresh()
        serve(game("zelda", flags = listOf("rising" to 15.0)), game("sonic", console = "sega", flags = listOf("crashing" to -20.0)))
        val refreshed = refresh() as Refreshed.Updated
        assertEquals(listOf("sega/sonic" to Direction.CRASHING), refreshed.newFlags.map { it.id })
        assertEquals(setOf("sega/sonic" to Direction.CRASHING), store.newlyFlaggedKeys())
    }

    @Test
    fun downloadingTheSameFileAgainKeepsTheNewMarkers() {
        serve(game("zelda", flags = listOf("rising" to 15.0)))
        refresh()
        serve(game("zelda", flags = listOf("rising" to 15.0)), game("sonic", flags = listOf("crashing" to -20.0)))
        refresh()
        val again = refresh() as Refreshed.Updated
        assertEquals(emptyList<NewFlag>(), again.newFlags)
        assertEquals(setOf("nes/sonic" to Direction.CRASHING), store.newlyFlaggedKeys())
    }

    @Test
    fun failedDownloadsKeepTheLastGoodFile() {
        // Covers AE5 and the offline case.
        serve(game("zelda", flags = listOf("rising" to 15.0)))
        refresh()
        for (bad in listOf(
            Fetched.Ok(results(game("x"), version = 2)) to Failure.UNSUPPORTED,
            Fetched.Ok("{broken") to Failure.NOT_UNDERSTOOD,
            Fetched.Failed(Failure.BAD_TOKEN) to Failure.BAD_TOKEN,
        )) {
            served = bad.first
            assertEquals(Refreshed.Failed(bad.second), refresh())
            assertEquals(listOf("zelda"), store.read()!!.games.map { it.game })
        }
    }

    @Test
    fun aCacheThatCannotBeWrittenIsAFailureNotACrash() {
        serve(game("zelda", flags = listOf("rising" to 15.0)))
        folder.root.setWritable(false)
        try {
            assertEquals(Refreshed.Failed(Failure.STORAGE), refresh())
        } finally {
            folder.root.setWritable(true)
        }
    }

    @Test
    fun clearingTheCacheMakesTheNextDownloadAFirstDownload() {
        serve(game("zelda", flags = listOf("rising" to 15.0)))
        refresh()
        store.clear()
        serve(game("zelda", flags = listOf("rising" to 15.0)), game("sonic", flags = listOf("crashing" to -20.0)))
        assertTrue((refresh() as Refreshed.Updated).newFlags.isEmpty())
    }

    @Test
    fun testDoesNotTouchTheCache() {
        serve(game("zelda", flags = listOf("rising" to 15.0)))
        assertTrue(sync.test("https://api.github.com/x", "token") is Refreshed.Updated)
        assertEquals(null, store.read())
    }

    @Test
    fun onlyAChangedLocationCountsAsNew() {
        assertTrue(isNewLocation("https://api.github.com/a", "https://api.github.com/b"))
        assertEquals(false, isNewLocation("https://api.github.com/a", "  https://api.github.com/a "))
    }
}
