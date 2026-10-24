package io.github.alutterb.retroanalytics.data

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LocationTest {
    @Test
    fun defaultLocationIsAllowed() {
        assertTrue(isAllowedResultsUrl(DEFAULT_RESULTS_URL))
    }

    @Test
    fun onlyHttpsGitHubApiReceivesTheToken() {
        assertTrue(isAllowedResultsUrl("https://api.github.com/repos/me/data/contents/out.json?ref=dev"))
        assertFalse(isAllowedResultsUrl("http://api.github.com/repos/me/data/contents/out.json"))
        assertFalse(isAllowedResultsUrl("https://api.github.com.evil.example/repos/me/data/contents/out.json"))
        assertFalse(isAllowedResultsUrl("https://evil.example/?https://api.github.com/"))
        assertFalse(isAllowedResultsUrl("https://user@api.github.com/repos/me/data/contents/out.json"))
        assertFalse(isAllowedResultsUrl("not a url"))
    }
}
