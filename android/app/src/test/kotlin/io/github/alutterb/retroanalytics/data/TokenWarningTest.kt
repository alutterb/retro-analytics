package io.github.alutterb.retroanalytics.data

import org.junit.Assert.assertEquals
import org.junit.Test

class TokenWarningTest {
    private val updated = Refreshed.Updated(Fixtures.parsed(), emptyList())

    @Test
    fun firstRejectionWarns() {
        assertEquals(TokenWarning.POST, tokenWarning(Refreshed.Failed(Failure.BAD_TOKEN), alreadyWarned = false))
    }

    @Test
    fun laterRejectionsStayQuiet() {
        assertEquals(TokenWarning.NONE, tokenWarning(Refreshed.Failed(Failure.BAD_TOKEN), alreadyWarned = true))
    }

    @Test
    fun otherFailuresNeitherWarnNorReset() {
        for (failure in listOf(Failure.NO_NETWORK, Failure.SERVER, Failure.MISSING)) {
            assertEquals(TokenWarning.NONE, tokenWarning(Refreshed.Failed(failure), alreadyWarned = true))
            assertEquals(TokenWarning.NONE, tokenWarning(Refreshed.Failed(failure), alreadyWarned = false))
        }
    }

    @Test
    fun successResetsOnlyAfterAWarning() {
        assertEquals(TokenWarning.RESET, tokenWarning(updated, alreadyWarned = true))
        assertEquals(TokenWarning.NONE, tokenWarning(updated, alreadyWarned = false))
    }
}
