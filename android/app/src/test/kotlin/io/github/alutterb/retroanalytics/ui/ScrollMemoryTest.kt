package io.github.alutterb.retroanalytics.ui

import org.junit.Test
import org.junit.Assert.assertEquals

class ScrollMemoryTest {
    private val keys = listOf("a", "b", "c", "d", "e")

    @Test
    fun startsAtTop() {
        assertEquals(0 to 0, ScrollMemory().restore(Tab.RISING, keys))
    }

    @Test
    fun scrollOpenGameAndBackKeepsPlace() {
        val memory = ScrollMemory()
        memory.save(Tab.RISING, "d", 37)
        // Detail screen: list leaves composition, then comes back and asks again.
        assertEquals(3 to 37, memory.restore(Tab.RISING, keys))
    }

    @Test
    fun eachTabKeepsItsOwnPlace() {
        val memory = ScrollMemory()
        memory.save(Tab.RISING, "c", 5)
        memory.save(Tab.CRASHING, "b", 9)
        assertEquals(2 to 5, memory.restore(Tab.RISING, keys))
        assertEquals(1 to 9, memory.restore(Tab.CRASHING, keys))
        assertEquals(0 to 0, memory.restore(Tab.REBOUND, keys))
    }

    @Test
    fun underratedTabKeepsItsOwnPlace() {
        val memory = ScrollMemory()
        memory.save(Tab.RISING, "c", 5)
        memory.save(Tab.UNDERRATED, "e", 30)
        assertEquals(4 to 30, memory.restore(Tab.UNDERRATED, keys))
        assertEquals(2 to 5, memory.restore(Tab.RISING, keys))
    }

    @Test
    fun refreshThatMovesGameKeepsPlace() {
        val memory = ScrollMemory()
        memory.save(Tab.RISING, "c", 5)
        assertEquals(3 to 5, memory.restore(Tab.RISING, listOf("x", "a", "b", "c")))
    }

    @Test
    fun goneGameResetsToTop() {
        val memory = ScrollMemory()
        memory.save(Tab.RISING, "c", 5)
        assertEquals(0 to 0, memory.restore(Tab.RISING, listOf("a", "b")))
    }

    @Test
    fun clearForgetsEveryTab() {
        val memory = ScrollMemory()
        memory.save(Tab.RISING, "c", 5)
        memory.clear()
        assertEquals(0 to 0, memory.restore(Tab.RISING, keys))
    }
}
