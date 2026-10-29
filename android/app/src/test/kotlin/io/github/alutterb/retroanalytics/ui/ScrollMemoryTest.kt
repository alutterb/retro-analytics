package io.github.alutterb.retroanalytics.ui

import io.github.alutterb.retroanalytics.data.Direction
import org.junit.Test
import org.junit.Assert.assertEquals

class ScrollMemoryTest {
    private val keys = listOf("a", "b", "c", "d", "e")

    @Test
    fun startsAtTop() {
        assertEquals(0 to 0, ScrollMemory().restore(Direction.RISING, keys))
    }

    @Test
    fun scrollOpenGameAndBackKeepsPlace() {
        val memory = ScrollMemory()
        memory.save(Direction.RISING, "d", 37)
        // Detail screen: list leaves composition, then comes back and asks again.
        assertEquals(3 to 37, memory.restore(Direction.RISING, keys))
    }

    @Test
    fun eachTabKeepsItsOwnPlace() {
        val memory = ScrollMemory()
        memory.save(Direction.RISING, "c", 5)
        memory.save(Direction.CRASHING, "b", 9)
        assertEquals(2 to 5, memory.restore(Direction.RISING, keys))
        assertEquals(1 to 9, memory.restore(Direction.CRASHING, keys))
        assertEquals(0 to 0, memory.restore(Direction.REBOUND, keys))
    }

    @Test
    fun refreshThatMovesGameKeepsPlace() {
        val memory = ScrollMemory()
        memory.save(Direction.RISING, "c", 5)
        assertEquals(3 to 5, memory.restore(Direction.RISING, listOf("x", "a", "b", "c")))
    }

    @Test
    fun goneGameResetsToTop() {
        val memory = ScrollMemory()
        memory.save(Direction.RISING, "c", 5)
        assertEquals(0 to 0, memory.restore(Direction.RISING, listOf("a", "b")))
    }

    @Test
    fun clearForgetsEveryTab() {
        val memory = ScrollMemory()
        memory.save(Direction.RISING, "c", 5)
        memory.clear()
        assertEquals(0 to 0, memory.restore(Direction.RISING, keys))
    }
}
