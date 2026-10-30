package io.github.alutterb.retroanalytics.data

import io.github.alutterb.retroanalytics.data.Fixtures.game
import io.github.alutterb.retroanalytics.data.Fixtures.results
import io.github.alutterb.retroanalytics.data.Fixtures.underrated
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ResultsTest {
    @Test
    fun parsesAValidFile() {
        val text = results(
            game("zelda", "The Legend of Zelda", flags = listOf("rising" to 15.0), url = "https://example.com/zelda"),
            game("sonic", console = "sega-genesis", flags = listOf("crashing" to -20.0), sixMonth = -25.0),
        )
        val parsed = parseResults(text)
        assertTrue(parsed is ParseResult.Ok)
        val results = (parsed as ParseResult.Ok).results
        assertEquals("2026-10-25", results.generated)
        assertEquals(2, results.games.size)
        val zelda = results.games[0]
        assertEquals("The Legend of Zelda", zelda.title)
        assertEquals("https://example.com/zelda", zelda.url)
        assertEquals(100.0, zelda.lastPrice, 0.0)
        assertEquals(-12.0, zelda.changesPct.oneMonth!!, 0.0)
        assertNull(zelda.changesPct.sixMonths)
        assertEquals(listOf(101.0, 102.0), zelda.forecast.prices)
        assertEquals(Direction.RISING, zelda.flags.single().direction)
        assertEquals(-25.0, results.games[1].changesPct.sixMonths!!, 0.0)
    }

    @Test
    fun reboundIsAnOptionalFlagOnTheGame() {
        val text = results(
            game("vmu", console = "dreamcast", flags = listOf("crashing" to -25.0), rebound = 81.0),
            game("sonic", flags = listOf("crashing" to -20.0)),
        )
        val (vmu, sonic) = (parseResults(text) as ParseResult.Ok).results.games
        assertEquals(Direction.REBOUND, vmu.rebound!!.direction)
        assertEquals(81.0, vmu.rebound!!.valuePct, 0.0)
        assertEquals(listOf(Direction.CRASHING), vmu.flags.map { it.direction })
        assertEquals(listOf(Direction.CRASHING, Direction.REBOUND), vmu.allFlags.map { it.direction })
        assertNull(sonic.rebound)
        assertEquals(listOf(Direction.CRASHING), sonic.allFlags.map { it.direction })
    }

    @Test
    fun fileWithoutUnderratedListParsesWithAnEmptyOne() {
        val results = (parseResults(results(game("zelda", flags = listOf("rising" to 15.0)))) as ParseResult.Ok).results
        assertEquals(listOf("zelda"), results.games.map { it.game })
        assertEquals(emptyList<Underrated>(), results.underrated)
    }

    @Test
    fun underratedRowsParseEveryField() {
        val text = results(
            game("zelda"),
            underrated = listOf(
                underrated("bomberman", "Bomberman", console = "gameboy", score = 2.37, rank = 1),
                underrated("windjammers", console = "neo-geo-aes", score = -0.4, rank = 3, price = null),
            ),
        )
        val (bomberman, windjammers) = (parseResults(text) as ParseResult.Ok).results.underrated
        assertEquals(
            Underrated(
                console = "gameboy", game = "bomberman", title = "Bomberman",
                url = "https://example.com/bomberman", igdbUrl = "https://igdb.example/bomberman",
                score = 2.37, rank = 1, rating = 86.6, votes = 42, popularity = 120,
                lastPrice = 30.0, lastMonth = "2026-10-01",
            ),
            bomberman,
        )
        assertEquals("gameboy/bomberman", bomberman.key)
        assertEquals(-0.4, windjammers.score, 0.0)
        assertNull(windjammers.lastPrice)
        assertNull(windjammers.lastMonth)
    }

    @Test
    fun nullableFieldsMayBeNullOrAbsent() {
        val text = results(game("zelda").replace(""""growth_pct":15.0,""", ""))
        val zelda = (parseResults(text) as ParseResult.Ok).results.games.single()
        assertNull(zelda.url)
        assertNull(zelda.forecast.growthPct)
    }

    @Test
    fun unknownFieldsAreIgnored() {
        val text = results(game("zelda").replace(""""title":"zelda",""", """"title":"zelda","rarity":"high",""")).replace(
            """"horizon_months":6,""", """"horizon_months":6,"source":"pc",""",
        )
        assertTrue(parseResults(text) is ParseResult.Ok)
    }

    @Test
    fun anotherSchemaVersionIsUnsupported() {
        assertEquals(ParseResult.Unsupported(2), parseResults(results(game("zelda"), version = 2)))
    }

    @Test
    fun malformedJsonIsInvalid() {
        assertTrue(parseResults("{not json") is ParseResult.Invalid)
    }

    @Test
    fun missingRequiredFieldIsInvalid() {
        val text = results(game("zelda").replace(""""last_price":100.0,""", ""))
        assertTrue(parseResults(text) is ParseResult.Invalid)
    }

    @Test
    fun nonNumericSchemaVersionOrNonObjectRootIsInvalid() {
        assertTrue(parseResults("""{"schema_version":"one","games":[]}""") is ParseResult.Invalid)
        assertTrue(parseResults("[]") is ParseResult.Invalid)
        assertTrue(parseResults("\"x\"") is ParseResult.Invalid)
    }

    @Test
    fun missingSchemaVersionIsInvalid() {
        assertTrue(parseResults("""{"games":[]}""") is ParseResult.Invalid)
    }
}
