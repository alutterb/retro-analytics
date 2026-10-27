package io.github.alutterb.retroanalytics.data

/** Builds results-file JSON for tests; prices are made up. */
object Fixtures {
    fun game(
        slug: String,
        title: String = slug,
        console: String = "nes",
        price: Double = 100.0,
        flags: List<Pair<String, Double>> = emptyList(),
        url: String? = null,
        sixMonth: Double? = null,
        rebound: Double? = null,
    ): String {
        val flagJson = flags.joinToString(",") { (direction, value) ->
            """{"direction":"$direction","signal":"forecast","value_pct":$value,"threshold_pct":10.0,"text":"$direction $value"}"""
        }
        val reboundJson = if (rebound == null) "" else
            ""","rebound":{"direction":"rebound","signal":"change_1m","value_pct":$rebound,"threshold_pct":10.0,"text":"down 12% in 1 month, forecast up $rebound% in 6 months"}"""
        val urlJson = if (url == null) "null" else "\"$url\""
        return """
            {"console":"$console","game":"$slug","title":"$title","url":$urlJson,
             "last_month":"2026-10-01","last_price":$price,
             "changes_pct":{"1m":-12.0,"3m":-20.0,"6m":${sixMonth ?: "null"}},
             "forecast":{"growth_pct":15.0,"months":["2026-11-01","2026-12-01"],"prices":[101.0,102.0]},
             "history":{"months":["2026-09-01","2026-10-01"],"prices":[110.0,$price]},
             "flags":[$flagJson]$reboundJson}
        """.trimIndent()
    }

    fun results(vararg games: String, version: Int = 1): String = """
        {"schema_version":$version,"generated":"2026-10-25","horizon_months":6,
         "thresholds":{"rising":{"forecast_pct":10.0},"crashing":{"forecast_pct":-10.0}},
         "games":[${games.joinToString(",")}]}
    """.trimIndent()

    fun parsed(vararg games: String): Results = (parseResults(results(*games)) as ParseResult.Ok).results
}
