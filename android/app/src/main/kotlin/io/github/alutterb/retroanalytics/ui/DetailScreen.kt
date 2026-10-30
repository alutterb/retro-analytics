package io.github.alutterb.retroanalytics.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.unit.dp
import io.github.alutterb.retroanalytics.data.Game

@Composable
fun DetailScreen(game: Game, modifier: Modifier = Modifier) {
    val uriHandler = LocalUriHandler.current
    Column(
        modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(game.title, style = MaterialTheme.typography.headlineSmall)
        Text(
            "${game.console} - complete in box ${money(game.lastPrice)} (${game.lastMonth.take(7)})",
            style = MaterialTheme.typography.bodyMedium,
        )
        PriceChart(game, Modifier.fillMaxWidth())
        Text("Why it is flagged", style = MaterialTheme.typography.titleMedium)
        game.allFlags.forEach { flag ->
            Text("- ${flag.text}", color = directionColor(flag.direction), style = MaterialTheme.typography.bodyMedium)
        }
        Text("Price changes", style = MaterialTheme.typography.titleMedium)
        Fact("Last month", pct(game.changesPct.oneMonth))
        Fact("Last 3 months", pct(game.changesPct.threeMonths))
        Fact("Last 6 months", pct(game.changesPct.sixMonths))
        Fact("Forecast", pct(game.forecast.growthPct))
        game.url?.let { url ->
            OutlinedButton(onClick = { uriHandler.openUri(url) }) { Text("Open on PriceCharting") }
        }
        Note("Forecasts are statistical extrapolations of past prices, not advice.")
    }
}

@Composable
fun Note(text: String) {
    Text(text, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
}

@Composable
fun Fact(label: String, value: String) {
    Row(Modifier.fillMaxWidth()) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.weight(1f))
        Text(value, style = MaterialTheme.typography.bodyMedium)
    }
}

private fun pct(value: Double?): String = value?.let { "%+.1f%%".format(it) } ?: "not enough history"
