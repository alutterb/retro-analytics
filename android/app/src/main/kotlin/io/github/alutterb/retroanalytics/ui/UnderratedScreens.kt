package io.github.alutterb.retroanalytics.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.unit.dp
import io.github.alutterb.retroanalytics.data.Underrated

/** A row on the Underrated tab: rank and votes show how much evidence is behind it. */
@Composable
fun UnderratedRow(row: Underrated, onClick: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().clickable(onClick = onClick).padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(row.title, style = MaterialTheme.typography.titleMedium)
            Text(row.console, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(
                "#${row.rank} in ${row.console} - ${count(row.votes)} votes",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface,
            )
        }
        val price = row.lastPrice
        if (price == null) {
            Text("No price", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        } else {
            Text(money(price), style = MaterialTheme.typography.titleMedium)
        }
    }
}

/** The evidence behind an underrated game; there is no price history for these. */
@Composable
fun UnderratedDetail(row: Underrated, modifier: Modifier = Modifier) {
    val uriHandler = LocalUriHandler.current
    Column(
        modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(row.title, style = MaterialTheme.typography.headlineSmall)
        Text(priceLine(row), style = MaterialTheme.typography.bodyMedium)
        Fact("Score", "%+.2f".format(row.score))
        Note("standard deviations above what its popularity predicts")
        Fact("Rank in console", "#${row.rank}")
        Fact("IGDB user rating", "%.1f / 100".format(row.rating))
        Fact("Votes", count(row.votes))
        Fact("Popularity", count(row.popularity))
        Note("votes including the parent game's")
        row.url?.let { url ->
            OutlinedButton(onClick = { uriHandler.openUri(url) }) { Text("Open on PriceCharting") }
        }
        row.igdbUrl?.let { url ->
            OutlinedButton(onClick = { uriHandler.openUri(url) }) { Text("Open on IGDB") }
        }
        Note("Ratings on retro games come from few voters and may reflect another platform's version.")
    }
}

private fun priceLine(row: Underrated): String {
    val price = row.lastPrice ?: return "${row.console} - no CIB price"
    val month = row.lastMonth?.let { " (${it.take(7)})" } ?: ""
    return "${row.console} - complete in box ${money(price)}$month"
}

private fun count(value: Int): String = "%,d".format(value)
