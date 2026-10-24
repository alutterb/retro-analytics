package io.github.alutterb.retroanalytics.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import io.github.alutterb.retroanalytics.data.Game

/** Monthly prices as a solid line, then the forecast as a dashed line that continues it. */
@Composable
fun PriceChart(game: Game, modifier: Modifier = Modifier) {
    val history = game.history.prices
    val forecast = game.forecast.prices
    val values = history + forecast
    if (values.isEmpty()) return
    val low = values.min()
    val high = values.max()
    val line = MaterialTheme.colorScheme.primary
    val grid = MaterialTheme.colorScheme.outlineVariant
    val description = "${history.size} months of prices from ${money(history.firstOrNull() ?: low)} " +
        "to ${money(game.lastPrice)}, then ${forecast.size} forecast months ending at ${money(forecast.lastOrNull() ?: game.lastPrice)}"

    Column(modifier) {
        Text(money(high), style = MaterialTheme.typography.labelSmall)
        Canvas(
            Modifier
                .fillMaxWidth()
                .height(180.dp)
                .semantics { contentDescription = description },
        ) {
            val span = (high - low).takeIf { it > 0 } ?: 1.0
            val step = size.width / (values.size - 1).coerceAtLeast(1)
            fun point(index: Int, value: Double) =
                Offset(index * step, ((high - value) / span * size.height).toFloat())

            listOf(0f, size.height / 2, size.height).forEach { y ->
                drawLine(grid, Offset(0f, y), Offset(size.width, y), strokeWidth = 1f)
            }
            val past = Path().apply {
                history.forEachIndexed { i, v ->
                    val p = point(i, v)
                    if (i == 0) moveTo(p.x, p.y) else lineTo(p.x, p.y)
                }
            }
            drawPath(past, line, style = Stroke(width = 5f))
            if (forecast.isNotEmpty() && history.isNotEmpty()) {
                val start = history.size - 1
                val future = Path().apply {
                    val origin = point(start, history.last())
                    moveTo(origin.x, origin.y)
                    forecast.forEachIndexed { i, v ->
                        val p = point(start + i + 1, v)
                        lineTo(p.x, p.y)
                    }
                }
                drawPath(
                    future, line.copy(alpha = 0.8f),
                    style = Stroke(width = 5f, pathEffect = PathEffect.dashPathEffect(floatArrayOf(18f, 12f))),
                )
            }
        }
        Text(money(low), style = MaterialTheme.typography.labelSmall)
        Row {
            Text(game.history.months.firstOrNull()?.take(7) ?: "", style = MaterialTheme.typography.labelSmall)
            Spacer(Modifier.weight(1f))
            Text("forecast to ${game.forecast.months.lastOrNull()?.take(7) ?: ""}", style = MaterialTheme.typography.labelSmall)
            Spacer(Modifier.width(4.dp))
        }
    }
}

fun money(value: Double): String = "$" + "%,.2f".format(value)
