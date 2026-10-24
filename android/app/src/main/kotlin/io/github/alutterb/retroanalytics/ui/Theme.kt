package io.github.alutterb.retroanalytics.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import io.github.alutterb.retroanalytics.data.Direction

private val Navy = Color(0xFF1D2B53)
private val Sky = Color(0xFF7AA2FF)

/** Green for rising and red for crashing, tuned per theme for contrast. */
@Composable
fun directionColor(direction: Direction): Color {
    val dark = isSystemInDarkTheme()
    return when (direction) {
        Direction.RISING -> if (dark) Color(0xFF4CC26B) else Color(0xFF1A7F37)
        Direction.CRASHING -> if (dark) Color(0xFFFF6B6B) else Color(0xFFC62828)
    }
}

@Composable
fun RetroTheme(content: @Composable () -> Unit) {
    val colors = if (isSystemInDarkTheme()) {
        darkColorScheme(primary = Sky, secondary = Sky)
    } else {
        lightColorScheme(primary = Navy, secondary = Navy)
    }
    MaterialTheme(colorScheme = colors, content = content)
}
