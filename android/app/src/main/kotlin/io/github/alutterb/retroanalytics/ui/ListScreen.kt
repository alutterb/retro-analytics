package io.github.alutterb.retroanalytics.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.PrimaryTabRow
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab as TabHeader
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import io.github.alutterb.retroanalytics.data.Direction
import io.github.alutterb.retroanalytics.data.Failure
import io.github.alutterb.retroanalytics.data.Game
import io.github.alutterb.retroanalytics.data.Results
import io.github.alutterb.retroanalytics.data.SortOrder
import io.github.alutterb.retroanalytics.data.browse
import io.github.alutterb.retroanalytics.data.browseUnderrated
import io.github.alutterb.retroanalytics.data.consoles
import io.github.alutterb.retroanalytics.data.headline

@Composable
fun ListScreen(state: UiState, model: AppViewModel, modifier: Modifier = Modifier) {
    val results = state.results
    Column(modifier.fillMaxSize()) {
        Banner(state, model)
        if (results == null) {
            NoData(state, model)
            return@Column
        }
        PrimaryTabRow(selectedTabIndex = state.tab.ordinal) {
            Tab.entries.forEach { tab ->
                TabHeader(selected = state.tab == tab, onClick = { model.openTab(tab) }, text = { Text(tab.label) })
            }
        }
        Controls(results, state, model)
        val direction = state.tab.direction
        if (direction == null) {
            val rows = remember(results, state.console, state.sort) { browseUnderrated(results, state.console, state.sort) }
            RememberedList(state, model, rows, { it.key }, "No underrated games yet") { row ->
                UnderratedRow(row) { model.open(row.key) }
            }
        } else {
            val games = remember(results, direction, state.console, state.sort) {
                browse(results, direction, state.console, state.sort)
            }
            RememberedList(state, model, games, { it.key }, "No ${direction.label} games") { game ->
                GameRow(game, direction, (game.key to direction) in state.newKeys) { model.open(game.key) }
            }
        }
    }
}

/** A tab's rows, scrolled back to where that tab was when the list returns from a detail screen. */
@Composable
private fun <T> RememberedList(
    state: UiState,
    model: AppViewModel,
    rows: List<T>,
    keyOf: (T) -> String,
    emptyText: String,
    row: @Composable (T) -> Unit,
) {
    val keys = remember(rows) { rows.map(keyOf) }
    val currentKeys = rememberUpdatedState(keys)
    if (rows.isEmpty()) {
        Empty(if (state.console == null) emptyText else "No games for this console")
        return
    }
    // Fresh state per tab, seeded from the view model, since the list leaves composition on the detail screen.
    key(state.tab, state.console, state.sort) {
        val (index, offset) = remember { model.scrollMemory.restore(state.tab, keys) }
        val listState = rememberLazyListState(index, offset)
        LaunchedEffect(listState, state.tab) {
            snapshotFlow { listState.firstVisibleItemIndex to listState.firstVisibleItemScrollOffset }
                .collect { (first, scrolled) ->
                    currentKeys.value.getOrNull(first)?.let { model.scrollMemory.save(state.tab, it, scrolled) }
                }
        }
        LazyColumn(state = listState) {
            items(rows, key = keyOf) { item ->
                row(item)
                HorizontalDivider()
            }
        }
    }
}

@Composable
private fun Banner(state: UiState, model: AppViewModel) {
    val failure = state.failure ?: return
    if (state.results == null) return
    val dated = "Showing data from ${state.results.generated}."
    val text = if (failure == Failure.NO_NETWORK) "Offline. $dated It may be out of date." else "${failure.message}. $dated"
    Surface(color = MaterialTheme.colorScheme.secondaryContainer, modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.padding(horizontal = 16.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(text, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
            if (failure.fixInSettings) TextButton(onClick = model::openSettings) { Text("Settings") }
        }
    }
}

@Composable
private fun NoData(state: UiState, model: AppViewModel) {
    Box(Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
        if (state.loading || state.failure == null) {
            CircularProgressIndicator()
            return@Box
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("No results yet", style = MaterialTheme.typography.titleMedium)
            Text(state.failure.message, style = MaterialTheme.typography.bodyMedium)
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = model::refresh) { Text("Retry") }
                OutlinedButton(onClick = model::openSettings) { Text("Settings") }
            }
        }
    }
}

@Composable
private fun Controls(results: Results, state: UiState, model: AppViewModel) {
    var sortOpen by remember { mutableStateOf(false) }
    Row(
        Modifier.fillMaxWidth().padding(horizontal = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Row(
            Modifier.weight(1f).horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            FilterChip(selected = state.console == null, onClick = { model.filterConsole(null) }, label = { Text("All") })
            remember(results, state.tab) { consoles(results, underrated = state.tab == Tab.UNDERRATED) }.forEach { console ->
                FilterChip(
                    selected = state.console == console,
                    onClick = { model.filterConsole(console) },
                    label = { Text(console) },
                )
            }
        }
        Box {
            TextButton(onClick = { sortOpen = true }) { Text(state.tab.sortLabel(state.sort)) }
            DropdownMenu(expanded = sortOpen, onDismissRequest = { sortOpen = false }) {
                SortOrder.entries.forEach { sort ->
                    DropdownMenuItem(text = { Text(state.tab.sortLabel(sort)) }, onClick = { model.sortBy(sort); sortOpen = false })
                }
            }
        }
    }
}

@Composable
private fun GameRow(game: Game, direction: Direction, isNew: Boolean, onClick: () -> Unit) {
    val flag = game.headline(direction) ?: return
    Row(
        Modifier.fillMaxWidth().clickable(onClick = onClick).padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(game.title, style = MaterialTheme.typography.titleMedium)
                if (isNew) {
                    Spacer(Modifier.width(8.dp))
                    Text("New", color = MaterialTheme.colorScheme.primary, style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                }
            }
            Text(game.console, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(flag.text, style = MaterialTheme.typography.bodyMedium, color = directionColor(direction))
        }
        Text(money(game.lastPrice), style = MaterialTheme.typography.titleMedium)
    }
}

@Composable
private fun Empty(text: String) {
    Box(Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
        Text(text, style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}
