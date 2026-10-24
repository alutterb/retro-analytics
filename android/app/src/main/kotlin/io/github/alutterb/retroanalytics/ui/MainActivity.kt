package io.github.alutterb.retroanalytics.ui

import android.Manifest
import android.content.Intent
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import io.github.alutterb.retroanalytics.data.Direction
import io.github.alutterb.retroanalytics.work.Notifier
import io.github.alutterb.retroanalytics.work.RefreshWorker

class MainActivity : ComponentActivity() {
    private val model: AppViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Notifier.createChannels(this)
        RefreshWorker.schedule(this)
        openTabFrom(intent)
        setContent { RetroTheme { App(model) } }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        openTabFrom(intent)
    }

    private fun openTabFrom(intent: Intent?) {
        val tab = intent?.getStringExtra(Notifier.EXTRA_TAB) ?: return
        Direction.entries.firstOrNull { it.name == tab }?.let(model::openTab)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun App(model: AppViewModel) {
    val state by model.state.collectAsStateWithLifecycle()
    val askPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {}
    val selected = remember(state.results, state.selected) { state.results?.games?.firstOrNull { it.key == state.selected } }
    val screen = if (state.screen == Screen.DETAIL && selected == null) Screen.LIST else state.screen
    // Settings has no way back until a token is saved.
    val canGoBack = screen != Screen.LIST && (screen != Screen.SETTINGS || state.settings.hasSavedToken)

    BackHandler(enabled = canGoBack) { model.back() }
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        when (screen) {
                            Screen.SETTINGS -> "Settings"
                            Screen.DETAIL -> "Game"
                            Screen.LIST -> state.results?.let { "Retro prices - ${it.generated}" } ?: "Retro prices"
                        },
                    )
                },
                navigationIcon = {
                    if (canGoBack) TextButton(onClick = model::back) { Text("Back") }
                },
                actions = {
                    if (screen == Screen.LIST) {
                        TextButton(onClick = model::refresh, enabled = !state.loading) { Text("Refresh") }
                        TextButton(onClick = model::openSettings) { Text("Settings") }
                    }
                },
            )
        },
    ) { padding ->
        val modifier = Modifier.padding(padding)
        when (screen) {
            Screen.LIST -> ListScreen(state, model, modifier)
            Screen.DETAIL -> DetailScreen(selected!!, modifier)
            Screen.SETTINGS -> SettingsScreen(state.settings, model, onSaved = {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    askPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
                }
            }, modifier)
        }
    }
}
