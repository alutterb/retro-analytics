package io.github.alutterb.retroanalytics.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.github.alutterb.retroanalytics.data.Direction
import io.github.alutterb.retroanalytics.data.Failure
import io.github.alutterb.retroanalytics.data.Refreshed
import io.github.alutterb.retroanalytics.data.Results
import io.github.alutterb.retroanalytics.data.ResultsRepository
import io.github.alutterb.retroanalytics.data.SortOrder
import io.github.alutterb.retroanalytics.work.Notifier
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

enum class Screen { LIST, DETAIL, SETTINGS }

data class UiState(
    val screen: Screen,
    val results: Results? = null,
    val newKeys: Set<Pair<String, Direction>> = emptySet(),
    val loading: Boolean = false,
    val failure: Failure? = null,
    val tab: Direction = Direction.RISING,
    val console: String? = null,
    val sort: SortOrder = SortOrder.entries.first(),
    val selected: String? = null,
    val settings: SettingsState,
)

data class SettingsState(
    val url: String,
    val token: String = "",
    val hasSavedToken: Boolean,
    val testing: Boolean = false,
    val testMessage: String? = null,
)

class AppViewModel(application: Application) : AndroidViewModel(application) {
    private val repository = ResultsRepository(application)

    private val _state: MutableStateFlow<UiState>
    val state: StateFlow<UiState>
    private var refreshJob: Job? = null

    init {
        val hasToken = repository.hasToken
        _state = MutableStateFlow(
            UiState(
                screen = if (hasToken) Screen.LIST else Screen.SETTINGS,
                results = repository.cached(),
                newKeys = repository.newKeys(),
                settings = SettingsState(url = repository.resultsUrl, hasSavedToken = hasToken),
            ),
        )
        state = _state.asStateFlow()
        if (hasToken) refresh()
    }

    /** Downloads now; also picks up anything the background worker cached meanwhile. */
    fun refresh() {
        if (refreshJob?.isActive == true) return
        _state.update { it.copy(loading = true) }
        refreshJob = viewModelScope.launch {
            val refreshed = repository.refresh()
            Notifier.afterRefresh(getApplication(), repository, refreshed, background = false)
            val cached = repository.cached()
            val newKeys = repository.newKeys()
            _state.update {
                when (refreshed) {
                    is Refreshed.Updated -> it.copy(loading = false, failure = null, results = refreshed.results, newKeys = newKeys)
                    is Refreshed.Failed -> it.copy(
                        loading = false, failure = refreshed.failure, results = cached ?: it.results, newKeys = newKeys,
                    )
                }
            }
        }
    }

    fun openTab(direction: Direction) = _state.update { it.copy(tab = direction, screen = Screen.LIST) }
    fun filterConsole(console: String?) = _state.update { it.copy(console = console) }
    fun sortBy(sort: SortOrder) = _state.update { it.copy(sort = sort) }
    fun open(key: String) = _state.update { it.copy(selected = key, screen = Screen.DETAIL) }
    fun back() = _state.update { it.copy(screen = Screen.LIST) }

    fun openSettings() {
        resetSettingsForm()
        _state.update { it.copy(screen = Screen.SETTINGS) }
    }

    fun setUrl(url: String) = _state.update { it.copy(settings = it.settings.copy(url = url, testMessage = null)) }
    fun setToken(token: String) = _state.update { it.copy(settings = it.settings.copy(token = token, testMessage = null)) }

    /** Tries the values on screen without saving them. */
    fun testConnection() {
        val form = _state.value.settings
        val token = form.token.ifBlank { repository.savedToken() }
        if (token == null) {
            _state.update { it.copy(settings = it.settings.copy(testMessage = Failure.NO_TOKEN.message)) }
            return
        }
        _state.update { it.copy(settings = it.settings.copy(testing = true, testMessage = null)) }
        viewModelScope.launch {
            val message = when (val tested = repository.test(form.url, token)) {
                is Refreshed.Failed -> tested.failure.message
                is Refreshed.Updated ->
                    "Connected: ${tested.results.games.size} flagged games, generated ${tested.results.generated}"
            }
            _state.update { it.copy(settings = it.settings.copy(testing = false, testMessage = message)) }
        }
    }

    /** Saves, then downloads with the new settings and shows the lists. */
    fun saveSettings() {
        val form = _state.value.settings
        _state.update { it.copy(screen = Screen.LIST, failure = null) }
        viewModelScope.launch {
            refreshJob?.join()
            val locationChanged = repository.save(form.url, form.token)
            _state.update {
                if (locationChanged) it.copy(results = null, newKeys = emptySet()) else it
            }
            resetSettingsForm()
            refresh()
        }
    }

    private fun resetSettingsForm() = _state.update {
        it.copy(settings = SettingsState(url = repository.resultsUrl, hasSavedToken = repository.hasToken))
    }
}
