package io.github.alutterb.retroanalytics.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp

@Composable
fun SettingsScreen(form: SettingsState, model: AppViewModel, onSaved: () -> Unit, modifier: Modifier = Modifier) {
    Column(
        modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            "Paste a fine-grained GitHub token with read-only access to your results repository. " +
                "It is stored encrypted on this phone and only sent to api.github.com.",
            style = MaterialTheme.typography.bodyMedium,
        )
        OutlinedTextField(
            value = form.token,
            onValueChange = model::setToken,
            label = { Text("GitHub token") },
            placeholder = { if (form.hasSavedToken) Text("Saved - type to replace") },
            singleLine = true,
            visualTransformation = PasswordVisualTransformation(),
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password, autoCorrectEnabled = false),
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = form.url,
            onValueChange = model::setUrl,
            label = { Text("Results location") },
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri, autoCorrectEnabled = false),
            modifier = Modifier.fillMaxWidth(),
        )
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
            OutlinedButton(onClick = model::testConnection, enabled = !form.testing) { Text("Test connection") }
            Button(
                onClick = { model.saveSettings(); onSaved() },
                enabled = form.token.isNotBlank() || form.hasSavedToken,
            ) { Text("Save") }
            if (form.testing) CircularProgressIndicator(Modifier.size(24.dp))
        }
        form.testMessage?.let { Text(it, style = MaterialTheme.typography.bodyMedium) }
    }
}
