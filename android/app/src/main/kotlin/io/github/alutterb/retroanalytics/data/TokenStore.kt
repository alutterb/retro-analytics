package io.github.alutterb.retroanalytics.data

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.security.GeneralSecurityException
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * The GitHub token and results location.
 *
 * The token is encrypted with an AES-GCM key that lives in the Android
 * Keystore and cannot leave it; only the ciphertext is stored. Nothing here
 * logs or returns the token except to the code that sends it to GitHub.
 */
class TokenStore(context: Context) {
    private val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)

    var resultsUrl: String
        get() = prefs.getString(KEY_URL, null) ?: DEFAULT_RESULTS_URL
        set(value) = prefs.edit().putString(KEY_URL, value.trim()).apply()

    val hasToken: Boolean get() = token() != null

    /** The saved token, or null when none is saved or it can no longer be decrypted. */
    fun token(): String? {
        val sealed = prefs.getString(KEY_TOKEN, null) ?: return null
        return try {
            val (iv, data) = sealed.split(':').map { Base64.decode(it, Base64.NO_WRAP) }
            val cipher = Cipher.getInstance(TRANSFORMATION)
            cipher.init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(128, iv))
            String(cipher.doFinal(data), Charsets.UTF_8)
        } catch (e: GeneralSecurityException) {
            null
        } catch (e: IllegalArgumentException) {
            null
        }
    }

    fun saveToken(token: String) {
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, key())
        val data = cipher.doFinal(token.trim().toByteArray(Charsets.UTF_8))
        val sealed = listOf(cipher.iv, data).joinToString(":") { Base64.encodeToString(it, Base64.NO_WRAP) }
        prefs.edit().putString(KEY_TOKEN, sealed).apply()
    }

    /** Whether the owner was already told the background refresh is rejecting the token. */
    var tokenWarningShown: Boolean
        get() = prefs.getBoolean(KEY_WARNED, false)
        set(value) = prefs.edit().putBoolean(KEY_WARNED, value).apply()

    private fun key(): SecretKey {
        val keyStore = KeyStore.getInstance(KEYSTORE).apply { load(null) }
        (keyStore.getKey(KEY_ALIAS, null) as? SecretKey)?.let { return it }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, KEYSTORE)
        generator.init(
            KeyGenParameterSpec.Builder(KEY_ALIAS, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .build(),
        )
        return generator.generateKey()
    }

    private companion object {
        const val KEYSTORE = "AndroidKeyStore"
        const val KEY_ALIAS = "github-token"
        const val TRANSFORMATION = "AES/GCM/NoPadding"
        const val KEY_URL = "results_url"
        const val KEY_TOKEN = "token"
        const val KEY_WARNED = "token_warning_shown"
    }
}
