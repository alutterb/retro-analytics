package io.github.alutterb.retroanalytics.work

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import io.github.alutterb.retroanalytics.R
import io.github.alutterb.retroanalytics.data.Direction
import io.github.alutterb.retroanalytics.data.NewFlag
import io.github.alutterb.retroanalytics.data.Refreshed
import io.github.alutterb.retroanalytics.data.ResultsRepository
import io.github.alutterb.retroanalytics.data.TokenWarning
import io.github.alutterb.retroanalytics.data.summaryText
import io.github.alutterb.retroanalytics.data.tokenWarning
import io.github.alutterb.retroanalytics.ui.MainActivity

/** Local notifications: one summary per download, and one warning when the token stops working. */
object Notifier {
    const val EXTRA_TAB = "tab"
    private const val CHANNEL_FLAGS = "flags"
    private const val CHANNEL_PROBLEMS = "problems"
    private const val ID_FLAGS = 1
    private const val ID_TOKEN = 2

    fun createChannels(context: Context) {
        val manager = context.getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL_FLAGS, "Newly flagged games", NotificationManager.IMPORTANCE_DEFAULT),
        )
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL_PROBLEMS, "Refresh problems", NotificationManager.IMPORTANCE_LOW),
        )
    }

    /**
     * After any download: announce new flags. Only the background worker warns
     * about a rejected token; in the app the banner already says so.
     */
    fun afterRefresh(context: Context, repository: ResultsRepository, refreshed: Refreshed, background: Boolean) {
        if (refreshed is Refreshed.Updated) newFlags(context, refreshed.newFlags)
        when (tokenWarning(refreshed, repository.rejectedTokenNotified)) {
            TokenWarning.RESET -> repository.rejectedTokenNotified = false
            TokenWarning.POST -> if (background && canPost(context)) {
                repository.rejectedTokenNotified = true
                tokenRejected(context)
            }
            TokenWarning.NONE -> Unit
        }
    }

    private fun newFlags(context: Context, newFlags: List<NewFlag>) {
        if (newFlags.isEmpty()) return
        val text = summaryText(newFlags)
        post(context, ID_FLAGS, CHANNEL_FLAGS, "Retro price alert", text, newFlags.first().direction)
    }

    private fun tokenRejected(context: Context) {
        post(
            context, ID_TOKEN, CHANNEL_PROBLEMS, "Refresh failing: token rejected",
            "GitHub no longer accepts the saved token. Open settings to paste a new one.", null,
        )
    }

    private fun canPost(context: Context): Boolean =
        Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU ||
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) ==
            PackageManager.PERMISSION_GRANTED

    private fun post(context: Context, id: Int, channel: String, title: String, text: String, tab: Direction?) {
        if (!canPost(context)) return
        val intent = Intent(context, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
            .apply { tab?.let { putExtra(EXTRA_TAB, it.name) } }
        val pending = PendingIntent.getActivity(
            context, id, intent, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = NotificationCompat.Builder(context, channel)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(title)
            .setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .setContentIntent(pending)
            .setAutoCancel(true)
            .build()
        try {
            NotificationManagerCompat.from(context).notify(id, notification)
        } catch (e: SecurityException) {
            // Permission withdrawn between the check and the post.
        }
    }
}
