package io.github.alutterb.retroanalytics.work

import android.content.Context
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.NetworkType
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import io.github.alutterb.retroanalytics.data.Failure
import io.github.alutterb.retroanalytics.data.Refreshed
import io.github.alutterb.retroanalytics.data.ResultsRepository
import java.util.concurrent.TimeUnit

/** Daily background download; the PC publishes weekly, so once a day is plenty. */
class RefreshWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result {
        val repository = ResultsRepository(applicationContext)
        handle(applicationContext, repository, repository.refresh())
        return Result.success()
    }

    companion object {
        private const val NAME = "refresh-results"

        fun schedule(context: Context) {
            val request = PeriodicWorkRequestBuilder<RefreshWorker>(1, TimeUnit.DAYS)
                .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
                .build()
            WorkManager.getInstance(context)
                .enqueueUniquePeriodicWork(NAME, ExistingPeriodicWorkPolicy.KEEP, request)
        }

        /** Notifications after any download, from the worker or the app. */
        fun handle(context: Context, repository: ResultsRepository, refreshed: Refreshed) {
            when (refreshed) {
                is Refreshed.Updated -> {
                    repository.settings.tokenWarningShown = false
                    Notifier.newFlags(context, refreshed.newFlags)
                }
                is Refreshed.Failed -> if (refreshed.failure == Failure.BAD_TOKEN &&
                    !repository.settings.tokenWarningShown
                ) {
                    repository.settings.tokenWarningShown = true
                    Notifier.tokenRejected(context)
                }
            }
        }
    }
}
