package io.github.alutterb.retroanalytics.work

import android.content.Context
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.NetworkType
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import io.github.alutterb.retroanalytics.data.ResultsRepository
import java.util.concurrent.TimeUnit

/** Daily background download; the PC publishes weekly, so once a day is plenty. */
class RefreshWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result {
        val repository = ResultsRepository(applicationContext)
        Notifier.afterRefresh(applicationContext, repository, repository.refresh(), background = true)
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
    }
}
