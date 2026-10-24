package io.github.alutterb.retroanalytics.data

import android.content.Context
import java.io.File

/** The last results file that parsed, kept so the app works offline. */
class ResultsStore(context: Context) {
    private val file = File(context.filesDir, "latest.json")
    private val newFile = File(context.filesDir, "new.txt")

    fun read(): Results? =
        if (file.exists()) (parseResults(file.readText()) as? ParseResult.Ok)?.results else null

    /** Replaces the cached file in one rename, so a crash never leaves half a file. */
    fun write(text: String, newFlags: List<NewFlag>) {
        val temporary = File(file.parentFile, "latest.json.tmp")
        temporary.writeText(text)
        check(temporary.renameTo(file)) { "could not replace cached results" }
        newFile.writeText(newFlags.joinToString("\n") { "${it.game.key} ${it.direction.name}" })
    }

    /** Keys and directions newly flagged in the last download, for the "New" marker. */
    fun newlyFlaggedKeys(): Set<Pair<String, Direction>> =
        if (!newFile.exists()) emptySet() else newFile.readLines().mapNotNull { line ->
            val (key, direction) = line.split(' ').takeIf { it.size == 2 } ?: return@mapNotNull null
            key to Direction.valueOf(direction)
        }.toSet()

    fun clear() {
        file.delete()
        newFile.delete()
    }
}
