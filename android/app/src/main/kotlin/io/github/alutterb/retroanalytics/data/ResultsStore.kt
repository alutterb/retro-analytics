package io.github.alutterb.retroanalytics.data

import java.io.File

/**
 * The last results file that parsed, kept so the app works offline, plus its
 * "New" markers and the flags still waiting for a notification.
 */
class ResultsStore(directory: File) {
    private val file = File(directory, "latest.json")
    private val newFile = File(directory, "new.txt")
    private val pendingFile = File(directory, "pending.txt")

    fun readText(): String? = file.takeIf { it.exists() }?.readText()

    fun read(): Results? = readText()?.let { (parseResults(it) as? ParseResult.Ok)?.results }

    /**
     * Throws IOException or IllegalStateException when the cache cannot be written.
     * The markers go first: a crash in between leaves old results that the
     * next download still differs from, so the pair is rewritten.
     */
    fun write(text: String, newFlags: List<NewFlag>) {
        writeKeys(newFile, newFlags.map { it.id })
        val temporary = File(file.parentFile, "latest.json.tmp")
        temporary.writeText(text)
        check(temporary.renameTo(file)) { "could not replace cached results" }
    }

    fun newlyFlaggedKeys(): Set<FlagKey> = readKeys(newFile)

    fun pendingKeys(): Set<FlagKey> = readKeys(pendingFile)

    fun setPendingKeys(keys: Set<FlagKey>) = writeKeys(pendingFile, keys)

    fun clear() {
        file.delete()
        newFile.delete()
        pendingFile.delete()
    }

    private fun writeKeys(target: File, keys: Collection<FlagKey>) =
        target.writeText(keys.joinToString("\n") { "${it.first} ${it.second.name}" })

    private fun readKeys(source: File): Set<FlagKey> =
        if (!source.exists()) emptySet() else source.readLines().mapNotNull { line ->
            val (key, direction) = line.split(' ').takeIf { it.size == 2 } ?: return@mapNotNull null
            Direction.entries.firstOrNull { it.name == direction }?.let { key to it }
        }.toSet()
}
