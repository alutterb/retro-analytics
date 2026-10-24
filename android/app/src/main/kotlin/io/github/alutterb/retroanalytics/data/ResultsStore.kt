package io.github.alutterb.retroanalytics.data

import java.io.File

/** The last results file that parsed, kept so the app works offline, plus its "New" markers. */
class ResultsStore(directory: File) {
    private val file = File(directory, "latest.json")
    private val newFile = File(directory, "new.txt")

    fun readText(): String? = file.takeIf { it.exists() }?.readText()

    fun read(): Results? = readText()?.let { (parseResults(it) as? ParseResult.Ok)?.results }

    /** Throws IOException or IllegalStateException when the cache cannot be written. */
    fun write(text: String, newFlags: List<NewFlag>) {
        val temporary = File(file.parentFile, "latest.json.tmp")
        temporary.writeText(text)
        check(temporary.renameTo(file)) { "could not replace cached results" }
        newFile.writeText(newFlags.joinToString("\n") { "${it.id.first} ${it.id.second.name}" })
    }

    fun newlyFlaggedKeys(): Set<Pair<String, Direction>> =
        if (!newFile.exists()) emptySet() else newFile.readLines().mapNotNull { line ->
            val (key, direction) = line.split(' ').takeIf { it.size == 2 } ?: return@mapNotNull null
            Direction.entries.firstOrNull { it.name == direction }?.let { key to it }
        }.toSet()

    fun clear() {
        file.delete()
        newFile.delete()
    }
}
