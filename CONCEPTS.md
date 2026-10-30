# Concepts

## Price signals

### Forecast growth
How much a game's complete-in-box price is expected to change over the forecast horizon, as a percentage of its latest monthly price. The ranking orders games by it.

### Realized change
How much a game's complete-in-box price actually moved over a recent window of months, as a percentage of the price at the start of the window. A window the game has too little history for has no value; it does not count as no change.

### Rising
A game whose forecast growth or any realized change is at or above that signal's rising threshold.

### Crashing
A game whose forecast growth or any realized change is at or below that signal's crashing threshold. A game can be rising and crashing at once, for example after a sharp drop that the forecast expects to recover.

### Rebound
A game that meets a crashing threshold on a realized price change (1, 3 or 6 months) while its forecast growth is at or above the rebound threshold: it recently crashed, but the model expects it to climb. It is also crashing; the rebound is a separate field in the export, ranked by forecast growth.

### Flag reason
One condition that made a game rising or crashing: which signal, its value, the threshold it crossed, and a short sentence for people. A flagged game keeps every reason that applied.

## Ratings

### Underrated score
How far a game's IGDB user rating, shrunk toward its console's median rating, sits above what its IGDB popularity (its votes plus its parent entry's) predicts among the games on the same console, in that console's standard deviations. Games need at least 10 votes to be listed; bundles, compilations and hardware are left out, and price plays no part.

## Delivery

### Export
The versioned file of flagged games, with enough price history and forecast for the phone app to draw them, plus each console's top underrated games. Its flagged games are only the rising or crashing ones, never the whole tracked set.

### Results repository
The owner's private GitHub repository that receives the export each week and that the phone app reads. Publishing refuses any repository it cannot confirm is private, because the prices are for personal use only.
