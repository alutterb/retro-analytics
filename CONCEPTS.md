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

### Flag reason
One condition that made a game rising or crashing: which signal, its value, the threshold it crossed, and a short sentence for people. A flagged game keeps every reason that applied.

## Delivery

### Export
The versioned file of flagged games, with enough price history and forecast for the phone app to draw them. It holds only rising or crashing games, never the whole tracked set.

### Results repository
The owner's private GitHub repository that receives the export each week and that the phone app reads. Publishing refuses any repository it cannot confirm is private, because the prices are for personal use only.
