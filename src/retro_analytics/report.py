"""Write the ranking as one self-contained HTML page.

The page has no external files or scripts: a sortable table, a console filter
and, per game, a sparkline of recent monthly prices followed by the dashed
forecast. Open it from disk in any browser.
"""

from datetime import date
from html import escape

import pandas as pd

from retro_analytics import paths

# Months of history drawn before the forecast in each sparkline.
SPARK_MONTHS = 24
SPARK_WIDTH = 160
SPARK_HEIGHT = 36


def sparkline(history, forecast, width=SPARK_WIDTH, height=SPARK_HEIGHT, pad=3):
    """Inline SVG: history as a solid line, forecast as a dashed line continuing it."""
    history, forecast = list(history), list(forecast)
    values = history + forecast
    if not values:
        return ''
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    step = (width - 2 * pad) / max(len(values) - 1, 1)

    def point(i, value):
        return f"{pad + i * step:.1f},{pad + (high - value) / span * (height - 2 * pad):.1f}"

    past = ' '.join(point(i, v) for i, v in enumerate(history))
    # The forecast starts from the last known price so the two lines join.
    start = len(history) - 1
    future = ' '.join(point(start + i, v) for i, v in enumerate(history[-1:] + forecast))
    label = f"{len(history)} months of prices then {len(forecast)} forecast months"
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="{label}">'
        f'<polyline class="past" points="{past}"/>'
        f'<polyline class="future" points="{future}"/></svg>'
    )


def money(value):
    return f"${value:,.2f}"


def table_rows(results, series, forecasts, urls):
    history = {key: rows.sort_values('month')['price'].tail(SPARK_MONTHS).tolist()
               for key, rows in series.groupby(['console', 'game'])}
    future = {key: rows.sort_values('step')['forecast'].tolist()
              for key, rows in forecasts.groupby(['console', 'game'])}

    rows = []
    for r in results.itertuples(index=False):
        key = (r.console, r.game)
        title = escape(str(r.title))
        url = urls.get(key)
        name = f'<a href="{escape(url)}">{title}</a>' if url else title
        has_sentiment = pd.notna(r.sentiment)
        sentiment_cell = f"{r.sentiment:+.2f} ({r.sentiment_texts})" if has_sentiment else "–"
        cells = [
            (r.rank, str(r.rank)),
            (str(r.title).lower(), name),
            (r.console, escape(r.console)),
            (r.last_price, f"{money(r.last_price)} <small>{escape(r.last_month[:7])}</small>"),
            (r.forecast_price, f"{money(r.forecast_price)} <small>{escape(r.forecast_month[:7])}</small>"),
            (r.growth_pct, f"{r.growth_pct:+.1f}%"),
            (r.sentiment if has_sentiment else -2, sentiment_cell),
        ]
        tds = ''.join(f'<td data-sort="{escape(str(sort))}">{shown}</td>' for sort, shown in cells)
        tds += f'<td>{sparkline(history.get(key, []), future.get(key, []))}</td>'
        growth = 'up' if r.growth_pct > 0 else 'down' if r.growth_pct < 0 else 'flat'
        rows.append(f'<tr data-console="{escape(r.console)}" class="{growth}">{tds}</tr>')
    return '\n'.join(rows)


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Retro game price outlook</title>
<style>
:root {{ --bg: #fff; --fg: #1d1d1f; --muted: #6b6b70; --line: #e3e3e8; --up: #1a7f37; --down: #c62828; --spark: #3b6fd8; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #141416; --fg: #ececf0; --muted: #9a9aa2; --line: #2c2c31; --up: #4cc26b; --down: #ff6b6b; --spark: #7aa2ff; }}
}}
body {{ margin: 0; padding: 24px 16px; background: var(--bg); color: var(--fg);
  font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1100px; margin: 0 auto; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
p {{ color: var(--muted); margin: 4px 0 16px; }}
a {{ color: inherit; }}
.controls {{ margin-bottom: 12px; }}
.wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }}
th {{ cursor: pointer; user-select: none; position: sticky; top: 0; background: var(--bg); }}
th[aria-sort="ascending"]::after {{ content: " ▲"; }}
th[aria-sort="descending"]::after {{ content: " ▼"; }}
td:nth-child(1), td:nth-child(4), td:nth-child(5), td:nth-child(6), td:nth-child(7) {{ text-align: right; font-variant-numeric: tabular-nums; }}
small {{ color: var(--muted); }}
tr.up td:nth-child(6) {{ color: var(--up); }}
tr.down td:nth-child(6) {{ color: var(--down); }}
.spark polyline {{ fill: none; stroke: var(--spark); stroke-width: 1.5; }}
.spark .future {{ stroke-dasharray: 3 2; opacity: .8; }}
</style>
</head>
<body>
<main>
<h1>Retro game price outlook</h1>
<p>{count} games ranked by forecast complete-in-box price growth over the next {horizon} months.
Generated {generated}. Prices from <a href="https://www.pricecharting.com">PriceCharting</a>, for private use.
Forecasts are statistical extrapolations, not advice.</p>
<div class="controls"><label>Console <select id="console">
<option value="">All consoles</option>
{options}
</select></label></div>
<div class="wrap"><table id="ranking">
<thead><tr><th>#</th><th>Game</th><th>Console</th><th>Latest</th><th>Forecast</th><th>Growth</th><th>Sentiment</th><th>Trend</th></tr></thead>
<tbody>
{rows}
</tbody>
</table></div>
</main>
<script>
const table = document.getElementById("ranking");
const body = table.tBodies[0];
const headers = Array.from(table.tHead.rows[0].cells);
headers.forEach((th, col) => {{
  if (col === headers.length - 1) return;
  th.addEventListener("click", () => {{
    const ascending = th.getAttribute("aria-sort") !== "ascending";
    headers.forEach(h => h.removeAttribute("aria-sort"));
    th.setAttribute("aria-sort", ascending ? "ascending" : "descending");
    const value = row => row.cells[col].dataset.sort;
    const rows = Array.from(body.rows).sort((a, b) => {{
      const x = value(a), y = value(b);
      const order = isNaN(x) || isNaN(y) ? x.localeCompare(y) : x - y;
      return ascending ? order : -order;
    }});
    rows.forEach(row => body.appendChild(row));
  }});
}});
document.getElementById("console").addEventListener("change", event => {{
  for (const row of body.rows) row.hidden = event.target.value !== "" && row.dataset.console !== event.target.value;
}});
</script>
</body>
</html>
"""


def render(results, series, forecasts, games=None, generated=None):
    urls = {}
    if games is not None and 'url' in games:
        urls = {(g.console, g.game): g.url for g in games.itertuples(index=False)}
    horizon = int(forecasts['step'].max()) if not forecasts.empty else 0
    options = '\n'.join(f'<option value="{escape(c)}">{escape(c)}</option>' for c in sorted(results['console'].unique()))
    return PAGE.format(
        count=len(results),
        horizon=horizon,
        generated=escape(generated or date.today().isoformat()),
        options=options,
        rows=table_rows(results, series, forecasts, urls),
    )


def main():
    results = pd.read_csv(paths.require(paths.results(), "rank"))
    series = pd.read_csv(paths.require(paths.series(), "prep"))
    forecasts = pd.read_csv(paths.require(paths.forecasts(), "forecast"))
    games = pd.read_csv(paths.games()) if paths.games().exists() else None
    paths.report().write_text(render(results, series, forecasts, games), encoding='utf-8')
    print(f"Wrote {paths.report()}")


if __name__ == "__main__":
    main()
