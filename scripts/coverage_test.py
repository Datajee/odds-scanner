"""One-off coverage test: what do the free PropLine and SportsGameOdds tiers
actually return for today's NBA slate? Writes a markdown report to the
GitHub Actions run summary. API keys are read from env and never printed."""
import os, json, re
from collections import Counter
from datetime import datetime, timezone
import requests

NBA = "basketball_nba"
PL = "https://api.prop-line.com/v1"
SGO = "https://api.sportsgameodds.com/v2"
WATCH = ["pinnacle", "fanduel", "bet365", "draftkings", "betmgm", "betrivers"]
MAX_GAMES = 3  # keeps request count tiny
out = []


def say(s=""):
    out.append(s)
    print(s)


def shape(obj, depth=0):
    """Keys-only skeleton of a JSON object, so we can see structure safely."""
    if depth > 5:
        return "..."
    if isinstance(obj, dict):
        return {k: shape(v, depth + 1) for k, v in list(obj.items())[:12]}
    if isinstance(obj, list):
        return [shape(obj[0], depth + 1)] if obj else []
    return type(obj).__name__


def get(url, headers, params=None):
    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} on {url.split('?')[0]}: {r.text[:300]}")
    return r


def watch_table(books):
    say("\n| Book | Present? |\n|---|---|")
    for b in WATCH:
        hit = any(b in k.lower() for k in books)
        say(f"| {b} | {'✅' if hit else '❌'} |")


def propline():
    key = os.environ.get("PROPLINE_API_KEY")
    say("## PropLine")
    if not key:
        say("❌ PROPLINE_API_KEY secret not found")
        return
    h = {"X-API-Key": key}
    r = get(f"{PL}/sports/{NBA}/events", h)
    events = r.json()
    events = events if isinstance(events, list) else events.get("data", events)
    say(f"- Events returned: **{len(events)}**")
    if not events:
        return
    books, mkts = set(), Counter()
    players, newest = set(), None
    for ev in events[:MAX_GAMES]:
        eid = ev.get("id")
        say(f"- {ev.get('away_team')} @ {ev.get('home_team')} — {ev.get('commence_time')}")
        listing = get(f"{PL}/sports/{NBA}/events/{eid}/markets", h).json()
        keys = sorted({m.get("key") if isinstance(m, dict) else m
                       for m in (listing if isinstance(listing, list)
                                 else listing.get("markets", []))} - {None})
        params = {"markets": ",".join(keys)} if keys else None
        r = get(f"{PL}/sports/{NBA}/events/{eid}/odds", h, params)
        data = r.json()
        for bk in data.get("bookmakers", []):
            books.add(bk.get("key", "?"))
            for m in bk.get("markets", []):
                mkts[m.get("key", "?")] += len(m.get("outcomes", []))
                for o in m.get("outcomes", []):
                    if o.get("description"):
                        players.add(o["description"])
                lu = m.get("last_update") or bk.get("last_update")
                if lu and (newest is None or lu > newest):
                    newest = lu
    say(f"- Daily quota left: **{r.headers.get('X-Daily-Remaining', 'n/a')}**"
        f" of {r.headers.get('X-Daily-Limit', 'n/a')}")
    say(f"- Books seen ({len(books)}): {', '.join(sorted(books))}")
    watch_table(books)
    say(f"\n- Distinct players with props: **{len(players)}**")
    say(f"- Newest price timestamp: {newest}")
    say("\n| Market | Prices (across books) |\n|---|---|")
    for k, n in mkts.most_common(40):
        say(f"| {k} | {n} |")
    say("\n<details><summary>Raw odds shape</summary>\n\n```json\n"
        + json.dumps(shape(data), indent=1)[:3000] + "\n```\n</details>")


def sgo():
    key = os.environ.get("SGO_API_KEY")
    say("\n## SportsGameOdds")
    if not key:
        say("❌ SGO_API_KEY secret not found")
        return
    p = {"leagueID": "NBA", "oddsAvailable": "true", "includeAltLines": "true",
         "limit": str(MAX_GAMES)}
    body = get(f"{SGO}/events", {"x-api-key": key}, p).json()
    events = body.get("data", [])
    say(f"- Events returned: **{len(events)}**")
    books, bet, fair, players = set(), Counter(), 0, set()
    alt, newest, sample = 0, None, None
    for ev in events:
        t = ev.get("teams", {})
        nm = lambda s: (t.get(s, {}).get("names", {}) or {}).get("long", s)
        say(f"- {nm('away')} @ {nm('home')} — {ev.get('status', {}).get('startsAt')}")
        for oid, o in (ev.get("odds") or {}).items():
            sample = sample or o
            parts = oid.split("-")
            bet[parts[3] if len(parts) > 3 else "?"] += 1
            if len(parts) > 1 and parts[1] not in ("home", "away", "all"):
                players.add(parts[1])
            if o.get("fairOdds"):
                fair += 1
            for b, bo in (o.get("byBookmaker") or {}).items():
                books.add(b)
                if isinstance(bo, dict):
                    alt += len(bo.get("altLines") or [])
                    lu = bo.get("lastUpdatedAt")
                    if lu and (newest is None or lu > newest):
                        newest = lu
    say(f"- Books seen ({len(books)}): {', '.join(sorted(books))}")
    watch_table(books)
    say(f"\n- Markets with fair (no-vig) odds: **{fair}**")
    say(f"- Alt lines found: **{alt}**")
    say(f"- Distinct players with props: **{len(players)}**")
    say(f"- Newest price timestamp: {newest}")
    say("\n| Bet type | Markets |\n|---|---|")
    for k, n in bet.most_common():
        say(f"| {k} | {n} |")
    say("\n<details><summary>Raw odds shape</summary>\n\n```json\n"
        + json.dumps(shape(sample), indent=1)[:3000] + "\n```\n</details>")


say(f"# Coverage test — {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
for fn in (propline, sgo):
    try:
        fn()
    except Exception as e:  # report and keep going
        msg = re.sub(r"[A-Za-z0-9_\-]{24,}", "[redacted]", str(e))
        say(f"\n❌ **{fn.__name__} failed:** {msg}")

summary = os.environ.get("GITHUB_STEP_SUMMARY")
if summary:
    with open(summary, "a") as f:
        f.write("\n".join(out) + "\n")
