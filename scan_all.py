"""Odds scanner (single-file build): FanDuel NBA alt spreads with 10-25 pts of
room. Fair price = fat-tailed margin model centred on Pinnacle, tail width
fitted to other books' alt ladders. Run: python scan_all.py"""
from collections import Counter
from datetime import datetime, timedelta, timezone
from math import isclose
from scipy.stats import t as student_t
from statistics import median
from zoneinfo import ZoneInfo
import csv
import os
import requests

# ---- scanner/oddsmath.py ----
def american_to_decimal(a):
    a = float(a)
    if 1.0 < a < 100.0:  # already decimal (American odds are never between -100 and +100)
        return a
    return 1 + (a / 100 if a > 0 else 100 / abs(a))


def implied(dec):
    return 1 / dec


def devig_pair(dec_a, dec_b):
    """Fair probabilities for a two-way market (multiplicative method)."""
    pa, pb = implied(dec_a), implied(dec_b)
    s = pa + pb
    return pa / s, pb / s


def ev(fair_win, fair_push, dec):
    """Expected return per $1 staked. A push refunds the stake."""
    lose = 1 - fair_win - fair_push
    return fair_win * (dec - 1) - lose


class MarginModel:
    """Home-minus-away final margin ~ mu + scale * T(df), treated as integer.

    sd (tail width) is fitted per game to other books' alt ladders where
    possible, else defaults to 12.5 pts. df=6 gives fatter-than-normal tails,
    since NBA blowouts happen more often than a bell curve predicts."""

    def __init__(self, sd=12.5, df=6):
        self.df = df
        self.scale = sd * ((df - 2) / df) ** 0.5
        self.mu = 0.0

    def _cdf(self, x):
        return student_t.cdf((x - self.mu) / self.scale, self.df)

    def p_ge(self, k):  # P(M >= k) for integer k
        return 1 - self._cdf(k - 0.5)

    def p_eq(self, k):
        return self._cdf(k + 0.5) - self._cdf(k - 0.5)

    def home_cover(self, home_line):
        """(win, push) for HOME getting `home_line` points (e.g. -5.5 or +12)."""
        x = -home_line  # home covers when M > x
        if isclose(x, round(x)):
            k = int(round(x))
            return self.p_ge(k + 1), self.p_eq(k)
        return self.p_ge(int(x // 1) + 1), 0.0

    def away_cover(self, away_line):
        """(win, push) for AWAY getting `away_line` points: covers when M < away_line."""
        x = away_line
        if isclose(x, round(x)):
            k = int(round(x))
            return 1 - self.p_ge(k), self.p_eq(k)
        return 1 - self.p_ge(int(x // 1) + 1), 0.0

    def fit_to_main(self, home_line, fair_home_win):
        """Set mu so the model reproduces the sharp book's fair price on its
        main home spread. Bisection; monotone in mu."""
        lo, hi = -60.0, 60.0
        for _ in range(60):
            self.mu = (lo + hi) / 2
            win, push = self.home_cover(home_line)
            target = fair_home_win * (1 - push)  # sharp price excludes pushes
            if win < target:
                lo = self.mu
            else:
                hi = self.mu
        return self.mu


# ---- scanner/propline.py ----
BASE = "https://api.prop-line.com/v1"


class PropLine:
    def __init__(self, api_key=None):
        self.key = api_key or os.environ["PROPLINE_API_KEY"]
        self.requests_left = None
        self.daily_limit = None

    def _get(self, path, params=None):
        r = requests.get(f"{BASE}{path}", headers={"X-API-Key": self.key},
                         params=params, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"PropLine HTTP {r.status_code} on {path}: {r.text[:300]}")
        self.requests_left = r.headers.get("X-Daily-Remaining", self.requests_left)
        self.daily_limit = r.headers.get("X-Daily-Limit", self.daily_limit)
        return r.json()

    def events(self, sport):
        data = self._get(f"/sports/{sport}/events")
        return data if isinstance(data, list) else data.get("data", [])

    def odds(self, sport, event_id, markets):
        return self._get(f"/sports/{sport}/events/{event_id}/odds",
                         {"markets": ",".join(markets)})


# ---- scanner/strategies/nba_alt_spreads.py ----
SPORT = "basketball_nba"
MARKETS = ["spreads"]
BET_BOOK = "fanduel"
SHARP_BOOK = "pinnacle"
ODDS_MIN, ODDS_MAX = 1.05, 1.35
ROOM_MIN, ROOM_MAX = 10, 25
MIN_CONSENSUS_BOOKS = 3
MIN_SCORE = 55
DEFAULT_SD = 12.5                            # fallback tail width (points)
SD_GRID = [x / 2 for x in range(16, 41)]     # 8.0 .. 20.0 pts searched when fitting
MIN_LADDER_POINTS = 8                        # other-book alt prices needed to fit tails
# Pick'em / DFS apps don't post real two-way prices; never use them as a fair line.
NOT_PRICES = {"prizepicks", "underdog", "sleeper", "dabble", "pick6", "dst_propbuilder",
              "thrillzz", "sportzino", "courtside", "rebet"}


def _parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _pairs(book):
    """{(team, point): (decimal_price, fair_prob_vs_other_side)} for full-game,
    on-board spread markets that have both sides."""
    out, mains = {}, []
    for m in book.get("markets", []):
        if m.get("key") != "spreads" or m.get("period") or m.get("suspended_at"):
            continue
        oc = [o for o in m.get("outcomes", []) if o.get("point") is not None]
        if len(oc) != 2:
            continue
        a, b = oc
        da, db = american_to_decimal(a["price"]), american_to_decimal(b["price"])
        fa, fb = devig_pair(da, db)
        out[(a["name"], float(a["point"]))] = (da, fa)
        out[(b["name"], float(b["point"]))] = (db, fb)
        if m.get("line_type") == "main":
            mains.append(((a["name"], float(a["point"]), fa), (b["name"], float(b["point"]), fb)))
    return out, mains


def _home_main(home, mains):
    for side_a, side_b in mains:
        hs = side_a if side_a[0] == home else side_b if side_b[0] == home else None
        if hs:
            return hs[1], hs[2]  # (home line, fair home cover prob)
    return None


def _anchor(home, books):
    """Main line to centre the model on: Pinnacle, else the most common main
    line across other books (fair prob averaged). Returns (line, fair, source)."""
    if SHARP_BOOK in books:
        hm = _home_main(home, books[SHARP_BOOK][1])
        if hm:
            return hm[0], hm[1], "Pinnacle"
    mains = [hm for key, (_, ms) in books.items() if key not in NOT_PRICES | {BET_BOOK}
             for hm in [_home_main(home, ms)] if hm]
    if len(mains) < MIN_CONSENSUS_BOOKS:
        return None
    line = Counter(l for l, _ in mains).most_common(1)[0][0]
    fair = sum(f for l, f in mains if l == line) / sum(1 for l, _ in mains if l == line)
    return line, fair, f"{len(mains)}-book consensus"


def _ladder(home, books):
    """Every other book's devigged home-cover price across its alt ladder."""
    return [(key, point, fp) for key, (pairs, _) in books.items()
            if key not in NOT_PRICES | {BET_BOOK}
            for (team, point), (_, fp) in pairs.items()
            if team == home and 0.015 < fp < 0.985]


def _fit_model(home, books):
    """Centre the margin model on the anchor main line, then pick the tail
    width (SD) that best reproduces other books' alt-ladder prices. Falls back
    to an assumed SD when there isn't enough ladder data.
    Returns (model, home_main_line, anchor_source, tails_source)."""
    anc = _anchor(home, books)
    if not anc:
        return None, None, None, None
    line, fair, source = anc
    pts = _ladder(home, books)
    if len(pts) < MIN_LADDER_POINTS or len({k for k, _, _ in pts}) < 2:
        m = MarginModel(sd=DEFAULT_SD)
        m.fit_to_main(line, fair)
        return m, line, source, f"assumed SD {DEFAULT_SD:g}"
    best = None
    for sd in SD_GRID:
        m = MarginModel(sd=sd)
        m.fit_to_main(line, fair)
        err = 0.0
        for _, pt, fp in pts:
            w, p = m.home_cover(pt)
            err += (w / (1 - p) - fp) ** 2  # books' devigged prices exclude pushes
        if best is None or err < best[0]:
            best = (err, sd, m)
    n_books = len({k for k, _, _ in pts})
    return best[2], line, source, f"market-fit SD {best[1]:g} ({len(pts)} prices, {n_books} books)"


def scan_event(ev_meta, odds):
    home, away = ev_meta["home_team"], ev_meta["away_team"]
    books = {bk["key"]: _pairs(bk) for bk in odds.get("bookmakers", [])
             if not bk.get("pregame_only")}
    stats = {"fd_prices": 0, "in_range": 0, "no_fair": 0, "neg_ev": 0}
    game = f"{away} @ {home}"
    if BET_BOOK not in books:
        return [], stats, {"game": game, "main": None, "anchor": "no FanDuel prices", "tails": "—"}
    model, home_main, anchor, tails = _fit_model(home, books)
    info = {"game": game, "main": home_main, "home": home,
            "anchor": anchor or "no reliable main line", "tails": tails or "—"}
    cands = []
    for (team, point), (dec, _) in books[BET_BOOK][0].items():
        stats["fd_prices"] += 1
        if not (ODDS_MIN <= dec <= ODDS_MAX):
            continue
        if model is None:
            stats["no_fair"] += 1
            continue
        main_for_team = home_main if team == home else -home_main
        room = point - main_for_team
        if not (ROOM_MIN <= room <= ROOM_MAX):
            continue
        stats["in_range"] += 1
        win, push = model.home_cover(point) if team == home else model.away_cover(point)
        # Other books' devigged price at the exact same line (excludes pushes).
        cons = [p for key, (pairs, _) in books.items()
                if key not in NOT_PRICES | {BET_BOOK} and (team, point) in pairs
                for p in [pairs[(team, point)][1]]]
        cons_win = median(cons) * (1 - push) if len(cons) >= MIN_CONSENSUS_BOOKS else None
        fair_win = min(win, cons_win) if cons_win is not None else win  # conservative
        e = ev(fair_win, push, dec)
        if e <= 0 or (cons_win is None and e < 0.01):
            stats["neg_ev"] += 1
            continue
        opp = away if team == home else home
        cands.append({
            "game": game, "start": ev_meta["commence_time"], "tails": tails,
            "bet": f"{team} {point:+g}", "team": team, "opponent": opp,
            "odds": round(dec, 3), "main_line": main_for_team, "room": room,
            "model_win": win, "cons_win": cons_win, "n_books": len(cons),
            "push": push, "fair_win": fair_win, "breakeven": 1 / dec,
            "ev": e, "anchor": anchor,
        })
    return cands, stats, info


def score(c):
    """Market-only score, 0-100."""
    parts = {
        "Market edge (EV)": 45 * min(c["ev"] / 0.05, 1),
        "Books agreeing": 15 * min(c["n_books"] / 6, 1),
        "Model vs books agree": 0 if c["cons_win"] is None else
            15 * max(0.0, 1 - abs(c["model_win"] - c["cons_win"]) / 0.04),
        "Fair-line quality": (5 if c["anchor"] == "Pinnacle" else 0)
            + (5 if c["tails"].startswith("market-fit") else 0),
        "Points of room": 15 * min(max((c["room"] - ROOM_MIN) / (ROOM_MAX - ROOM_MIN), 0), 1),
    }
    return round(sum(parts.values())), parts


def run(client, now=None):
    now = now or datetime.now(timezone.utc)
    events = [e for e in client.events(SPORT)
              if not e.get("live") and now < _parse_time(e["commence_time"]) <= now + timedelta(hours=24)]
    all_c, totals = [], {"games": len(events), "fd_prices": 0, "in_range": 0, "no_fair": 0,
                         "neg_ev": 0, "game_info": []}
    for e in events:
        cands, st, info = scan_event(e, client.odds(SPORT, e["id"], MARKETS))
        totals["game_info"].append(info)
        all_c += cands
        for k, v in st.items():
            totals[k] += v
    for c in all_c:
        c["score"], c["parts"] = score(c)
    all_c.sort(key=lambda c: c["score"], reverse=True)
    return all_c, totals


# ---- scanner/report.py ----
TZ = ZoneInfo("America/Toronto")
pct = lambda x: f"{x * 100:.1f}%"


def _local(iso):
    from datetime import datetime
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(TZ).strftime("%-I:%M%p")


def markdown(cands, totals, now, min_score, quota):
    t = now.astimezone(TZ)
    L = [f"# 🏀 NBA Alt-Spread Scan — {t:%a %b %-d} · {t:%-I:%M%p} ET", "",
         f"*Games: {totals['games']} · FanDuel spread prices: {totals['fd_prices']} · "
         f"In odds + room range: {totals['in_range']} · Positive EV: {len(cands)} · "
         f"PropLine quota left: {quota}*", ""]
    pick = next((c for c in cands if c["score"] >= min_score), None)
    if not pick:
        L += ["## ⏸️ Sit out today — no bet cleared the bar", "",
              f"Best score was {cands[0]['score']}/100 (need {min_score})." if cands else
              "No FanDuel alt spread was +EV in range.", ""]
    else:
        L += ["## ✅ Today's pick", "", "| | |", "|---|---|",
              f"| Bet | **{pick['bet']}** vs {pick['opponent']} |",
              f"| Game | {pick['game']} · {_local(pick['start'])} ET |",
              f"| FanDuel odds | **{pick['odds']:.2f}** |",
              f"| Main line → room | {pick['main_line']:+g} → **{pick['room']:g} pts** of room |",
              f"| Fair win probability | {pct(pick['fair_win'])} |",
              f"| Break-even | {pct(pick['breakeven'])} |",
              f"| EV | **+{pct(pick['ev'])}** |",
              f"| Score | **{pick['score']}/100** |", "",
              "| Factor | Points | Detail |", "|---|---|---|"]
        detail = {
            "Market edge (EV)": f"+{pct(pick['ev'])} expected return",
            "Books agreeing": f"{pick['n_books']} books price this exact line",
            "Model vs books agree": "no other books at this line" if pick["cons_win"] is None
                else f"model {pct(pick['model_win'])} vs books {pct(pick['cons_win'])}",
            "Fair-line quality": f"centre: {pick['anchor']} · tails: {pick['tails']}",
            "Points of room": f"{pick['room']:g} pts beyond main line",
        }
        for k, v in pick["parts"].items():
            L.append(f"| {k} | {v:.0f} | {detail[k]} |")
        L.append("")
    safest = max(cands, key=lambda c: c["fair_win"]) if cands else None
    if safest and safest is not pick:
        L += [f"**Safest +EV alternative:** {safest['bet']} at {safest['odds']:.2f} "
              f"({pct(safest['fair_win'])} fair, +{pct(safest['ev'])} EV, score {safest['score']})", ""]
    rest = [c for c in cands if c is not pick][:8]
    if rest:
        picked_games = {pick["game"]} if pick else set()
        L += ["## Runners-up" if pick else "## Best candidates (below the bar)", "",
              "| # | Bet | Odds | Room | Fair | EV | Score | Flag |",
              "|---|---|---|---|---|---|---|---|"]
        for i, c in enumerate(rest, 2 if pick else 1):
            flags = []
            if c["game"] in picked_games:
                flags.append("same game as pick")
            if c["cons_win"] is None:
                flags.append("model only")
            if not c["tails"].startswith("market-fit"):
                flags.append("assumed tails")
            L.append(f"| {i} | {c['bet']} | {c['odds']:.2f} | {c['room']:g} | {pct(c['fair_win'])} | "
                     f"+{pct(c['ev'])} | {c['score']} | {', '.join(flags) or '—'} |")
        L.append("")
    L += [f"**Skipped:** {totals['neg_ev']} in range but not +EV · {totals['no_fair']} with no reliable fair line", ""]
    if totals.get("game_info"):
        L += ["## Games", "", "| Game | Home main line | Centre | Tail width |", "|---|---|---|---|"]
        for g in totals["game_info"]:
            main = f"{g['main']:+g}" if g.get("main") is not None else "—"
            L.append(f"| {g['game']} | {main} | {g['anchor']} | {g['tails']} |")
        L.append("")
    L += ["<sub>Market-only scoring. Tail width = how far final margins typically land from the "
          "expected margin, fitted to other books' alt ladders (higher = more blowouts priced in). "
          "Injuries, rest and blowout risk not yet scored — check the news before betting.</sub>"]
    return "\n".join(L) + "\n", pick


FIELDS = ["scan_time", "game", "start", "bet", "odds", "main_line", "room", "model_win",
          "cons_win", "n_books", "push", "fair_win", "breakeven", "ev", "anchor", "tails", "score"]


def write_csv(path, rows, now, append=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not (append and os.path.exists(path))
    with open(path, "a" if append else "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            row = {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
            w.writerow({**row, "scan_time": now.isoformat(timespec="seconds")})


# ---- scan.py ----
def main():
    now = datetime.now(timezone.utc)
    client = PropLine()
    cands, totals = run(client, now)
    quota = f"{client.requests_left} of {client.daily_limit}"
    md, pick = markdown(cands, totals, now, MIN_SCORE, quota)
    local = now.astimezone(TZ)
    os.makedirs("reports", exist_ok=True)
    for path in ("reports/latest.md", f"reports/{local:%Y-%m-%d}.md"):
        with open(path, "w") as f:
            f.write(md)
    write_csv(f"data/scans/{local:%Y-%m-%d_%H%M}.csv", cands, now)
    if pick:
        write_csv("data/picks.csv", [pick], now, append=True)
    print(md)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(md)


if __name__ == "__main__":
    main()
