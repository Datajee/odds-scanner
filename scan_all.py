"""Odds scanner — NBA decision board (single-file build).
Prices every FanDuel bet at 1.05-1.35 across spreads, totals, team totals,
moneylines and player props; shows a low/mid/high EV range and a risk level.
Run: python scan_all.py"""
import csv
import json
import os
import re
import requests
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from math import isclose, sqrt
from scipy.stats import t as student_t, nbinom
from statistics import median
from zoneinfo import ZoneInfo

# ==== scanner/oddsmath.py ====
# ---------- prices ----------
def american_to_decimal(a):
    a = float(a)
    if 1.0 < a < 100.0:  # already decimal (American odds are never between -100 and +100)
        return a
    return 1 + (a / 100 if a > 0 else 100 / abs(a))


def devig_mult(dec_a, dec_b):
    """Fair probabilities for a two-way market: scale implied probs to sum to 1.
    Spreads the vig evenly, which under-rates the favourite on lopsided markets."""
    qa, qb = 1 / dec_a, 1 / dec_b
    return qa / (qa + qb), qb / (qa + qb)


def devig_shin(dec_a, dec_b):
    """Shin's method: assumes books protect against informed money by loading
    more margin on the long shot. Gives the favourite a higher fair probability
    than the multiplicative method on lopsided markets."""
    q = [1 / dec_a, 1 / dec_b]
    Q = sum(q)
    if Q <= 1:  # no vig to remove
        return q[0] / Q, q[1] / Q

    def probs(z):
        return [(sqrt(z * z + 4 * (1 - z) * qi * qi / Q) - z) / (2 * (1 - z)) for qi in q]

    lo, hi = 0.0, 0.5
    for _ in range(60):
        z = (lo + hi) / 2
        if sum(probs(z)) > 1:
            lo = z
        else:
            hi = z
    p = probs((lo + hi) / 2)
    s = sum(p)
    return p[0] / s, p[1] / s


def ev(win, push, dec):
    """Expected return per $1 staked. A push refunds the stake."""
    return win * (dec - 1) - (1 - win - push)


# ---------- discrete models ----------
class _Discrete:
    """Shared logic; subclasses provide _ge(k) = P(X >= k) for integer k."""

    def eq(self, k):
        return self._ge(k) - self._ge(k + 1)

    def cover(self, side, x):
        """(win, push) for a bet that wins when X > x ('gt') or X < x ('lt')."""
        if isclose(x, round(x)):
            k = int(round(x))
            push = self.eq(k)
            win = self._ge(k + 1) if side == "gt" else 1 - self._ge(k)
            return win, push
        k = int(x // 1) + 1  # first integer above x
        return (self._ge(k), 0.0) if side == "gt" else (1 - self._ge(k), 0.0)

    def fair_no_push(self, side, x):
        win, push = self.cover(side, x)
        return win / (1 - push) if push < 1 else 0.0

    def fit_center(self, side, x, fair):
        """Move the centre until the model reproduces `fair` (a no-push
        probability, as books quote) for the bet (side, x). Bisection."""
        lo, hi = self._center_bounds(x)
        for _ in range(50):
            self.center = (lo + hi) / 2
            p_gt = self.fair_no_push("gt", x)
            target = fair if side == "gt" else 1 - fair
            if p_gt < target:
                lo = self.center
            else:
                hi = self.center
        return self


class TModel(_Discrete):
    """X ~ round(center + scale * T(df)). For score margins and totals:
    symmetric with fatter-than-normal tails (blowouts happen more often than a
    bell curve predicts). `width` is the standard deviation in points."""

    def __init__(self, width, df=6, center=0.0):
        self.width, self.df, self.center = width, df, center
        self.scale = width * ((df - 2) / df) ** 0.5

    def _ge(self, k):
        return float(1 - student_t.cdf((k - 0.5 - self.center) / self.scale, self.df))

    def _center_bounds(self, x):
        return x - 80.0, x + 80.0


class NBModel(_Discrete):
    """X ~ NegativeBinomial with mean `center` and dispersion `width` (r).
    For player stat counts: can't go below zero, right-skewed, and spread out
    more than Poisson. Large r -> close to Poisson (rebounds, threes);
    small r -> boom/bust (points for streaky scorers)."""

    def __init__(self, width, center=10.0):
        self.width, self.center = width, center

    def _ge(self, k):
        if k <= 0:
            return 1.0
        r, m = self.width, max(self.center, 1e-6)
        return float(nbinom.sf(k - 1, r, r / (r + m)))

    def _center_bounds(self, x):
        return 0.01, max(4 * x + 10, 20.0)


# ==== scanner/markets.py ====
STAT_NAMES = {
    "player_points": "points", "player_rebounds": "rebounds", "player_assists": "assists",
    "player_threes": "threes", "player_steals": "steals", "player_blocks": "blocks",
    "player_turnovers": "turnovers", "player_points_rebounds_assists": "pts+reb+ast",
    "player_points_rebounds": "pts+reb", "player_points_assists": "pts+ast",
    "player_rebounds_assists": "reb+ast", "player_blocks_steals": "blk+stl",
}
YESNO_PROPS = {"player_double_double": "double-double", "player_triple_double": "triple-double"}
PROP_MARKETS = list(STAT_NAMES) + list(YESNO_PROPS) + ["player_points_10plus", "player_points_15plus"]
ALL_MARKETS = ["h2h", "spreads", "totals"] + PROP_MARKETS
_NPLUS = re.compile(r"^(player_[a-z_]+?)_(\d+)plus$")


def _is_over(name):
    return str(name).strip().lower() in ("over", "yes")


def _whole(v):
    return abs(v - round(v)) < 1e-9


def parse_event(ev_meta, odds):
    home, away = ev_meta["home_team"], ev_meta["away_team"]
    out = []
    for bk in odds.get("bookmakers", []):
        if bk.get("pregame_only"):
            continue
        for m in bk.get("markets", []):
            if m.get("period") or m.get("suspended_at"):
                continue
            rows = _parse_market(m, home, away)
            if not rows:
                continue
            if len(rows) == 2:  # two-sided row: remove the vig against each other
                pm = devig_mult(rows[0]["dec"], rows[1]["dec"])
                ps = devig_shin(rows[0]["dec"], rows[1]["dec"])
                for r, a, b in zip(rows, pm, ps):
                    r["fair_mult"], r["fair_shin"] = a, b
            for r in rows:
                r.setdefault("fair_mult", None)
                r.setdefault("fair_shin", None)
                r["book"] = bk.get("key", "?")
                r["line_type"] = m.get("line_type") or "main"
                out.append(r)
    return out


def _parse_market(m, home, away):
    key, lt = m.get("key"), m.get("line_type")
    rows = []
    for o in m.get("outcomes", []):
        try:
            dec = american_to_decimal(o["price"])
        except (KeyError, TypeError, ValueError):
            return []
        name, pt = o.get("name"), o.get("point")
        r = {"dec": dec}
        if key == "h2h":
            if name not in (home, away):
                return []
            r.update(family="ml", subject=("ml",), side=None, x=None, outcome=name,
                     label=f"{name} moneyline")
        elif key == "spreads":
            if pt is None or name not in (home, away):
                return []
            pt = float(pt)
            gt = name == home  # model X = home score minus away score
            r.update(family="spread", subject=("spread",), side="gt" if gt else "lt",
                     x=-pt if gt else pt, outcome=name, label=f"{name} {pt:+g}")
        elif key == "totals":
            if pt is None:
                return []
            pt, team = float(pt), m.get("team")
            side = "gt" if _is_over(name) else "lt"
            word = "Over" if side == "gt" else "Under"
            if team:
                r.update(family="team_total", subject=("team_total", team), side=side, x=pt,
                         outcome=word, label=f"{team} team total {word.lower()} {pt:g}")
            else:
                r.update(family="total", subject=("total",), side=side, x=pt, outcome=word,
                         label=f"Game total {word.lower()} {pt:g}")
        elif key in YESNO_PROPS:
            player = o.get("description")
            if not player:
                return []
            yn = "Yes" if _is_over(name) else "No"
            r.update(family="yesno", subject=("yesno", key, player), side=None, x=None,
                     outcome=yn, label=f"{player} {YESNO_PROPS[key]}: {yn}")
        else:
            nplus = _NPLUS.match(key or "")
            base = nplus.group(1) if nplus else key
            if base not in STAT_NAMES:
                return []
            player = o.get("description")
            if not player:
                return []
            side = "gt" if _is_over(name) else "lt"
            stat = STAT_NAMES[base]
            if nplus:  # e.g. player_points_15plus: Yes = 15 or more
                n = int(nplus.group(2))
                x, milestone = n - 0.5, n
            elif pt is None:
                return []
            elif lt == "milestone" and _whole(float(pt)):  # "N+" rung = N or more
                x, milestone = float(pt) - 0.5, int(round(float(pt)))
            else:
                x, milestone = float(pt), None
            if milestone is not None:
                label = f"{player} {milestone}+ {stat}" if side == "gt" else f"{player} under {milestone} {stat}"
            else:
                label = f"{player} {'over' if side == 'gt' else 'under'} {x:g} {stat}"
            r.update(family="prop", subject=("prop", base, player), side=side, x=x,
                     outcome=name, label=label, player=player)
        rows.append(r)
    return rows


# ==== scanner/propline.py ====
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


# ==== scanner/strategies/nba_board.py ====
SPORT = "basketball_nba"
BET_BOOK = "fanduel"
SHARP_BOOK = "pinnacle"
ODDS_MIN, ODDS_MAX = 1.05, 1.35
SPREAD_ROOM = (10, 25)          # your alt-spread preference: points beyond the main line
MIN_CONSENSUS_BOOKS = 3
LEVELS = [(0.93, "L1"), (0.88, "L2"), (0.80, "L3"), (0.0, "L4")]
# Pick'em / DFS apps don't post real two-way prices; never use them as a fair line.
NOT_PRICES = {"prizepicks", "underdog", "sleeper", "dabble", "pick6", "dst_propbuilder",
              "thrillzz", "sportzino", "courtside", "rebet"}

MODELS = {  # family: (model factory, width grid searched, default width, min books for anchor)
    "spread": (lambda w: TModel(w, df=6), [x / 2 for x in range(16, 41)], 12.5, 3),
    "total": (lambda w: TModel(w, df=8), [float(x) for x in range(12, 31)], 18.0, 3),
    "team_total": (lambda w: TModel(w, df=8), [x / 2 for x in range(12, 37)], 10.5, 3),
    "prop": (lambda w: NBModel(w), [1.0, 1.5, 2, 3, 4, 6, 8, 12, 16, 25, 40, 60, 100, 200], 20.0, 2),
}
MARKET_NAMES = {"spread": "Spread", "total": "Game total", "team_total": "Team total",
                "prop": "Player prop", "ml": "Moneyline", "yesno": "Player yes/no"}
MIN_LADDER_POINTS, MIN_LADDER_BOOKS = 6, 2


def _parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def level_of(p):
    return next(name for floor, name in LEVELS if p >= floor)


# ---------- fitting a line-market subject ----------
def _gt_view(line, method):
    """(x, P(X > x | no push)) for a paired line, whichever side it is."""
    f = line[f"fair_{method}"]
    return line["x"], (f if line["side"] == "gt" else 1 - f)


def _anchor(lines, family, method):
    paired = [l for l in lines if l[f"fair_{method}"] is not None and l["line_type"] == "main"]
    sharp = [l for l in paired if l["book"] == SHARP_BOOK]
    if sharp:
        x, f = min((_gt_view(l, method) for l in sharp), key=lambda v: abs(v[1] - 0.5))
        return x, f, "Pinnacle"
    views = {}
    for l in paired:  # one view per book
        views.setdefault(l["book"], _gt_view(l, method))
    if len(views) < MODELS[family][3]:
        return None
    x = Counter(v[0] for v in views.values()).most_common(1)[0][0]
    fs = [v[1] for v in views.values() if v[0] == x]
    return x, sum(fs) / len(fs), f"{len(views)}-book consensus"


def fit_subject(lines, family):
    """Fit models for one subject from other books' lines.
    Returns None if there's no usable main line."""
    make, grid, default_w, _ = MODELS[family]
    out = {}
    for method in ("shin", "mult"):
        anc = _anchor(lines, family, method)
        if not anc:
            return None
        x0, f0, source = anc
        ladder = [_gt_view(l, method) for l in lines if l[f"fair_{method}"] is not None]
        ladder = [(x, f) for x, f in ladder if 0.015 < f < 0.985]
        n_books = len({l["book"] for l in lines if l[f"fair_{method}"] is not None})
        # Average books at the same line: same least-squares fit, far fewer model calls.
        by_x = defaultdict(list)
        for x, f in ladder:
            by_x[x].append(f)
        grouped = [(x, sum(fs) / len(fs), len(fs)) for x, fs in by_x.items()]
        if len(ladder) >= MIN_LADDER_POINTS and n_books >= MIN_LADDER_BOOKS:
            best = None
            for w in grid:
                m = make(w).fit_center("gt", x0, f0)
                err = sum(n * (m.fair_no_push("gt", x) - f) ** 2 for x, f, n in grouped)
                if best is None or err < best[0]:
                    best = (err, m)
            model, fitted = best[1], True
        else:
            model, fitted = make(default_w).fit_center("gt", x0, f0), False
        out[method] = model
        if method == "shin":
            out.update(anchor_x=x0, source=source, fitted=fitted, n_ladder=len(ladder), n_books=n_books,
                       default=make(default_w).fit_center("gt", x0, f0))
    return out


def _width_desc(fit, family):
    w = fit["shin"].width
    unit = "dispersion" if family == "prop" else "pts"
    if fit["fitted"]:
        return f"fitted {w:g} {unit} ({fit['n_ladder']} prices, {fit['n_books']} books)"
    return f"typical {w:g} {unit} (thin ladder)"


# ---------- pricing ----------
def _estimates_line(fd, fit, others):
    side, x = fd["side"], fd["x"]
    est = {}
    win, push = fit["shin"].cover(side, x)
    est["mid"] = win
    est["mult"] = fit["mult"].cover(side, x)[0]
    est["default"] = fit["default"].cover(side, x)[0]
    same = [l["fair_shin"] for l in others if l["side"] == side and l["fair_shin"] is not None
            and abs(l["x"] - x) < 1e-9]
    n = len(same)
    if n >= MIN_CONSENSUS_BOOKS:
        est["books"] = median(same) * (1 - push)
    return est, push, n


def _estimates_binary(fd, others, spread_fit, home):
    est, n = {}, 0
    sharp = [l for l in others if l["book"] == SHARP_BOOK and l["outcome"] == fd["outcome"]
             and l["fair_shin"] is not None]
    if sharp:
        est["mid"] = sharp[0]["fair_shin"]
        est["mult"] = sharp[0]["fair_mult"]
    same = [l["fair_shin"] for l in others if l["outcome"] == fd["outcome"] and l["fair_shin"] is not None]
    n = len(same)
    if n >= MIN_CONSENSUS_BOOKS:
        est["books"] = median(same)
    if fd["family"] == "ml" and spread_fit:
        p_home = spread_fit["shin"].fair_no_push("gt", 0)  # NBA has no ties
        est["spread model"] = p_home if fd["outcome"] == home else 1 - p_home
    if "mid" not in est:
        for k in ("books", "spread model"):
            if k in est:
                est["mid"] = est[k]
                break
    return est, 0.0, n


def scan_event(ev_meta, odds):
    home, away = ev_meta["home_team"], ev_meta["away_team"]
    game = f"{away} @ {home}"
    lines = parse_event(ev_meta, odds)
    others = defaultdict(list)
    for l in lines:
        if l["book"] not in NOT_PRICES | {BET_BOOK}:
            others[l["subject"]].append(l)
    fits = {}

    def get_fit(subject, family):
        if subject not in fits:
            fits[subject] = fit_subject(others.get(subject, []), family)
        return fits[subject]

    stats = Counter()
    cands = []
    for fd in (l for l in lines if l["book"] == BET_BOOK):
        fam = fd["family"]
        stats[f"{fam}:fd"] += 1
        if not (ODDS_MIN <= fd["dec"] <= ODDS_MAX):
            continue
        room, main_disp, width = None, "—", "—"
        if fam in MODELS:
            fit = get_fit(fd["subject"], fam)
            if not fit:
                stats[f"{fam}:no_fair"] += 1
                continue
            x0 = fit["anchor_x"]
            room = (x0 - fd["x"]) if fd["side"] == "gt" else (fd["x"] - x0)
            if fam == "spread":
                if not (SPREAD_ROOM[0] <= room <= SPREAD_ROOM[1]):
                    continue
                main_disp = f"{(-x0 if fd['outcome'] == home else x0):+g}"
            else:
                main_disp = f"{x0:g}"
            est, push, n_books = _estimates_line(fd, fit, others[fd["subject"]])
            width, source = _width_desc(fit, fam), fit["source"]
        else:
            spread_fit = get_fit(("spread",), "spread") if fam == "ml" else None
            est, push, n_books = _estimates_binary(fd, others[fd["subject"]], spread_fit, home)
            if "mid" not in est:
                stats[f"{fam}:no_fair"] += 1
                continue
            source = "Pinnacle" if "mult" in est else "other books" if "books" in est else "spread model"
        stats[f"{fam}:priced"] += 1
        wins = list(est.values())
        lo, hi, mid = min(wins), max(wins), est["mid"]
        dec = fd["dec"]
        flags = []
        if fam in MODELS and n_books < MIN_CONSENSUS_BOOKS:
            flags.append("model only")
        if fam in MODELS and not fit["fitted"]:
            flags.append("thin ladder")
        if ev(hi, push, dec) - ev(lo, push, dec) > 0.04:
            flags.append("wide range")
        cands.append({
            "game": game, "start": ev_meta["commence_time"], "market": MARKET_NAMES[fam],
            "family": fam, "bet": fd["label"], "odds": round(dec, 3), "main": main_disp,
            "room": room, "win_low": lo, "win_mid": mid, "win_high": hi, "push": push,
            "breakeven": 1 / dec, "ev_low": ev(lo, push, dec), "ev_mid": ev(mid, push, dec),
            "ev_high": ev(hi, push, dec), "level": level_of(mid), "n_books": n_books,
            "estimates": {k: round(v, 4) for k, v in est.items()}, "source": source,
            "width": width, "flags": flags,
        })
    return cands, stats


def run(client, now=None):
    now = now or datetime.now(timezone.utc)
    events = [e for e in client.events(SPORT)
              if not e.get("live") and now < _parse_time(e["commence_time"]) <= now + timedelta(hours=24)]
    all_c, stats = [], Counter()
    for e in events:
        cands, st = scan_event(e, client.odds(SPORT, e["id"], ALL_MARKETS))
        all_c += cands
        stats.update(st)
    all_c.sort(key=lambda c: c["ev_mid"], reverse=True)
    return all_c, {"games": len(events), "stats": stats}


# ==== scanner/report.py ====
TZ = ZoneInfo("America/Toronto")
LEVEL_INFO = {
    "L1": ("🟦 L1 · Very safe", "≥ 93% win"),
    "L2": ("🟩 L2 · Safe", "88–93% win"),
    "L3": ("🟨 L3 · Moderate", "80–88% win"),
    "L4": ("🟥 L4 · Aggressive", "< 80% win"),
}
ROWS_PER_LEVEL = 8


def pct(x, sign=False):
    return f"{x * 100:+.1f}%" if sign else f"{x * 100:.1f}%"


def ev_dot(e):
    return "🟢" if e >= 0 else "🟡" if e >= -0.02 else "🔴"


def short_game(game):
    away, home = game.split(" @ ")
    return f"{away.split()[-1]}@{home.split()[-1]}"


def _local(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(TZ).strftime("%-I:%M%p")


def markdown(cands, info, now, quota):
    t = now.astimezone(TZ)
    L = [f"# 🏀 NBA Decision Board — {t:%a %b %-d} · {t:%-I:%M%p} ET", "",
         f"*{info['games']} games · {len(cands)} FanDuel bets priced between 1.05 and 1.35 · "
         f"PropLine quota left: {quota}*", "",
         "**EV key:** 🟢 positive · 🟡 within 2% of fair · 🔴 worse than −2% "
         "(EV shown as low / **mid** / high)", ""]
    by_level = {lv: [c for c in cands if c["level"] == lv] for lv in LEVEL_INFO}

    # Overview
    L += ["## Overview", "", "| Level | Win chance | Bets | 🟢 | 🟡 | 🔴 | Best bet (mid EV) |",
          "|---|---|---|---|---|---|---|"]
    for lv, (name, rng) in LEVEL_INFO.items():
        cs = by_level[lv]
        dots = Counter(ev_dot(c["ev_mid"]) for c in cs)
        best = f"{cs[0]['bet']} {cs[0]['odds']:.2f} ({pct(cs[0]['ev_mid'], True)})" if cs else "—"
        L.append(f"| {name} | {rng} | {len(cs)} | {dots['🟢']} | {dots['🟡']} | {dots['🔴']} | {best} |")
    L.append("")

    # Per-level tables
    for lv, (name, rng) in LEVEL_INFO.items():
        cs = by_level[lv][:ROWS_PER_LEVEL]
        if not cs:
            continue
        games = Counter(c["game"] for c in cs)
        L += [f"## {name} ({rng})", "",
              "| | Bet | Game | Odds | Win % | EV low / mid / high | Notes |",
              "|---|---|---|---|---|---|---|"]
        for c in cs:
            notes = list(c["flags"])
            if games[c["game"]] > 1:
                notes.append("same game as another row")
            if c["room"] is not None:
                notes.insert(0, f"{c['room']:g} room vs {c['main']}")
            L.append(f"| {ev_dot(c['ev_mid'])} | {c['bet']} | {short_game(c['game'])} {_local(c['start'])} | "
                     f"{c['odds']:.2f} | {pct(c['win_mid'])} | {pct(c['ev_low'], True)} / "
                     f"**{pct(c['ev_mid'], True)}** / {pct(c['ev_high'], True)} | {', '.join(notes) or '—'} |")
        more = len(by_level[lv]) - len(cs)
        if more > 0:
            L.append(f"\n*+{more} more {lv} bets in the CSV log.*")
        L.append("")

    # Coverage
    st = info["stats"]
    fams = ["spread", "total", "team_total", "ml", "prop", "yesno"]
    names = {"spread": "Alt spreads", "total": "Game totals", "team_total": "Team totals",
             "ml": "Moneylines", "prop": "Player props", "yesno": "Player yes/no"}
    L += ["## Coverage", "", "| Market | FanDuel prices | Priced in range | No fair line |", "|---|---|---|---|"]
    for f in fams:
        L.append(f"| {names[f]} | {st.get(f + ':fd', 0)} | {st.get(f + ':priced', 0)} | {st.get(f + ':no_fair', 0)} |")
    L += ["", "<sub>Win % = mid estimate: model centred on Pinnacle's main line, width fitted to other "
          "books' alt ladders, Shin vig removal. Low/high = worst/best of: multiplicative vig removal, "
          "a typical width, and other books at the exact line. Alt spreads limited to 10–25 pts of room. "
          "Injuries/rest not scored — check news, especially for player props.</sub>"]
    return "\n".join(L) + "\n"


FIELDS = ["scan_time", "level", "market", "game", "start", "bet", "odds", "main", "room",
          "win_low", "win_mid", "win_high", "push", "breakeven", "ev_low", "ev_mid", "ev_high",
          "n_books", "source", "width", "flags", "estimates"]


def write_csv(path, rows, now):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
            row["flags"] = "; ".join(r["flags"])
            row["estimates"] = json.dumps(r["estimates"])
            w.writerow({**row, "scan_time": now.isoformat(timespec="seconds")})


# ==== scan.py ====
def main():
    now = datetime.now(timezone.utc)
    client = PropLine()
    cands, info = run(client, now)
    md = markdown(cands, info, now, f"{client.requests_left} of {client.daily_limit}")
    local = now.astimezone(TZ)
    os.makedirs("reports", exist_ok=True)
    for path in ("reports/latest.md", f"reports/{local:%Y-%m-%d}.md"):
        with open(path, "w") as f:
            f.write(md)
    write_csv(f"data/scans/{local:%Y-%m-%d_%H%M}.csv", cands, now)
    print(md)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(md)


if __name__ == "__main__":
    main()
