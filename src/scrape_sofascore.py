"""
scrape_sofascore.py
-------------------
Scrape l'API Sofascore (non-officielle) via Playwright.

Données pré-match :
  - Absents confirmés / douteux par position (GK absent = signal under)
  - Cotes ouverture + fermeture → mouvement de ligne (sharp money)
  - Marchés : 1X2, BTTS, over/under 0.5→4.5, asian handicap, corners, cards
  - Crowd votes : 1X2, BTTS, first scorer

Données post-match (pour features rolling) :
  - Stats complètes : xG, shots, possession, passes, duels, pressing, GK
  - Graph momentum minute par minute (92 pts)

Usage :
    python scrape_sofascore.py --league E1 --season 23/24 --mode historical
    python scrape_sofascore.py --event-id 16117275 --mode prematch
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

_stealth = Stealth()

# Sofascore tournament IDs
TOURNAMENT_IDS = {
    "E0":  17,    # Premier League
    "E1":  18,    # Championship
    "F1":  34,    # Ligue 1
    "F2":  182,   # Ligue 2
    "D1":  35,    # Bundesliga
    "D2":  36,    # 2. Bundesliga
    "SP1": 8,     # La Liga
    "SP2": 54,    # Segunda
    "I1":  23,    # Serie A
    "I2":  53,    # Serie B
}

BASE_URL = "https://api.sofascore.com/api/v1"
SLEEP    = 0.8


# ─── Playwright helper ─────────────────────────────────────────

_DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _make_browser(playwright):
    return playwright.chromium.launch(headless=True)


def _make_page(browser, extra_headers: dict | None = None):
    """Crée une page avec stealth + headers anti-bot."""
    headers = {
        "User-Agent": _DEFAULT_UA,
        "Referer":    "https://www.sofascore.com/",
        "Origin":     "https://www.sofascore.com",
    }
    if extra_headers:
        headers.update(extra_headers)
    page = browser.new_page(extra_http_headers=headers)
    _stealth.apply_stealth_sync(page)
    return page


def _warmup(page) -> None:
    """Navigue sur sofascore.com pour établir les cookies de session."""
    try:
        page.goto("https://www.sofascore.com", wait_until="domcontentloaded", timeout=20000)
        time.sleep(2)
    except Exception:
        pass


def _fetch(page, endpoint: str) -> dict | None:
    time.sleep(SLEEP)
    url = f"{BASE_URL}/{endpoint}"
    # Méthode 1 : fetch() JS depuis le contexte du browser (cookies natifs)
    try:
        result = page.evaluate(f"""
            async () => {{
                try {{
                    const r = await fetch('{url}', {{
                        headers: {{
                            'Accept': 'application/json, text/plain, */*',
                            'Referer': 'https://www.sofascore.com/'
                        }}
                    }});
                    if (!r.ok) return null;
                    return await r.json();
                }} catch(e) {{ return null; }}
            }}
        """)
        if result:
            return result
    except Exception:
        pass
    # Méthode 2 : page.goto direct (fallback — marche pour la plupart des endpoints)
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        raw = page.locator("pre").inner_text(timeout=4000)
        return json.loads(raw)
    except Exception:
        return None


# ─── Helpers parsing ───────────────────────────────────────────

def _frac_to_dec(fv: str) -> float | None:
    """Convertit '7/5' → 2.4, ou '2' → 3.0."""
    if not fv:
        return None
    try:
        if "/" in str(fv):
            n, d = str(fv).split("/")
            return round(int(n) / int(d) + 1, 3)
        return round(float(fv) + 1, 3)
    except Exception:
        return None


def _implied_prob(dec_odds: float | None) -> float | None:
    if not dec_odds or dec_odds <= 1:
        return None
    return round(1 / dec_odds, 4)


# ─── 1. Saisons disponibles ────────────────────────────────────

def get_seasons(tournament_id: int, page, retries: int = 3, retry_sleep: int = 15) -> list[dict]:
    """Fetche les saisons. Warm-up sur sofascore.com puis fetch JS (cookies natifs)."""
    _warmup(page)
    for attempt in range(retries):
        data = _fetch(page, f"unique-tournament/{tournament_id}/seasons")
        if data:
            seasons = data.get("seasons", [])
            if seasons:
                return seasons
        if attempt < retries - 1:
            print(f"    ↻ get_seasons vide (tentative {attempt+1}/{retries}), "
                  f"retry dans {retry_sleep}s...")
            time.sleep(retry_sleep)
    return []


# ─── 2. Tous les matchs d'une saison ──────────────────────────

def get_all_events_by_date(tournament_id: int, season_year: int, page) -> list[dict]:
    """
    Fallback : reconstruit la liste des matchs en itérant sur
    sport/football/scheduled-events/{date} jour par jour.
    Contourne le blocage IP sur unique-tournament/.../events/...
    ~365 appels × SLEEP ≈ 5 min/saison.
    """
    from datetime import date, timedelta

    start   = date(season_year, 7, 1)
    end     = date(season_year + 1, 6, 30)
    current = start
    all_events: list[dict] = []
    seen_ids: set[int]     = set()

    total_days = (end - start).days + 1
    print(f"    ↻ fallback date-by-date ({total_days} jours)...", flush=True)

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        data = _fetch(page, f"sport/football/scheduled-events/{date_str}")
        if data:
            for ev in data.get("events", []):
                tid = (ev.get("tournament", {})
                          .get("uniqueTournament", {})
                          .get("id"))
                eid = ev.get("id")
                if tid == tournament_id and eid and eid not in seen_ids:
                    seen_ids.add(eid)
                    all_events.append(ev)
        current += timedelta(days=1)

    finished = [e for e in all_events if e.get("status", {}).get("type") == "finished"]
    print(f"    ↻ date-by-date : {len(finished)} matchs terminés trouvés", flush=True)
    return finished


def get_all_events(
    tournament_id: int,
    season_id: int,
    page,
    season_year: int | None = None,
) -> list[dict]:
    """
    Pagine sur /events/last/{page} jusqu'à hasNextPage=False.
    Si l'endpoint retourne 0 résultats (blocage IP) et que season_year
    est fourni, bascule sur get_all_events_by_date().
    """
    all_events = []
    p = 0
    while True:
        data = _fetch(page, f"unique-tournament/{tournament_id}/season/{season_id}/events/last/{p}")
        if not data:
            break
        events = data.get("events", [])
        all_events.extend(events)
        if not data.get("hasNextPage", False):
            break
        p += 1
        if p > 30:  # safety
            break

    finished = [e for e in all_events if e.get("status", {}).get("type") == "finished"]

    if not finished and season_year is not None:
        finished = get_all_events_by_date(tournament_id, season_year, page)

    return finished


# ─── 3. Stats post-match ───────────────────────────────────────

def get_match_stats(event_id: int, page) -> dict:
    """
    Retourne les stats complètes d'un match terminé.
    Clés principales : xG, shots, possession, passes, duels, GK, momentum.
    """
    stats_raw = _fetch(page, f"event/{event_id}/statistics")
    graph_raw = _fetch(page, f"event/{event_id}/graph")

    result = {"event_id": event_id}

    # Statistics
    if stats_raw:
        for period in stats_raw.get("statistics", []):
            if period.get("period") != "ALL":
                continue
            for group in period.get("groups", []):
                for item in group.get("statisticsItems", []):
                    name = (item.get("name", "")
                            .lower()
                            .replace(" ", "_")
                            .replace("/", "_"))
                    result[f"h_{name}"] = item.get("home")
                    result[f"a_{name}"] = item.get("away")

    # Graph momentum
    if graph_raw:
        pts  = graph_raw.get("graphPoints", [])
        vals = [p["value"] for p in pts]
        if vals:
            result["h_momentum_avg"]  = round(sum(v for v in vals if v > 0) / len(vals), 2)
            result["a_momentum_avg"]  = round(sum(abs(v) for v in vals if v < 0) / len(vals), 2)
            result["momentum_std"]    = round(pd.Series(vals).std(), 2)
            late = [p["value"] for p in pts if p["minute"] >= 75]
            result["momentum_late_75"] = round(sum(late) / len(late), 2) if late else 0

    return result


# ─── 3b. Incidents (cartons) ───────────────────────────────────

def get_match_incidents(event_id: int, page) -> list[dict]:
    """
    Retourne les cartons rouges (rouge direct + rouge après jaune) du match.
    Chaque dict : event_id, player_id, player_name, player_pos,
                  is_home, time, reason, rescinded, incidentClass
    Utilisé pour reconstruire les suspensions historiques.
    """
    data = _fetch(page, f"event/{event_id}/incidents")
    if not data:
        return []

    rows = []
    for inc in data.get("incidents", []):
        if inc.get("incidentType") != "card":
            continue
        cls = inc.get("incidentClass", "")
        if cls not in ("red", "yellowRed"):
            continue
        player = inc.get("player") or {}
        rows.append({
            "event_id":    event_id,
            "player_id":   player.get("id"),
            "player_name": player.get("name") or player.get("shortName"),
            "player_pos":  player.get("position"),
            "is_home":     inc.get("isHome", True),
            "time":        inc.get("time"),
            "reason":      inc.get("reason", ""),
            "rescinded":   bool(inc.get("rescinded", False)),
            "card_class":  cls,
        })
    return rows


# ─── 4. Données pré-match ──────────────────────────────────────


def get_prematch_features(event_id: int, page) -> dict:
    """
    Tout ce qui est disponible AVANT le coup d'envoi :
    absents, cotes (opening+closing), crowd votes,
    H2H agrégat, pregame form Sofascore, forme récente par équipe.
    """
    lu_raw    = _fetch(page, f"event/{event_id}/lineups")
    odds_raw  = _fetch(page, f"event/{event_id}/odds/1/all")
    votes_raw = _fetch(page, f"event/{event_id}/votes")
    ev_raw    = _fetch(page, f"event/{event_id}")

    result = {"event_id": event_id}

    # ── Infos match + IDs équipes ──
    ev = ev_raw.get("event", {}) if ev_raw else {}
    home_id = ev.get("homeTeam", {}).get("id")
    away_id = ev.get("awayTeam", {}).get("id")
    result["home_team_sf_id"]  = home_id
    result["away_team_sf_id"]  = away_id
    result["home_team_name"]   = ev.get("homeTeam", {}).get("name")
    result["away_team_name"]   = ev.get("awayTeam", {}).get("name")
    result["start_timestamp"]  = ev.get("startTimestamp")
    result["round"]            = ev.get("roundInfo", {}).get("round")

    # ── Arbitre (stats carrière — toujours disponibles pré-match) ──
    ref = ev.get("referee", {})
    ref_games = ref.get("games", 0)
    if ref_games > 10:
        result["ref_sf_id"]         = ref.get("id")
        result["ref_sf_yc_per_game"] = round(ref.get("yellowCards", 0) / ref_games, 3)
        result["ref_sf_rc_per_game"] = round(ref.get("redCards", 0) / ref_games, 3)
        result["ref_sf_games"]       = ref_games

    # ── Stade ──
    venue = ev.get("venue", {})
    cap = venue.get("stadium", {}).get("capacity") or venue.get("capacity")
    if cap:
        result["venue_capacity"] = cap

    # ── Manager IDs ──
    result["home_manager_id"] = ev.get("homeTeam", {}).get("manager", {}).get("id")
    result["away_manager_id"] = ev.get("awayTeam", {}).get("manager", {}).get("id")

    # ── H2H agrégat ──
    h2h_raw = _fetch(page, f"event/{event_id}/h2h")
    if h2h_raw:
        td = h2h_raw.get("teamDuel", {})
        hw = td.get("homeWins", 0)
        aw = td.get("awayWins", 0)
        dr = td.get("draws",    0)
        total_h2h = hw + aw + dr
        if total_h2h > 0:
            result["h2h_home_win_rate"] = round(hw / total_h2h, 3)
            result["h2h_away_win_rate"] = round(aw / total_h2h, 3)
            result["h2h_draw_rate"]     = round(dr / total_h2h, 3)
            result["h2h_n_matches"]     = total_h2h

    # ── Pregame form Sofascore ──
    pgf_raw = _fetch(page, f"event/{event_id}/pregame-form")
    if pgf_raw:
        for side, prefix in [("homeTeam", "h"), ("awayTeam", "a")]:
            fd = pgf_raw.get(side, {})
            # avgRating = note moyenne des joueurs sur les derniers matchs
            try:
                result[f"{prefix}_sf_avg_rating"] = float(fd.get("avgRating") or 0) or None
            except (TypeError, ValueError):
                pass
            # position au classement live
            result[f"{prefix}_sf_position"] = fd.get("position")
            # points cumulés (value = pts dans la table)
            try:
                result[f"{prefix}_sf_pts"] = int(fd.get("value") or 0) or None
            except (TypeError, ValueError):
                pass
            # Forme derniers 5 : encode W=3, D=1, L=0
            form_list = fd.get("form", [])
            if form_list:
                pts_map = {"W": 3, "D": 1, "L": 0}
                pts = [pts_map.get(r, 1) for r in form_list[:5]]
                result[f"{prefix}_sf_form_pts5"] = sum(pts)
                result[f"{prefix}_sf_form_str"]  = "".join(form_list[:5])   # ex: "DWLDL"
                # Wins et under dans la fenêtre (approximé sans scores)
                result[f"{prefix}_sf_wins5"] = sum(1 for r in form_list[:5] if r == "W")

    # ── Valeurs marchandes squad (/team/{id}/players) ──
    # Utilisable en historique : on scrape chaque match à sa date dans
    # build_historical_dataset, donc les valeurs reflètent l'époque du match.
    # Pour usage live seulement → valeurs actuelles.
    missing_ids = {}
    if lu_raw:
        for side, prefix in [("home", "h"), ("away", "a")]:
            missing_ids[prefix] = {
                m.get("player", {}).get("id")
                for m in lu_raw.get(side, {}).get("missingPlayers", [])
                if m.get("type") == "missing"
            }

    for tid, prefix in [(home_id, "h"), (away_id, "a")]:
        if not tid:
            continue
        pl_raw = _fetch(page, f"team/{tid}/players")
        if not pl_raw or "error" in pl_raw:
            continue
        players = pl_raw.get("players", [])
        total_val, avail_val = 0.0, 0.0
        ages = []
        for p in players:
            player = p.get("player", {})
            raw = player.get("proposedMarketValueRaw", {})
            val = raw.get("value") or 0
            pid = player.get("id")
            dob = player.get("dateOfBirthTimestamp")
            total_val += val
            if pid not in missing_ids.get(prefix, set()):
                avail_val += val
            if dob:
                import datetime
                age = (datetime.date.today() - datetime.date.fromtimestamp(dob)).days / 365.25
                ages.append(age)

        if total_val > 0:
            result[f"{prefix}_squad_value"]       = round(total_val / 1e6, 2)   # en M€
            result[f"{prefix}_squad_value_avail"] = round(avail_val / 1e6, 2)
            result[f"{prefix}_squad_value_ratio"] = round(avail_val / total_val, 4)
        if ages:
            result[f"{prefix}_squad_avg_age"] = round(sum(ages) / len(ages), 1)

    # Note : /team/{id}/events/last/0 est un endpoint live → leakage pour
    # les données historiques. La forme rolling est déjà gérée proprement
    # dans features.py via _compute_team_rolling (under_rate_3/5/10, form_5...).

    # ── Absents ──
    if lu_raw:
        for side, prefix in [("home", "h"), ("away", "a")]:
            missing = lu_raw.get(side, {}).get("missingPlayers", [])
            result[f"{prefix}_missing_total"]    = len(missing)
            result[f"{prefix}_missing_confirmed"] = sum(1 for m in missing if m.get("type") == "missing")
            result[f"{prefix}_missing_doubtful"]  = sum(1 for m in missing if m.get("type") == "doubtful")
            # Par position
            for pos, label in [("G", "gk"), ("D", "def"), ("M", "mid"), ("F", "fwd")]:
                result[f"{prefix}_missing_{label}"] = sum(
                    1 for m in missing
                    if m.get("player", {}).get("position") == pos
                    and m.get("type") == "missing"
                )
            # IDs des absents (pour enrichissement valeur marchande TM)
            result[f"{prefix}_missing_player_ids"] = [
                m.get("player", {}).get("id") for m in missing
            ]

    # ── Cotes + mouvement de ligne ──
    if odds_raw:
        markets = odds_raw.get("markets", [])
        seen_goals_lines = []

        for mkt in markets:
            name = mkt.get("marketName", "")
            choices = mkt.get("choices", [])

            if name == "Full time" and len(choices) == 3:
                for c in choices:
                    side = {"1": "h", "X": "d", "2": "a"}.get(c["name"])
                    if side:
                        curr = _frac_to_dec(c.get("fractionalValue"))
                        init = _frac_to_dec(c.get("initialFractionalValue"))
                        result[f"odds_{side}_curr"]  = curr
                        result[f"odds_{side}_open"]  = init
                        result[f"odds_{side}_move"]  = round(curr / init, 4) if curr and init else None
                        # change: +1 drift (longer), -1 shortened
                        result[f"odds_{side}_change"] = c.get("change")

            elif name == "Both teams to score" and len(choices) == 2:
                for c in choices:
                    k = "yes" if c["name"] == "Yes" else "no"
                    result[f"btts_{k}_odds"] = _frac_to_dec(c.get("fractionalValue"))
                result["btts_implied_yes"] = _implied_prob(result.get("btts_yes_odds"))

            elif name == "Match goals":
                # Sur/sous : 0.5, 1.5, 2.5, 3.5, 4.5...
                for c in choices:
                    dec = _frac_to_dec(c.get("fractionalValue"))
                    init = _frac_to_dec(c.get("initialFractionalValue"))
                    if c["name"] == "Over":
                        seen_goals_lines.append({
                            "over_odds": dec,
                            "over_open": init,
                            "under_odds": None,
                            "under_open": None,
                        })
                    elif c["name"] == "Under" and seen_goals_lines:
                        seen_goals_lines[-1]["under_odds"] = dec
                        seen_goals_lines[-1]["under_open"] = init

                # Associe les lignes Over/Under (0.5→4.5)
                lines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
                for i, entry in enumerate(seen_goals_lines):
                    if i < len(lines):
                        line = lines[i]
                        ls = str(line).replace(".", "")
                        result[f"under_{ls}_odds"]    = entry.get("under_odds")
                        result[f"under_{ls}_open"]    = entry.get("under_open")
                        result[f"under_{ls}_implied"] = _implied_prob(entry.get("under_odds"))
                        result[f"under_{ls}_move"]    = (
                            round(entry["under_odds"] / entry["under_open"], 4)
                            if entry.get("under_odds") and entry.get("under_open") else None
                        )

            elif name == "Asian handicap" and len(choices) >= 2:
                for c in choices:
                    label = "h" if c == choices[0] else "a"
                    result[f"ah_{label}_odds"] = _frac_to_dec(c.get("fractionalValue"))
                    result[f"ah_{label}_open"] = _frac_to_dec(c.get("initialFractionalValue"))

            elif name == "Corners 2-Way" and len(choices) == 2:
                for c in choices:
                    k = "over" if c["name"] == "Over" else "under"
                    result[f"corners_{k}_odds"] = _frac_to_dec(c.get("fractionalValue"))

        # ── Features dérivées des cotes ──
        odds_h = result.get("odds_h_curr")
        odds_d = result.get("odds_d_curr")
        odds_a = result.get("odds_a_curr")
        if odds_h and odds_d and odds_a:
            raw_h = 1 / odds_h
            raw_d = 1 / odds_d
            raw_a = 1 / odds_a
            margin = raw_h + raw_d + raw_a
            result["odds_implied_h"]    = round(raw_h / margin, 4)
            result["odds_implied_d"]    = round(raw_d / margin, 4)  # nul élevé → match ouvert
            result["odds_implied_a"]    = round(raw_a / margin, 4)
            result["odds_implied_delta"] = round(raw_h / margin - raw_a / margin, 4)  # favoris dominance

    # ── Crowd votes ──
    if votes_raw:
        vote = votes_raw.get("vote", {})
        total_1x2 = sum([vote.get("vote1", 0), vote.get("voteX", 0), vote.get("vote2", 0)])
        if total_1x2 > 0:
            result["crowd_h_pct"]  = round(vote.get("vote1", 0) / total_1x2, 4)
            result["crowd_d_pct"]  = round(vote.get("voteX", 0) / total_1x2, 4)
            result["crowd_a_pct"]  = round(vote.get("vote2", 0) / total_1x2, 4)
            result["crowd_votes_n"] = total_1x2

        btts = votes_raw.get("bothTeamsToScoreVote", {})
        total_btts = btts.get("voteYes", 0) + btts.get("voteNo", 0)
        if total_btts > 0:
            result["crowd_btts_yes_pct"] = round(btts.get("voteYes", 0) / total_btts, 4)

    return result


# ─── 5. Pipeline historique ────────────────────────────────────

def build_historical_dataset(
    fd_league: str,
    season_year: int,      # ex: 2023 pour 23/24
    out_dir: Path,
    with_prematch: bool = True,
):
    """
    Scrape tous les matchs d'une saison : stats post-match + features pré-match.
    Sauvegarde deux CSV : match_stats_{league}_{season}.csv et prematch_{league}_{season}.csv
    """
    tournament_id = TOURNAMENT_IDS.get(fd_league)
    if not tournament_id:
        raise ValueError(f"Ligue inconnue: {fd_league}. Dispo: {list(TOURNAMENT_IDS.keys())}")

    out_dir.mkdir(parents=True, exist_ok=True)
    season_str = f"{season_year % 100:02d}{(season_year+1) % 100:02d}"

    with sync_playwright() as p:
        browser = _make_browser(p)
        page    = _make_page(browser)

        # Trouve le season_id Sofascore
        seasons = get_seasons(tournament_id, page)
        target_season = None
        for s in seasons:
            year_str = s.get("year", "")
            if str(season_year % 100).zfill(2) in year_str:
                target_season = s
                break

        if not target_season:
            print(f"Saison {season_year}/{season_year+1} non trouvée pour {fd_league}")
            print(f"Saisons dispo: {[s.get('year') for s in seasons[:8]]}")
            browser.close()
            return

        season_id = target_season["id"]
        print(f"\n[{fd_league}] Saison {target_season['year']} (id={season_id})")

        # Récupère tous les matchs terminés
        print("  Récupération des matchs...")
        events = get_all_events(tournament_id, season_id, page, season_year=season_year)
        print(f"  {len(events)} matchs terminés")

        # Scrape stats + pré-match
        stats_records    = []
        prematch_records = []

        for i, ev in enumerate(events):
            eid       = ev["id"]
            home_name = ev.get("homeTeam", {}).get("name", "?")
            away_name = ev.get("awayTeam", {}).get("name", "?")
            ts        = ev.get("startTimestamp", 0)
            date_str  = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else "?"

            print(f"  [{i+1:>3}/{len(events)}] {date_str} | {home_name:20} vs {away_name:20}", end=" ", flush=True)

            # Stats post-match
            if ev.get("hasXg"):
                stats = get_match_stats(eid, page)
                stats["date"]      = date_str
                stats["home_team"] = home_name
                stats["away_team"] = away_name
                stats["home_score"] = ev.get("homeScore", {}).get("current")
                stats["away_score"] = ev.get("awayScore", {}).get("current")
                stats_records.append(stats)
                print(f"xG={stats.get('h_expected_goals','?')}|{stats.get('a_expected_goals','?')}", end=" ")

            # Pré-match
            if with_prematch:
                pm = get_prematch_features(eid, page)
                pm["date"]      = date_str
                pm["home_team"] = home_name
                pm["away_team"] = away_name
                pm["home_score"] = ev.get("homeScore", {}).get("current")
                pm["away_score"] = ev.get("awayScore", {}).get("current")
                prematch_records.append(pm)
                n_miss = pm.get("h_missing_total", 0) + pm.get("a_missing_total", 0)
                print(f"absents={n_miss}", end="")

            print()

        browser.close()

    # Sauvegarde
    if stats_records:
        df_stats = pd.DataFrame(stats_records)
        path = out_dir / f"sofascore_stats_{fd_league}_{season_str}.csv"
        df_stats.to_csv(path, index=False)
        print(f"\n→ stats: {len(df_stats)} matchs | {path}")

    if prematch_records:
        df_pm = pd.DataFrame(prematch_records)
        path = out_dir / f"sofascore_prematch_{fd_league}_{season_str}.csv"
        df_pm.to_csv(path, index=False)
        print(f"→ prematch: {len(df_pm)} matchs | {path}")

    print("✅ Done")


# ─── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--league",  default="E1",  help="E0 E1 F1 F2 D1 D2 SP1 SP2 I1 I2")
    ap.add_argument("--season",  type=int, default=2023, help="Année de début (ex: 2023 pour 23/24)")
    ap.add_argument("--out",     default="./sofascore_data")
    ap.add_argument("--no-prematch", action="store_true")
    args = ap.parse_args()

    build_historical_dataset(
        fd_league=args.league,
        season_year=args.season,
        out_dir=Path(args.out),
        with_prematch=not args.no_prematch,
    )
