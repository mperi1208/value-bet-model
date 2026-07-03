"""
scrape_matchendirect_xg.py
--------------------------
Scrape les données post-match depuis matchendirect.fr pour E1 (Championship)
et F2 (Ligue 2), saisons 2018/19 à 2023/24.

Par match on récupère en 3 requêtes :
  1. ?p=stats        → toutes les stats (xG, tirs, passes, duels, etc.)
  2. page principale → événements (buts, cartons, remplacements + minute)
  3. ?p=compositions → XI titulaire des deux équipes + formation

Outputs (dans data/matchendirect/) :
  stats_{league}_{ss}.csv   — une ligne par match, colonnes larges
  events_{league}_{ss}.csv  — une ligne par événement (buts/cartons/subs)
  lineups_{league}_{ss}.csv — une ligne par joueur titulaire

Usage :
    python scrape_matchendirect_xg.py
    python scrape_matchendirect_xg.py --leagues E1 --seasons 2021 2022
    python scrape_matchendirect_xg.py --leagues F2 --no-events --no-lineups
"""

import argparse
import re
import time
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL = "https://www.matchendirect.fr"
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9",
}

COMP_IDS = {
    "E1": "7ntvbsyq31jnzoqoa8850b9b8",
    "F2": "4w7x0s5gfs5abasphlha5de8k",
}

SEASONS_RANGE = {"start": 2018, "end": 2023}

SLEEP_DATE  = 0.35
SLEEP_MATCH = 0.5   # par requête (3 requêtes/match → ~1.5s/match)

# Correspondance nom français → colonne anglaise
STAT_MAP = {
    "possession":                            "possession",
    "tirs":                                  "shots",
    "tirs cadrés":                           "shots_on_target",
    "tirs non cadrés":                       "shots_off_target",
    "tirs bloqués":                          "shots_blocked",
    "poteau":                                "woodwork",
    "occasions manquées":                    "big_chances_missed",
    "xg (buts attendus)":                    "xg",
    "ballons touches dans la surface adverse": "touches_in_box",
    "corners":                               "corners",
    "hors-jeu":                              "offsides",
    "rentree de touche":                     "throw_ins",
    "fautes":                                "fouls",
    "cartons janues":                        "yellow_cards",
    "cartons jaunes":                        "yellow_cards",
    "cartons rouges":                        "red_cards",
    "passes":                                "passes",
    "passes réussis":                        "passes_accurate",
    "passes réussis (%)":                    "passes_pct",
    "centres":                               "crosses",
    "centres réussis":                       "crosses_accurate",
    "duels réussis":                         "duels_won",
    "tacles réussis":                        "tackles_won",
    "duels aériens réussis":                 "aerial_duels_won",
    "dribblés réussis":                      "dribbles_won",
    "interceptions":                         "interceptions",
    "dégagements":                           "clearances",
}

# Types d'événements (classes CSS matchendirect)
EVENT_TYPES = {
    "G":   "goal",
    "PG":  "penalty_goal",
    "OG":  "own_goal",
    "MPG": "missed_penalty",
    "PSG": "penalty_saved",
    "PSM": "penalty_saved",
    "YC":  "yellow_card",
    "Y2C": "second_yellow",
    "RC":  "red_card",
    "S":   "substitution",
}


# ── URL helpers ───────────────────────────────────────────────────────────────

def season_dates(season_year: int, league: str):
    start = date(season_year, 7 if league == "F2" else 8, 1)
    end   = date(season_year + 1, 6, 30)
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def match_id_from_path(path: str) -> str:
    slug = path.split("/live-score/")[1].replace(".html", "")
    return slug.split("_")[-1] if "_" in slug else slug


def slug_teams(path: str) -> str:
    slug = path.split("/live-score/")[1].replace(".html", "")
    return slug.rsplit("_", 1)[0] if "_" in slug else slug


# ── Fetchers ──────────────────────────────────────────────────────────────────

def get(session: requests.Session, url: str) -> BeautifulSoup | None:
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "html.parser")
    except Exception:
        pass
    return None


def fetch_match_urls(session: requests.Session, d: date, comp_id: str) -> list[str]:
    url  = f"{BASE_URL}/resultat-foot-{d.strftime('%d-%m-%Y')}/"
    soup = get(session, url)
    if not soup:
        return []
    for ph in soup.find_all("div", class_="panel-heading"):
        if comp_id in str(ph):
            panel = ph.find_parent("div", class_="panel")
            if panel:
                body = panel.find("div", class_="panel-body")
                if body:
                    return list(set(
                        a["href"].split("?")[0]
                        for a in body.find_all("a", href=re.compile(r"/live-score/"))
                        if a.get("href", "").startswith("/live-score/")
                    ))
    return []


# ── Stats parser (?p=stats) ───────────────────────────────────────────────────

def parse_stats(soup: BeautifulSoup) -> dict:
    """Extrait toutes les stats du tab1 (match complet). Retourne {} si absent."""
    tab1 = soup.find(id="tab1ListeScore")
    if not tab1:
        return {}

    result = {}

    # ── Possession (structure spéciale : div.progressBarFirst + aria-valuenow) ──
    poss_bar = tab1.find("div", class_="progressBarFirst")
    if poss_bar:
        team_a = poss_bar.find("div", class_="teamA")
        team_b = poss_bar.find("div", class_="teamB")
        try:
            result["possession_home"] = float(team_a["aria-valuenow"]) if team_a else None
            result["possession_away"] = float(team_b["aria-valuenow"]) if team_b else None
        except (KeyError, TypeError, ValueError):
            result["possession_home"] = None
            result["possession_away"] = None

    # ── Autres stats (div.progressBar classique avec progressBarValue1/2) ──────
    for bar in tab1.find_all("div", class_="progressBar"):
        h5 = bar.find("h5", class_="progressHeaderTitle")
        if not h5:
            continue
        name = h5.get_text(strip=True).lower()
        col  = STAT_MAP.get(name)
        if col is None or col == "possession":  # possession déjà traitée
            continue
        v1 = bar.find("span", class_="progressBarValue1")
        v2 = bar.find("span", class_="progressBarValue2")
        try:
            result[f"{col}_home"] = float(v1.get_text(strip=True)) if v1 else None
            result[f"{col}_away"] = float(v2.get_text(strip=True)) if v2 else None
        except (ValueError, TypeError):
            result[f"{col}_home"] = None
            result[f"{col}_away"] = None

    # Valider : si xG absent ou tous deux = 0 → données probablement vides
    if result.get("xg_home") == 0.0 and result.get("xg_away") == 0.0:
        result["xg_home"] = None
        result["xg_away"] = None

    return result


# ── Events parser (page principale) ──────────────────────────────────────────

def parse_events(soup: BeautifulSoup, match_id: str) -> list[dict]:
    """Extrait buts, cartons, remplacements depuis la page résumé."""
    table = soup.find(id="match_evenement")
    if not table:
        return []

    events = []
    for row in table.find_all("tr"):
        tds = row.find_all("td")
        if not tds:
            continue

        # Détecter le type via classe CSS
        icon = row.find(class_=re.compile(r"ico_evenement[A-Z]"))
        if not icon:
            continue
        classes   = " ".join(icon.get("class", []))
        type_keys = re.findall(r"ico_evenement([A-Z2]+)", classes)
        if not type_keys:
            continue
        ev_type = EVENT_TYPES.get(type_keys[0], type_keys[0])

        # Structure HTML : 3 colonnes [home_player | minute | away_player]
        # td[0] = joueur côté home (vide si événement away)
        # td[1] = minute
        # td[2] = joueur côté away (vide si événement home)
        if len(tds) < 2:
            continue

        td0 = tds[0].get_text(separator=" ", strip=True)
        td1 = tds[1].get_text(separator=" ", strip=True) if len(tds) > 1 else ""
        td2 = tds[2].get_text(separator=" ", strip=True) if len(tds) > 2 else ""
        full = f"{td0} | {td1} | {td2}"

        # La minute est toujours dans td[1]
        minute_m = re.search(r"(\d+)(?:'?\+(\d+))?", td1)
        minute = None
        if minute_m:
            base  = int(minute_m.group(1))
            extra = int(minute_m.group(2)) if minute_m.group(2) else 0
            minute = base + extra

        # Côté : td[2] vide → home, td[0] vide → away
        side = "home" if td2 == "" else "away"

        # Joueurs (liens /joueur/)
        player_links = row.find_all("a", href=re.compile(r"/joueur/"))
        players = [a.get_text(strip=True) for a in player_links]

        events.append({
            "match_id":   match_id,
            "event_type": ev_type,
            "minute":     minute,
            "side":       side,
            "player1":    players[0] if len(players) > 0 else (td0 or td2 or None),
            "player2":    players[1] if len(players) > 1 else None,
            "raw":        full[:120],
        })

    return events


# ── Lineups parser (?p=compositions) ─────────────────────────────────────────

def parse_lineups(soup: BeautifulSoup, match_id: str) -> list[dict]:
    """Extrait les XI titulaires des deux équipes."""
    rows = []
    for tab_id, side in [("tab1ListeStat", "home"), ("tab2ListeStat", "away")]:
        tab = soup.find(id=tab_id)
        if not tab:
            continue
        for player_div in tab.find_all("div", class_="player"):
            link = player_div.find("a", href=re.compile(r"/joueur/"))
            num  = player_div.find("b")
            name_div = player_div.find(class_="player-name")
            if not (link and name_div):
                continue
            num_val = num.get_text(strip=True) if num else ""
            full_name = name_div.get_text(strip=True)
            # Enlever le numéro du nom
            name_clean = re.sub(r"^\s*" + re.escape(num_val) + r"\s*", "", full_name).strip()
            rows.append({
                "match_id":     match_id,
                "side":         side,
                "shirt_number": num_val,
                "player_name":  name_clean,
                "player_slug":  link["href"].split("/joueur/")[1].replace(".html", ""),
            })

    # Formation (chercher pattern X-X-X dans le HTML du tab)
    compo_tab = soup.find(id="tab_compositions")
    formation_home = formation_away = None
    if compo_tab:
        formations = re.findall(r"\b(\d-\d-\d(?:-\d)?)\b", compo_tab.get_text())
        if len(formations) >= 1:
            formation_home = formations[0]
        if len(formations) >= 2:
            formation_away = formations[1]

    for r in rows:
        r["formation"] = formation_home if r["side"] == "home" else formation_away

    return rows


# ── Season scraper ────────────────────────────────────────────────────────────

def scrape_season(
    league: str, season: int, out_dir: Path, session: requests.Session,
    do_events: bool = True, do_lineups: bool = True,
) -> None:
    comp_id    = COMP_IDS[league]
    ss         = f"{str(season)[2:]}{str(season+1)[2:]}"
    stats_path   = out_dir / f"stats_{league}_{ss}.csv"
    events_path  = out_dir / f"events_{league}_{ss}.csv"
    lineups_path = out_dir / f"lineups_{league}_{ss}.csv"

    # Charger les match_ids déjà traités (basé sur stats)
    done_ids: set[str] = set()
    existing_stats   : list[dict] = []
    existing_events  : list[dict] = []
    existing_lineups : list[dict] = []

    if stats_path.exists():
        ex = pd.read_csv(stats_path)
        done_ids = set(ex["match_id"].dropna().astype(str))
        existing_stats = ex.to_dict("records")
        print(f"  {league}_{season}: {len(done_ids)} matchs déjà en cache", flush=True)
    if do_events and events_path.exists():
        existing_events = pd.read_csv(events_path).to_dict("records")
    if do_lineups and lineups_path.exists():
        existing_lineups = pd.read_csv(lineups_path).to_dict("records")

    new_stats   : list[dict] = []
    new_events  : list[dict] = []
    new_lineups : list[dict] = []
    days_with_matches = 0

    for d in season_dates(season, league):
        time.sleep(SLEEP_DATE + random.uniform(0, 0.15))

        urls = fetch_match_urls(session, d, comp_id)
        if not urls:
            continue
        days_with_matches += 1

        for path in urls:
            mid = match_id_from_path(path)
            if mid in done_ids:
                continue

            # ── 1. Stats (?p=stats) ────────────────────────────────────────
            time.sleep(SLEEP_MATCH + random.uniform(0, 0.2))
            stats_soup = get(session, f"{BASE_URL}{path}?p=stats")
            stats_data = parse_stats(stats_soup) if stats_soup else {}

            if not stats_data:
                continue  # pas de stats → skip

            stat_row = {
                "match_id": mid,
                "slug":     slug_teams(path),
                "date":     d.isoformat(),
                "league":   league,
                "season":   season,
                **stats_data,
            }
            new_stats.append(stat_row)
            done_ids.add(mid)

            # ── 2. Events (page principale) ────────────────────────────────
            if do_events:
                time.sleep(SLEEP_MATCH + random.uniform(0, 0.2))
                main_soup = get(session, f"{BASE_URL}{path}")
                if main_soup:
                    new_events.extend(parse_events(main_soup, mid))

            # ── 3. Lineups (?p=compositions) ───────────────────────────────
            if do_lineups:
                time.sleep(SLEEP_MATCH + random.uniform(0, 0.2))
                compo_soup = get(session, f"{BASE_URL}{path}?p=compositions")
                if compo_soup:
                    new_lineups.extend(parse_lineups(compo_soup, mid))

        # Checkpoint toutes les 10 journées
        if days_with_matches % 10 == 0 and new_stats:
            _save(stats_path,   existing_stats   + new_stats,   "match_id")
            if do_events:
                _save(events_path,  existing_events  + new_events,  None)
            if do_lineups:
                _save(lineups_path, existing_lineups + new_lineups, None)
            print(f"    checkpoint — {len(new_stats)} nouveaux matchs", flush=True)

    # Sauvegarde finale
    all_stats = existing_stats + new_stats
    if not all_stats:
        print(f"  ⚠ {league}_{season}: aucune donnée récupérée", flush=True)
        return

    _save(stats_path,   all_stats,                           "match_id", sort="date")
    if do_events:
        _save(events_path,  existing_events  + new_events,  None)
    if do_lineups:
        _save(lineups_path, existing_lineups + new_lineups, None)

    print(
        f"  ✓ {league}_{season}: {len(new_stats)} nouveaux, {len(all_stats)} total "
        f"| {len(new_events)} events | {len(new_lineups)} lineups",
        flush=True,
    )


def _save(path: Path, rows: list[dict], dedup_col: str | None, sort: str | None = None):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if dedup_col and dedup_col in df.columns:
        df = df.drop_duplicates(subset=[dedup_col])
    if sort and sort in df.columns:
        df = df.sort_values(sort)
    df.to_csv(path, index=False)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(leagues, seasons, out_dir, do_events=True, do_lineups=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HTTP_HEADERS)

    for league in leagues:
        if league not in COMP_IDS:
            print(f"⚠ Ligue inconnue : {league}")
            continue
        print(f"\n=== {league} ===", flush=True)
        for season in seasons:
            print(f"  Saison {season}/{season+1} ...", flush=True)
            scrape_season(league, season, out_dir, session, do_events, do_lineups)
            time.sleep(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues",     nargs="+", default=["E1", "F2"])
    parser.add_argument("--seasons",     nargs="+", type=int,
                        default=list(range(SEASONS_RANGE["start"], SEASONS_RANGE["end"] + 1)))
    parser.add_argument("--out",         default="data/matchendirect")
    parser.add_argument("--no-events",   action="store_true")
    parser.add_argument("--no-lineups",  action="store_true")
    args = parser.parse_args()

    run(
        leagues    = args.leagues,
        seasons    = args.seasons,
        out_dir    = Path(args.out),
        do_events  = not args.no_events,
        do_lineups = not args.no_lineups,
    )
