"""
scrape_transfermarkt.py
-----------------------
Scrape Transfermarkt pour enrichir le pipeline value-bet.

Données collectées :
  1. Valeur marchande effectif  (requests — statique)
  2. Blessures + suspensions    (requests — statique)
  3. Stats buts/passes/cartons  (Playwright — JS)
  4. Prochain match (timestamp) (ceapi JSON)

Features produites :
  - squad_value_total, squad_avg_age
  - pct_value_available, n_injured, top5_injured
  - n_suspended
  - h_goals_top_scorer, h_assists_top_creator (meilleur joueur absent ?)
  - squad_instability (transferts entrée+sortie / taille effectif)
  - days_to_next_match

Usage :
    python scrape_transfermarkt.py --league E1 --seasons 2021 2022 2023
    python scrape_transfermarkt.py --league E1 --seasons 2023 --injuries --stats
"""

import argparse
import re
import time
import json
from pathlib import Path
from datetime import datetime, date

import requests
import pandas as pd
from bs4 import BeautifulSoup

# Playwright optionnel (stats seulement)
try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# football-data.co.uk → Transfermarkt
LEAGUE_MAP = {
    "E1": "GB2", "E0": "GB1",
    "F2": "FR2", "F1": "FR1",
    "D1": "L1",  "D2": "L2",
    "SP1": "ES1", "SP2": "ES2",
    "I1": "IT1",  "I2": "IT2",
}

SLEEP = 1.5  # secondes entre requêtes


# ─── Helpers ───────────────────────────────────────────────────

def _get(url):
    time.sleep(SLEEP)
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r


def _parse_value(text: str) -> int:
    text = text.strip().replace("€", "").replace(",", "")
    try:
        if "m" in text:
            return int(float(text.replace("m", "")) * 1_000_000)
        elif "k" in text:
            return int(float(text.replace("k", "")) * 1_000)
    except Exception:
        pass
    return 0


def _parse_date(text: str):
    for fmt in ("%d/%m/%Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except Exception:
            continue
    return None


# ─── 1. Équipes d'une ligue ────────────────────────────────────

def get_league_teams(tm_league_code: str, season: int | None = None) -> list[dict]:
    if season is not None:
        url = f"https://www.transfermarkt.com/_/startseite/wettbewerb/{tm_league_code}/plus/?saison_id={season}"
    else:
        url = f"https://www.transfermarkt.com/_/startseite/wettbewerb/{tm_league_code}"
    soup = BeautifulSoup(_get(url).content, "html.parser")
    teams = []
    for row in soup.find_all("tr", class_=["odd", "even"]):
        a = row.find("a", href=re.compile(r'/verein/\d+'))
        if not a:
            continue
        m = re.search(r'/verein/(\d+)', a["href"])
        if not m:
            continue
        teams.append({
            "team_id":      m.group(1),
            "team_name_tm": a.get("title", a.get_text(strip=True)),
            "slug":         a["href"].split("/")[1],
        })
    return teams


def get_all_historical_teams(tm_league_code: str, seasons: list[int]) -> list[dict]:
    """Collecte les équipes de chaque saison et déduplique par team_id."""
    seen = {}
    for s in seasons:
        print(f"  Equipes saison {s}...", end=" ", flush=True)
        try:
            for t in get_league_teams(tm_league_code, season=s):
                if t["team_id"] not in seen:
                    seen[t["team_id"]] = t
            print(f"{len(seen)} total")
        except Exception as e:
            print(f"⚠ {e}")
    return list(seen.values())


# ─── 2. Valeur effectif + âge moyen + instabilité transferts ──

def get_squad_info(team_id: str, season: int) -> dict | None:
    """Valeur totale, âge moyen, liste joueurs avec slug (nécessaire pour blessures)."""
    url = f"https://www.transfermarkt.com/_/kader/verein/{team_id}/saison_id/{season}/plus/1"
    try:
        soup = BeautifulSoup(_get(url).content, "html.parser")
    except Exception as e:
        print(f"    ⚠ squad_info({team_id}, {season}): {e}")
        return None

    table = soup.find("table", class_=re.compile(r'items'))
    if not table:
        return None

    players, ages = [], []
    for row in table.find_all("tr", class_=["odd", "even"]):
        name_tag = row.find("a", href=re.compile(r'/spieler/\d+'))
        if not name_tag:
            continue
        href = name_tag["href"]
        player_id = re.search(r'/spieler/(\d+)', href).group(1)
        slug = href.split("/")[1]
        name = name_tag.get_text(strip=True)

        # Âge
        dob_cell = row.find("td", class_="zentriert")
        dob_text = dob_cell.get_text(strip=True) if dob_cell else ""
        age_m = re.search(r'\((\d+)\)', dob_text)
        age = int(age_m.group(1)) if age_m else None
        if age:
            ages.append(age)

        val_td = row.find("td", class_="rechts hauptlink")
        val = _parse_value(val_td.get_text(strip=True)) if val_td else 0
        players.append({"name": name, "player_id": player_id, "slug": slug, "value": val})

    # Âge moyen via regex dans le HTML brut (TM l'affiche en bas)
    html_text = soup.get_text()
    avg_age_m = re.search(r'[Aa]verage age[:\s]+(\d+[\.,]\d+)', html_text)
    avg_age = float(avg_age_m.group(1).replace(',', '.')) if avg_age_m else (
        round(sum(ages) / len(ages), 1) if ages else None
    )

    # Instabilité transferts
    n_arr, n_dep = _get_transfer_activity(team_id, season)

    total_value = sum(p["value"] for p in players)
    return {
        "team_id":          team_id,
        "season":           season,
        "total_value":      total_value,
        "n_players":        len(players),
        "avg_age":          avg_age,
        "n_arrivals":       n_arr,
        "n_departures":     n_dep,
        "squad_instability": (n_arr + n_dep) / max(len(players), 1),
        "players":          players,
    }


def _get_transfer_activity(team_id: str, season: int) -> tuple[int, int]:
    url = f"https://www.transfermarkt.com/_/transfers/verein/{team_id}/saison_id/{season}"
    try:
        soup = BeautifulSoup(_get(url).content, "html.parser")
    except Exception:
        return 0, 0

    tables = soup.find_all("table", class_=re.compile(r'items'))
    arrivals = departures = 0
    for table in tables:
        box = table.find_parent("div", class_=re.compile(r'box'))
        title = box.find("h2") if box else None
        title_text = title.get_text(strip=True).lower() if title else ""
        n = len(table.find_all("tr", class_=["odd", "even"]))
        if "arrival" in title_text:
            arrivals = n
        elif "departure" in title_text:
            departures = n
    return arrivals, departures


# ─── 3. Blessures par joueur ───────────────────────────────────

def get_player_injuries(player_id: str, slug: str) -> list[dict]:
    url = f"https://www.transfermarkt.com/{slug}/verletzungen/spieler/{player_id}"
    try:
        soup = BeautifulSoup(_get(url).content, "html.parser")
    except Exception:
        return []

    table = soup.find("table", class_=re.compile(r'items'))
    if not table:
        return []

    injuries = []
    for row in table.find_all("tr", class_=["odd", "even"]):
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) < 5:
            continue
        d_from  = _parse_date(cells[2])
        d_until = _parse_date(cells[3])
        if not d_from:
            continue
        injuries.append({
            "player_id": player_id,
            "injury":    cells[1],
            "date_from": d_from,
            "date_until": d_until,
            "days":      cells[4],
        })
    return injuries


# ─── 3b. Suspensions historiques par joueur ───────────────────

def get_player_suspensions(player_id: str, slug: str) -> list[dict]:
    """
    Scrape /sperren/spieler/{id} : historique des suspensions avec dates.
    Retourne une liste de périodes (date_from, date_until, reason, matches).
    """
    url = f"https://www.transfermarkt.com/{slug}/sperren/spieler/{player_id}"
    try:
        soup = BeautifulSoup(_get(url).content, "html.parser")
    except Exception:
        return []

    table = soup.find("table", class_=re.compile(r'items'))
    if not table:
        return []

    suspensions = []
    for row in table.find_all("tr", class_=["odd", "even"]):
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) < 4:
            continue
        # Structure typique : season | competition | reason | matches | date_from | date_until
        # On cherche les deux dernières cellules date-like
        d_from = d_until = None
        for cell in cells:
            parsed = _parse_date(cell)
            if parsed and d_from is None:
                d_from = parsed
            elif parsed and d_from is not None:
                d_until = parsed
        if not d_from:
            continue
        suspensions.append({
            "player_id":  player_id,
            "reason":     cells[2] if len(cells) > 2 else None,
            "date_from":  d_from,
            "date_until": d_until or d_from,  # suspension 1 match → même date
        })
    return suspensions


# ─── 4. Suspensions + absences actuelles ──────────────────────

def get_team_absences(team_id: str, slug: str) -> dict:
    """
    Page 'sperrenundverletzungen' : blessures actives + suspensions + risque suspension.
    Retourne les joueurs absents avec leur motif et date de retour estimée.
    """
    url = f"https://www.transfermarkt.com/{slug}/sperrenundverletzungen/verein/{team_id}"
    try:
        soup = BeautifulSoup(_get(url).content, "html.parser")
    except Exception:
        return {"injured": [], "suspended": [], "suspension_risk": []}

    result = {"injured": [], "suspended": [], "suspension_risk": []}

    for table in soup.find_all("table", class_=re.compile(r'items')):
        box   = table.find_parent("div", class_=re.compile(r'box'))
        title = box.find("h2") if box else None
        title_text = title.get_text(strip=True).lower() if title else ""
        rows  = table.find_all("tr", class_=["odd", "even"])

        for row in rows:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            name_a = row.find("a", href=re.compile(r'/spieler/\d+'))
            if not name_a:
                continue
            pid_m = re.search(r'/spieler/(\d+)', name_a["href"])
            entry = {
                "player_id":   pid_m.group(1) if pid_m else None,
                "player_name": name_a.get_text(strip=True),
            }

            if "risk" in title_text or "yellow" in title_text:
                entry["yellow_cards"] = cells[2] if len(cells) > 2 else None
                result["suspension_risk"].append(entry)
            else:
                # Injurés ou suspendus
                entry["reason"]          = cells[2] if len(cells) > 2 else None
                entry["since"]           = _parse_date(cells[3]) if len(cells) > 3 else None
                entry["expected_return"] = cells[4] if len(cells) > 4 else None
                if "suspension" in title_text:
                    result["suspended"].append(entry)
                else:
                    result["injured"].append(entry)

    return result


# ─── 5. Stats buts/passes/cartons (Playwright) ────────────────

def get_player_stats(team_id: str, tm_league_code: str, season: int) -> pd.DataFrame:
    if not HAS_PLAYWRIGHT:
        print("  ⚠ Playwright non installé — pip install playwright && playwright install chromium")
        return pd.DataFrame()

    url = (f"https://www.transfermarkt.com/_/leistungsdaten/verein/{team_id}"
           f"/reldata/{tm_league_code}%26{season}/plus/1")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(extra_http_headers={
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": HEADERS["User-Agent"],
        })
        page.goto(url, wait_until="networkidle", timeout=30000)
        content = page.content()
        browser.close()

    soup = BeautifulSoup(content, "html.parser")
    tables = [t for t in soup.find_all("table") if len(t.find_all("tr")) > 3]
    if not tables:
        return pd.DataFrame()

    def cell_val(cells, idx):
        if idx >= len(cells):
            return None
        txt = cells[idx].get_text(strip=True).replace("'", "").replace(",", ".")
        try:
            return float(txt) if "." in txt else (int(txt) if txt.lstrip("-").isdigit() else None)
        except Exception:
            return None

    players = []
    for row in tables[0].find_all("tr", class_=["odd", "even"]):
        cells = row.find_all("td")
        name_a = row.find("a", href=lambda h: h and "/spieler/" in h)
        if not name_a or len(cells) < 14:
            continue
        pid_m = re.search(r'/spieler/(\d+)', name_a["href"])
        players.append({
            "player_id": pid_m.group(1) if pid_m else None,
            "slug":      name_a["href"].split("/")[1],
            "name":      name_a.get_text(strip=True),
            "apps":      cell_val(cells, 8),
            "goals":     cell_val(cells, 9),
            "assists":   cell_val(cells, 10),
            "yellow":    cell_val(cells, 11),
            "red":       cell_val(cells, 12),
            "minutes":   cell_val(cells, 17),
        })

    return pd.DataFrame(players)


# ─── 6. Prochain match (timestamp) ────────────────────────────

def get_next_match_date(team_id: str) -> date | None:
    url = f"https://www.transfermarkt.com/ceapi/nextMatches/team/{team_id}"
    try:
        r = requests.get(url, headers={**HEADERS, "X-Requested-With": "XMLHttpRequest"}, timeout=10)
        matches = r.json().get("matches", [])
        fixtures = [m for m in matches if m.get("match", {}).get("state") == "Fixture"]
        if fixtures:
            ts = fixtures[0]["match"]["time"]
            return datetime.utcfromtimestamp(ts).date()
    except Exception:
        pass
    return None


# ─── 7. Feature engineering ────────────────────────────────────

def compute_availability_features(
    squad: dict,
    injuries: list[dict],
    match_date,
    suspensions: list[dict] | None = None,
    stats_df: pd.DataFrame | None = None,
) -> dict:
    """
    Pour une date de match, calcule toutes les features de disponibilité.
    Fusionne blessures + suspensions historiques pour le cross-check à match_date.
    """
    # Index absence : player_id → liste (from, until)
    # On fusionne blessures et suspensions dans le même index
    abs_idx: dict[str, list] = {}
    for rec in injuries:
        abs_idx.setdefault(rec["player_id"], []).append(
            (rec["date_from"], rec["date_until"], "injury")
        )
    for rec in (suspensions or []):
        abs_idx.setdefault(rec["player_id"], []).append(
            (rec["date_from"], rec["date_until"], "suspension")
        )

    total_val   = squad["total_value"]
    unavail_val = 0
    n_injured   = 0
    n_suspended = 0
    players_sorted = sorted(squad["players"], key=lambda x: -x["value"])
    top5_ids = {p["player_id"] for p in players_sorted[:5]}

    for player in squad["players"]:
        pid = player["player_id"]
        for d_from, d_until, kind in abs_idx.get(pid, []):
            if d_from and d_until and d_from <= match_date <= d_until:
                unavail_val += player["value"]
                if kind == "injury":
                    n_injured += 1
                else:
                    n_suspended += 1
                break  # compte une seule fois par joueur

    # Top5 absent (blessure ou suspension)
    top5_absent = any(
        pid in top5_ids and any(
            d_from and d_until and d_from <= match_date <= d_until
            for d_from, d_until, _ in abs_idx.get(pid, [])
        )
        for pid in top5_ids
    )

    # Top scorer / créateur absent
    top_scorer_goals = top_creator_assists = 0
    if stats_df is not None and not stats_df.empty:
        ts = stats_df.dropna(subset=["goals"]).sort_values("goals", ascending=False)
        tc = stats_df.dropna(subset=["assists"]).sort_values("assists", ascending=False)
        if len(ts):
            top_scorer_goals = ts.iloc[0]["goals"]
        if len(tc):
            top_creator_assists = tc.iloc[0]["assists"]

    pct_available = (total_val - unavail_val) / total_val if total_val > 0 else 1.0

    return {
        "pct_value_available":  round(pct_available, 4),
        "n_injured":            n_injured,
        "n_suspended":          n_suspended,
        "n_unavailable":        n_injured + n_suspended,
        "top5_absent":          int(top5_absent),
        "squad_value_total":    total_val,
        "squad_avg_age":        squad.get("avg_age"),
        "squad_instability":    round(squad.get("squad_instability", 0), 3),
        "top_scorer_goals":     top_scorer_goals,
        "top_creator_assists":  top_creator_assists,
    }


# ─── 8. Pipeline principal ─────────────────────────────────────

def run(fd_league: str, seasons: list[int], out_dir: Path,
        with_injuries: bool = False, with_stats: bool = False):

    tm_code = LEAGUE_MAP.get(fd_league, fd_league)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Transfermarkt scraper | {fd_league} ({tm_code}) | saisons {seasons}")
    print(f"{'='*60}")

    print("\n[1] Récupération des équipes (toutes saisons)...")
    teams = get_all_historical_teams(tm_code, seasons)
    print(f"  {len(teams)} équipes uniques sur {len(seasons)} saisons")

    squad_records, injury_records, suspension_records, stat_records = [], [], [], []

    for season in seasons:
        print(f"\n[Saison {season}/{str(season+1)[-2:]}]")
        for t in teams:
            tid, tname, tslug = t["team_id"], t["team_name_tm"], t["slug"]
            print(f"  {tname}...", end=" ", flush=True)

            squad = get_squad_info(tid, season)
            if not squad:
                print("✗")
                continue

            squad_records.append({
                "team_id":          tid,
                "team_name_tm":     tname,
                "season":           season,
                "total_value":      squad["total_value"],
                "n_players":        squad["n_players"],
                "avg_age":          squad["avg_age"],
                "n_arrivals":       squad["n_arrivals"],
                "n_departures":     squad["n_departures"],
                "squad_instability": squad["squad_instability"],
            })

            if with_injuries:
                for player in squad["players"]:
                    pid, pslug = player["player_id"], player["slug"]
                    base = {
                        "team_id":      tid,
                        "team_name_tm": tname,
                        "season":       season,
                        "player_id":    pid,
                        "player_name":  player["name"],
                        "player_value": player["value"],
                    }
                    for inj in get_player_injuries(pid, pslug):
                        injury_records.append({**base, **inj})
                    for sus in get_player_suspensions(pid, pslug):
                        suspension_records.append({**base, **sus})

            if with_stats:
                df_stats = get_player_stats(tid, tm_code, season)
                if not df_stats.empty:
                    df_stats["team_id"]      = tid
                    df_stats["team_name_tm"] = tname
                    df_stats["season"]       = season
                    stat_records.append(df_stats)

            n_inj = len(injury_records) if with_injuries else 0
            n_sus = len(suspension_records) if with_injuries else 0
            print(f"✓ €{squad['total_value']/1e6:.0f}m | inj={n_inj} sus={n_sus}")

    # Sauvegarde (merge avec l'existant pour ne pas écraser les équipes déjà scrapées)
    def _merge_csv(new_records, path, dedup_cols):
        df_new = pd.DataFrame(new_records)
        if path.exists() and len(df_new):
            df_old = pd.read_csv(path)
            df_merged = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=dedup_cols)
            return df_merged
        return df_new

    df_squads = _merge_csv(squad_records, out_dir / f"squads_{fd_league}.csv",
                           dedup_cols=["team_id", "season"])
    df_squads.to_csv(out_dir / f"squads_{fd_league}.csv", index=False)
    print(f"\n→ squads: {len(df_squads)} lignes")

    if injury_records:
        df_inj = _merge_csv(injury_records, out_dir / f"injuries_{fd_league}.csv",
                            dedup_cols=["player_id", "date_from", "date_until"])
        df_inj["date_from"]  = pd.to_datetime(df_inj["date_from"],  errors="coerce")
        df_inj["date_until"] = pd.to_datetime(df_inj["date_until"], errors="coerce")
        df_inj.to_csv(out_dir / f"injuries_{fd_league}.csv", index=False)
        print(f"→ injuries: {len(df_inj)} lignes")

    if suspension_records:
        df_sus = _merge_csv(suspension_records, out_dir / f"suspensions_{fd_league}.csv",
                            dedup_cols=["player_id", "date_from", "date_until"])
        df_sus["date_from"]  = pd.to_datetime(df_sus["date_from"],  errors="coerce")
        df_sus["date_until"] = pd.to_datetime(df_sus["date_until"], errors="coerce")
        df_sus.to_csv(out_dir / f"suspensions_{fd_league}.csv", index=False)
        print(f"→ suspensions: {len(df_sus)} lignes")

    if stat_records:
        df_all_stats = pd.concat(stat_records, ignore_index=True)
        df_all_stats.to_csv(out_dir / f"stats_{fd_league}.csv", index=False)
        print(f"→ stats: {len(df_all_stats)} lignes")

    print("\n✅ Done")
    return df_squads


# ─── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--league",   default="E1")
    ap.add_argument("--seasons",  nargs="+", type=int, default=[2023])
    ap.add_argument("--injuries", action="store_true")
    ap.add_argument("--stats",    action="store_true")
    ap.add_argument("--out",      default="./tm_data")
    args = ap.parse_args()

    run(
        fd_league=args.league,
        seasons=args.seasons,
        out_dir=Path(args.out),
        with_injuries=args.injuries,
        with_stats=args.stats,
    )
