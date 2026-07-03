"""
scrape_sofascore_xg.py
----------------------
Scrape les xG post-match depuis l'API SofaScore pour E1 et F2.

Utilise l'infrastructure Playwright existante (scrape_sofascore.py).
Pour chaque saison connue, récupère la liste des matchs terminés puis
appelle event/{id}/statistics pour extraire expected_goals home/away.

Output : data/sofascore/sofascore_xg_{league}_{season}.csv
Colonnes : event_id, date, home_team, away_team, xg_home, xg_away, league, season

Usage :
    python scrape_sofascore_xg.py --leagues E1 F2
    python scrape_sofascore_xg.py --leagues E1 --seasons 2022 2023
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

from scrape_sofascore import (
    TOURNAMENT_IDS,
    _make_browser,
    _make_page,
    _warmup,
    _fetch,
    get_all_events,
)

SLEEP       = 0.8
SEASON_IDS  = json.loads(
    (Path(__file__).parent.parent / "data" / "sofascore" / "season_ids.json").read_text()
)


def _extract_xg(stats_raw: dict | None) -> tuple[float | None, float | None]:
    """Extrait xG home/away depuis la réponse statistics de l'API SofaScore."""
    if not stats_raw:
        return None, None
    for period in stats_raw.get("statistics", []):
        if period.get("period") != "ALL":
            continue
        for group in period.get("groups", []):
            for item in group.get("statisticsItems", []):
                name = item.get("name", "").lower()
                if "expected" in name and "goal" in name:
                    h = item.get("home")
                    a = item.get("away")
                    try:
                        return float(h), float(a)
                    except (TypeError, ValueError):
                        return None, None
    return None, None


def _event_to_row(ev: dict, xg_h: float, xg_a: float, league: str, season: int) -> dict:
    """Construit un dict de ligne depuis un event SofaScore + xG."""
    ts = ev.get("startTimestamp", 0)
    date = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""
    return {
        "event_id":  ev.get("id"),
        "date":      date,
        "home_team": ev.get("homeTeam", {}).get("name", ""),
        "away_team": ev.get("awayTeam", {}).get("name", ""),
        "xg_home":   xg_h,
        "xg_away":   xg_a,
        "league":    league,
        "season":    season,
    }


def scrape_xg_season(league: str, season: int, out_dir: Path, page) -> pd.DataFrame:
    """Scrape xG pour une saison. Retourne un DataFrame."""
    key = f"{league}_{season}"
    if key not in SEASON_IDS:
        print(f"  ⚠ {key} absent de season_ids.json")
        return pd.DataFrame()

    season_id   = SEASON_IDS[key]
    tournament_id = TOURNAMENT_IDS[league]

    out_path = out_dir / f"sofascore_xg_{league}_{str(season)[2:]}{str(season+1)[2:]}.csv"

    # Charge l'existant pour skip les event_id déjà scrappés
    done_ids: set[int] = set()
    existing_rows: list[dict] = []
    if out_path.exists():
        ex = pd.read_csv(out_path)
        done_ids = set(ex["event_id"].dropna().astype(int).tolist())
        existing_rows = ex.to_dict("records")
        print(f"  {key} : {len(done_ids)} matchs déjà en cache")

    print(f"  {key} : récupération des matchs...", flush=True)
    events = get_all_events(tournament_id, season_id, page, season_year=season)
    print(f"  {key} : {len(events)} matchs terminés trouvés")

    new_rows: list[dict] = []
    for ev in events:
        eid = ev.get("id")
        if eid in done_ids:
            continue
        time.sleep(SLEEP)
        stats_raw = _fetch(page, f"event/{eid}/statistics")
        xg_h, xg_a = _extract_xg(stats_raw)
        if xg_h is None:
            # Pas de xG pour ce match (certains matchs early saison n'en ont pas)
            continue
        row = _event_to_row(ev, xg_h, xg_a, league, season)
        new_rows.append(row)

    all_rows = existing_rows + new_rows
    if not all_rows:
        print(f"  ⚠ {key} : aucune donnée xG récupérée")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.drop_duplicates(subset=["event_id"]).sort_values("date").reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"  ✓ {key} : {len(new_rows)} nouveaux, {len(df)} total → {out_path.name}")
    return df


def run(leagues: list[str], seasons: list[int], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = _make_browser(pw)
        stealth  = Stealth()
        page     = _make_page(browser)
        stealth.apply_stealth_sync(page)
        _warmup(page)

        for league in leagues:
            if league not in TOURNAMENT_IDS:
                print(f"⚠ Ligue inconnue : {league}")
                continue
            print(f"\n=== {league} ===")
            for s in seasons:
                scrape_xg_season(league, s, out_dir, page)
                time.sleep(2)

        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues",  nargs="+", default=["E1", "F2"])
    parser.add_argument("--seasons",  nargs="+", type=int,
                        default=[2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
    parser.add_argument("--out", default="data/sofascore")
    args = parser.parse_args()

    run(
        leagues=args.leagues,
        seasons=args.seasons,
        out_dir=Path(args.out),
    )
