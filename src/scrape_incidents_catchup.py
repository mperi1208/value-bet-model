"""
scrape_incidents_catchup.py
---------------------------
Scrape les incidents (cartons rouges) pour des saisons déjà scrapées
en prématch, mais sans fichier incidents (ancienne version du code).

Usage :
    python3 scrape_incidents_catchup.py --leagues E1 --seasons 2016
    python3 scrape_incidents_catchup.py --leagues E1 F2 --seasons 2016 2017
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

import sys
sys.path.insert(0, str(Path(__file__).parent))

from scrape_sofascore import _make_browser, _make_page, get_match_incidents, SLEEP

OUT_DIR = Path(__file__).parent.parent / "data" / "sofascore"


def catchup_season(league: str, season_year: int, page, out_dir: Path) -> bool:
    season_str = f"{season_year % 100:02d}{(season_year + 1) % 100:02d}"
    tag        = f"[{league} {season_year}/{(season_year+1)%100:02d}]"

    pm_path  = out_dir / f"sofascore_prematch_{league}_{season_str}.csv"
    inc_path = out_dir / f"sofascore_incidents_{league}_{season_str}.csv"

    if not pm_path.exists():
        print(f"{tag} ✗ Pas de fichier prématch trouvé : {pm_path}")
        return False

    if inc_path.exists():
        print(f"{tag} Fichier incidents déjà présent ({inc_path.name}) — skip")
        return True

    df_pm = pd.read_csv(pm_path)
    if "event_id" not in df_pm.columns:
        print(f"{tag} ✗ Colonne event_id absente dans {pm_path.name}")
        return False

    eids  = df_pm["event_id"].dropna().astype(int).tolist()
    total = len(eids)
    print(f"\n{tag} {total} matchs à traiter pour les incidents")

    inc_rows = []
    errors   = 0

    for i, eid in enumerate(eids):
        # Récupère date/équipes depuis le CSV prématch
        row      = df_pm[df_pm["event_id"] == eid].iloc[0]
        date_str = str(row.get("date", "?"))
        home     = str(row.get("home_team", "?"))
        away     = str(row.get("away_team", "?"))

        print(f"  {tag} [{i+1:>3}/{total}] {date_str} | {home:22s} vs {away:22s}",
              end=" ", flush=True)

        try:
            reds = get_match_incidents(eid, page)
            for r in reds:
                r.update({"date": date_str, "home_team": home, "away_team": away})
            inc_rows.extend(reds)
            n_red = len([r for r in reds if not r["rescinded"]])
            print(f"red={n_red}")
            errors = 0
        except KeyboardInterrupt:
            print("\n⚠ Interrompu — sauvegarde en cours...")
            _save(inc_rows, inc_path, tag)
            raise
        except Exception as e:
            errors += 1
            print(f"✗ {e}")
            if errors >= 5:
                print(f"{tag} ✗ 5 erreurs consécutives — abandon")
                _save(inc_rows, inc_path, tag)
                return False
            time.sleep(3)

        # Sauvegarde intermédiaire tous les 100 matchs
        if (i + 1) % 100 == 0:
            _save(inc_rows, inc_path, tag, verbose=False)

    _save(inc_rows, inc_path, tag)
    return True


def _save(inc_rows, inc_path, tag, verbose=True):
    if inc_rows:
        pd.DataFrame(inc_rows).to_csv(inc_path, index=False)
        if verbose:
            n_red = sum(1 for r in inc_rows if not r.get("rescinded", False))
            print(f"{tag} ✓ {len(inc_rows)} incidents dont {n_red} rouges non-rescindés")
    else:
        if verbose:
            print(f"{tag} ✓ 0 incidents (aucun carton rouge cette saison)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues",  nargs="+", default=["E1"])
    ap.add_argument("--seasons",  nargs="+", type=int, default=[2016])
    ap.add_argument("--out",      default=str(OUT_DIR))
    ap.add_argument("--sleep",    type=float, default=SLEEP)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import scrape_sofascore as sf_mod
    sf_mod.SLEEP = args.sleep

    with sync_playwright() as p:
        browser = _make_browser(p)
        page    = _make_page(browser)

        for league in args.leagues:
            for season_year in args.seasons:
                catchup_season(league, season_year, page, out_dir)
                time.sleep(10)

        try:
            browser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
