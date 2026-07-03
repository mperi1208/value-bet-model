"""
scrape_batch.py
---------------
Scraping historique Sofascore avec checkpoint/reprise.

Usage :
    python scrape_batch.py --leagues E1 F2 --seasons 2016 2017 2018 2019 2020 2021 2022 2023
    python scrape_batch.py --leagues E1 --seasons 2023          # test 1 saison
    python scrape_batch.py --resume                              # reprend depuis checkpoint

Durée estimée : ~8s/match × 500 matchs × 16 saisons ≈ 18h
Lancer de nuit, le checkpoint permet de reprendre à tout moment.

Structure de sortie :
    data/sofascore/
        checkpoint.json                        ← saisons terminées
        sofascore_prematch_E1_1617.csv
        sofascore_stats_E1_1617.csv
        sofascore_prematch_F2_1617.csv
        ...
"""

import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

import sys
sys.path.insert(0, str(Path(__file__).parent))

from scrape_sofascore import (
    TOURNAMENT_IDS, _make_browser, _make_page, _fetch,
    get_seasons, get_all_events,
    get_match_stats, get_prematch_features,
    get_match_incidents,
    SLEEP,
)

OUT_DIR    = Path(__file__).parent.parent / "data" / "sofascore"
CHECKPOINT = OUT_DIR / "checkpoint.json"


# ─── Checkpoint ──────────────────────────────────────────────────────────────

def load_checkpoint() -> set:
    if CHECKPOINT.exists():
        return set(json.loads(CHECKPOINT.read_text()))
    return set()


def save_checkpoint(done: set):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(sorted(done), indent=2))


def checkpoint_key(league: str, season_year: int) -> str:
    return f"{league}_{season_year}"


# ─── Season ID : cache JSON + fallback API ────────────────────────────────────

_SEASON_ID_CACHE_PATH = OUT_DIR / "season_ids.json"
_season_id_cache: dict = {}

def _load_season_cache():
    global _season_id_cache
    if _SEASON_ID_CACHE_PATH.exists():
        _season_id_cache = json.loads(_SEASON_ID_CACHE_PATH.read_text())

def _get_season_id(league: str, season_year: int, tournament_id: int, page, tag: str):
    _load_season_cache()
    key = f"{league}_{season_year}"

    # 1. Cache fichier
    if key in _season_id_cache:
        sid = _season_id_cache[key]
        print(f"{tag} season_id={sid} (depuis cache)")
        return sid

    # 2. API Sofascore
    seasons = get_seasons(tournament_id, page)
    for s in seasons:
        if str(season_year % 100).zfill(2) in s.get("year", ""):
            sid = s["id"]
            # Mémorise pour les prochains runs
            _season_id_cache[key] = sid
            _SEASON_ID_CACHE_PATH.write_text(json.dumps(_season_id_cache, indent=2))
            return sid

    print(f"{tag} ✗ season_id introuvable. "
          f"Exécute get_season_ids.py ou ajoute manuellement dans {_SEASON_ID_CACHE_PATH}")
    return None


# ─── Scraping d'une saison ────────────────────────────────────────────────────

def scrape_season(league: str, season_year: int, page, out_dir: Path) -> bool:
    """
    Scrape une saison complète.
    Retourne True si succès, False si échec critique.
    """
    tournament_id = TOURNAMENT_IDS[league]
    season_str    = f"{season_year % 100:02d}{(season_year + 1) % 100:02d}"
    tag           = f"[{league} {season_year}/{(season_year+1)%100:02d}]"

    # Trouve le season_id — cache en priorité, API en fallback
    season_id = _get_season_id(league, season_year, tournament_id, page, tag)
    if not season_id:
        return False
    print(f"\n{tag} season_id={season_id}")

    # Vérifie si CSV déjà partiel (reprise intra-saison)
    pm_path  = out_dir / f"sofascore_prematch_{league}_{season_str}.csv"
    st_path  = out_dir / f"sofascore_stats_{league}_{season_str}.csv"
    inc_path = out_dir / f"sofascore_incidents_{league}_{season_str}.csv"

    done_eids: set = set()
    pm_existing, st_existing, inc_existing = [], [], []
    if pm_path.exists():
        df_ex = pd.read_csv(pm_path)
        done_eids = set(df_ex["event_id"].tolist())
        pm_existing = df_ex.to_dict("records")
        print(f"{tag} Reprise : {len(done_eids)} matchs déjà scrappés")
    if st_path.exists():
        st_existing = pd.read_csv(st_path).to_dict("records")
    if inc_path.exists():
        inc_existing = pd.read_csv(inc_path).to_dict("records")

    # Récupère tous les matchs
    events = get_all_events(tournament_id, season_id, page, season_year=season_year)
    total  = len(events)
    print(f"{tag} {total} matchs terminés")

    pm_rows  = list(pm_existing)
    st_rows  = list(st_existing)
    inc_rows = list(inc_existing)
    errors   = 0

    for i, ev in enumerate(events):
        eid       = ev["id"]
        home_name = ev.get("homeTeam", {}).get("name", "?")
        away_name = ev.get("awayTeam", {}).get("name", "?")
        ts        = ev.get("startTimestamp", 0)
        date_str  = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else "?"

        if eid in done_eids:
            continue

        print(f"  {tag} [{i+1:>3}/{total}] {date_str} | {home_name:22s} vs {away_name:22s}",
              end=" ", flush=True)

        try:
            # Stats post-match
            if ev.get("hasXg"):
                stats = get_match_stats(eid, page)
                stats.update({
                    "date": date_str, "home_team": home_name, "away_team": away_name,
                    "home_score": ev.get("homeScore", {}).get("current"),
                    "away_score": ev.get("awayScore", {}).get("current"),
                })
                st_rows.append(stats)
                print(f"xG={stats.get('h_expected_goals','?')}|{stats.get('a_expected_goals','?')}", end=" ")

            # Pré-match
            pm = get_prematch_features(eid, page)
            pm.update({
                "date": date_str, "home_team": home_name, "away_team": away_name,
                "home_score": ev.get("homeScore", {}).get("current"),
                "away_score": ev.get("awayScore", {}).get("current"),
            })
            pm_rows.append(pm)
            n_miss = (pm.get("h_missing_confirmed") or 0) + (pm.get("a_missing_confirmed") or 0)
            u25    = pm.get("under_25_odds")
            move   = pm.get("under_25_move")

            # Incidents (cartons rouges → suspensions)
            reds = get_match_incidents(eid, page)
            for r in reds:
                r.update({"date": date_str, "home_team": home_name, "away_team": away_name})
            inc_rows.extend(reds)
            n_red = len([r for r in reds if not r["rescinded"]])
            print(f"abs={n_miss} U2.5={u25} mv={move} red={n_red}")

            done_eids.add(eid)
            errors = 0  # reset error counter on success

        except KeyboardInterrupt:
            print("\n⚠ Interrompu — sauvegarde en cours...")
            _save(pm_rows, st_rows, inc_rows, pm_path, st_path, inc_path, tag)
            raise

        except Exception as e:
            errors += 1
            print(f"✗ {e}")
            if errors >= 5:
                print(f"{tag} ✗ 5 erreurs consécutives — abandon de la saison")
                _save(pm_rows, st_rows, inc_rows, pm_path, st_path, inc_path, tag)
                return False
            time.sleep(3)
            continue

        # Sauvegarde intermédiaire tous les 50 matchs
        if len(pm_rows) % 50 == 0:
            _save(pm_rows, st_rows, inc_rows, pm_path, st_path, inc_path, tag, verbose=False)

    _save(pm_rows, st_rows, inc_rows, pm_path, st_path, inc_path, tag)
    return True


def _save(pm_rows, st_rows, inc_rows, pm_path, st_path, inc_path, tag, verbose=True):
    if pm_rows:
        pd.DataFrame(pm_rows).to_csv(pm_path, index=False)
    if st_rows:
        pd.DataFrame(st_rows).to_csv(st_path, index=False)
    if inc_rows:
        pd.DataFrame(inc_rows).to_csv(inc_path, index=False)
    if verbose:
        n_red = sum(1 for r in inc_rows if not r.get("rescinded", False))
        print(f"{tag} ✓ Sauvegardé — {len(pm_rows)} prematch | {len(st_rows)} stats | {n_red} rouges")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues",  nargs="+", default=["E1", "F2"])
    ap.add_argument("--seasons",  nargs="+", type=int,
                    default=list(range(2016, 2024)))  # 2016→2023 = 8 saisons
    ap.add_argument("--out",      default=str(OUT_DIR))
    ap.add_argument("--resume",   action="store_true",
                    help="Reprend uniquement les saisons non terminées")
    ap.add_argument("--sleep",    type=float, default=SLEEP)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    done = load_checkpoint()
    todo = [
        (lg, sy)
        for lg in args.leagues
        for sy in args.seasons
        if lg in TOURNAMENT_IDS and checkpoint_key(lg, sy) not in done
    ]

    if not todo:
        print("✅ Tout est déjà scrappé selon le checkpoint.")
        return

    total_seasons = len(todo)
    est_h = total_seasons * 500 * 10 / 3600
    print(f"Leagues : {args.leagues}")
    print(f"Saisons : {args.seasons}")
    print(f"À scraper : {total_seasons} saisons | estimé ~{est_h:.0f}h")
    print(f"Sortie   : {out_dir}\n")

    # Patch SLEEP global
    import scrape_sofascore as sf_mod
    sf_mod.SLEEP = args.sleep

    def _new_page(playwright_ctx):
        browser = _make_browser(playwright_ctx)
        page    = _make_page(browser)
        return browser, page

    with sync_playwright() as p:
        browser, page = _new_page(p)

        for idx, (league, season_year) in enumerate(todo):
            key = checkpoint_key(league, season_year)
            print(f"\n{'='*60}")
            print(f"  Saison {idx+1}/{total_seasons} : {key}")
            print(f"{'='*60}")
            t0 = time.time()

            # Redémarre le browser entre chaque saison pour éviter le rate-limit
            if idx > 0:
                try:
                    browser.close()
                except Exception:
                    pass
                print("  ↻ Redémarrage du browser (anti rate-limit)...")
                time.sleep(15)
                browser, page = _new_page(p)

            try:
                ok = scrape_season(league, season_year, page, out_dir)
            except KeyboardInterrupt:
                print("\n⛔ Arrêt manuel. Checkpoint sauvegardé.")
                save_checkpoint(done)
                try:
                    browser.close()
                except Exception:
                    pass
                break
            except Exception as e:
                print(f"✗ Erreur inattendue sur {key}: {e}")
                traceback.print_exc()
                ok = False

            elapsed = (time.time() - t0) / 60
            if ok:
                done.add(key)
                save_checkpoint(done)
                print(f"  ✓ {key} terminé en {elapsed:.1f} min")
            else:
                print(f"  ✗ {key} échoué après {elapsed:.1f} min — continuer")

        try:
            browser.close()
        except Exception:
            pass

    print(f"\n✅ Batch terminé. {len(done)} saisons dans le checkpoint.")


if __name__ == "__main__":
    main()
