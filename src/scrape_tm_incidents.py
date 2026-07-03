"""
scrape_tm_incidents.py
----------------------
Scrape Transfermarkt pour les cartons rouges historiques.
Alternative à Sofascore (non soumise au blocage Cloudflare).

Output : data/sofascore/sofascore_incidents_{league}_{season_str}.csv
(même format que get_match_incidents() — compatible avec enrich_suspensions.py)

Champs produits :
  event_id, player_id, player_name, player_pos,
  is_home, time, reason, rescinded, card_class,
  date, home_team, away_team

Usage :
    python3 scrape_tm_incidents.py --leagues E1 F2
    python3 scrape_tm_incidents.py --leagues E1 --seasons 2018 2019
    python3 scrape_tm_incidents.py --resume          # sauts les saisons déjà faites
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ─── Config ────────────────────────────────────────────────────

LEAGUE_TO_TM = {
    "E1":  ("championship",     "GB2"),
    "F2":  ("ligue-2",          "FR2"),
    "D2":  ("2-bundesliga",     "L2"),
    "I2":  ("serie-b",          "IT2"),
    "SP2": ("segunda-division", "ES2"),
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.transfermarkt.com/",
}

SLEEP      = 1.5   # secondes entre requêtes
BASE_URL   = "https://www.transfermarkt.com"
OUT_DIR    = Path(__file__).parent.parent / "data" / "sofascore"
CHECKPOINT = OUT_DIR / "tm_incidents_checkpoint.json"


# ─── HTTP helper ───────────────────────────────────────────────

def _get(url: str) -> requests.Response:
    time.sleep(SLEEP)
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r


# ─── Checkpoint ────────────────────────────────────────────────

def load_checkpoint() -> set:
    if CHECKPOINT.exists():
        return set(json.loads(CHECKPOINT.read_text()))
    return set()


def save_checkpoint(done: set):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(sorted(done), indent=2))


# ─── 1. Schedule d'une saison ──────────────────────────────────

def get_schedule(league: str, season_year: int) -> list[dict]:
    """Retourne la liste de tous les matchs terminés d'une saison."""
    slug, tm_code = LEAGUE_TO_TM[league]
    url = (
        f"{BASE_URL}/{slug}/gesamtspielplan"
        f"/wettbewerb/{tm_code}/saison_id/{season_year}"
    )
    r = _get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    matches = []
    current_date = None

    for tr in soup.find_all("tr"):
        spielbericht = tr.find("a", href=lambda h: h and "/index/spielbericht/" in h)
        if not spielbericht:
            continue

        # Score = lien spielbericht — skip les matchs sans score (à venir)
        score_text = spielbericht.text.strip()
        if not score_text or ":" not in score_text:
            continue

        match_href = spielbericht["href"]
        match_id   = match_href.split("/spielbericht/")[-1]

        # Date : première cellule non-vide du row (parfois héritée de la row précédente)
        cells = tr.find_all("td")
        date_raw = cells[0].text.strip() if cells else ""
        date_raw = re.sub(r"^\w+\s*", "", date_raw).strip()  # retire "Fri ", "Sat "...
        if date_raw and len(date_raw) > 4:
            current_date = date_raw

        # Équipes : liens vers /spielplan/verein/
        team_links = [
            a.text.strip()
            for a in tr.find_all("a", href=True)
            if "/spielplan/verein/" in a["href"] and a.text.strip()
        ]
        if len(team_links) < 2:
            continue

        home_team = team_links[0]
        away_team = team_links[-1]

        try:
            parts = score_text.split(":")
            home_score = int(parts[0].strip())
            away_score = int(parts[1].strip())
        except Exception:
            home_score = away_score = None

        matches.append({
            "match_id":   match_id,
            "date_raw":   current_date or "",
            "home_team":  home_team,
            "away_team":  away_team,
            "home_score": home_score,
            "away_score": away_score,
            "url":        BASE_URL + match_href,
        })

    return matches


# ─── 2. Cartons rouges d'un match ──────────────────────────────

def get_red_cards(match: dict) -> list[dict]:
    """
    Scrape la section 'Cards' d'une fiche Transfermarkt.
    Retourne uniquement les rouges directs et rouges après jaune.

    TM représente un 2e jaune comme deux entrées séparées (sb-gelb + sb-rot)
    pour le même joueur. On le détecte en vérifiant si le joueur a aussi un jaune.
    """
    r = _get(match["url"])
    soup = BeautifulSoup(r.text, "html.parser")

    karten_div = soup.find("div", {"id": "sb-karten"})
    if not karten_div:
        return []

    # Collecte toutes les entrées de la section Cards
    entries = []  # list of (player_name, player_id, is_home, card_type, minute, reason)

    for li in karten_div.find_all("li"):
        li_cls = " ".join(li.get("class", []))
        if "sb-aktion" not in li_cls:
            continue
        is_home = "sb-aktion-heim" in li_cls

        # Détermine le type de carton à partir des classes de tous les spans
        all_span_cls = set()
        for span in li.find_all("span"):
            for c in span.get("class", []):
                all_span_cls.add(c)

        if "sb-rot" in all_span_cls:
            raw_type = "sb-rot"
        elif "sb-gelbrot" in all_span_cls:
            raw_type = "sb-gelbrot"
        elif "sb-gelb" in all_span_cls:
            raw_type = "sb-gelb"
        else:
            continue

        # Joueur
        player_link = li.find("a", class_="wichtig")
        if not player_link:
            continue
        player_name = player_link.text.strip()
        player_href = player_link.get("href", "")
        pid_match   = re.search(r"/spieler/(\d+)", player_href)
        player_id   = pid_match.group(1) if pid_match else None

        # Minute
        clock_span = li.find("span", class_="sb-sprite-uhr-klein")
        minute = _decode_minute(clock_span) if clock_span else None

        # Raison
        action_div = li.find("div", class_="sb-aktion-aktion")
        reason = ""
        if action_div:
            txt = action_div.get_text(separator=" ", strip=True)
            reason = re.sub(re.escape(player_name), "", txt).strip(" ,.")
            reason = re.sub(r"^\d+\.\s*(Yellow|Red|Gelb|Rot)\s*card\s*,?\s*", "", reason, flags=re.I).strip()

        entries.append({
            "player_name": player_name,
            "player_id":   player_id,
            "is_home":     is_home,
            "raw_type":    raw_type,
            "minute":      minute,
            "reason":      reason,
        })

    # Joueurs qui ont à la fois un jaune ET un rouge → 2e jaune (yellowRed)
    players_with_yellow = {
        (e["player_name"], e["is_home"])
        for e in entries if e["raw_type"] == "sb-gelb"
    }

    rows = []
    for e in entries:
        if e["raw_type"] not in ("sb-rot", "sb-gelbrot"):
            continue  # jaune seul → ignore

        key = (e["player_name"], e["is_home"])
        if e["raw_type"] == "sb-gelbrot":
            card_class = "yellowRed"
        elif key in players_with_yellow:
            card_class = "yellowRed"  # rouge après jaune dans ce match
        else:
            card_class = "red"

        rows.append({
            "event_id":    match["match_id"],
            "player_id":   e["player_id"],
            "player_name": e["player_name"],
            "player_pos":  None,
            "is_home":     e["is_home"],
            "time":        e["minute"],
            "reason":      e["reason"],
            "rescinded":   False,
            "card_class":  card_class,
            "date":        match["date_raw"],
            "home_team":   match["home_team"],
            "away_team":   match["away_team"],
        })

    return rows


def _decode_minute(span) -> int | None:
    """Décode la minute depuis le background-position du sprite Transfermarkt."""
    style = span.get("style", "")
    m = re.search(r"background-position:\s*(-?\d+)px\s+(-?\d+)px", style)
    if not m:
        return None
    x, y = int(m.group(1)), int(m.group(2))
    # TM sprite : chaque 'tick' = 36px horizontal, rangée = 36px vertical
    # minute = col * 10 + row * ... (approximatif, suffisant pour débogage)
    col = abs(x) // 36
    row = abs(y) // 36
    return col + row * 10 or None


# ─── 3. Pipeline saison complète ───────────────────────────────

def scrape_season(league: str, season_year: int) -> bool:
    slug, _ = LEAGUE_TO_TM[league]
    season_str = f"{season_year % 100:02d}{(season_year + 1) % 100:02d}"
    tag        = f"[{league} {season_year}/{(season_year+1)%100:02d}]"
    out_path   = OUT_DIR / f"sofascore_incidents_{league}_{season_str}.csv"

    if out_path.exists():
        print(f"{tag} Déjà présent : {out_path.name} — skip")
        return True

    print(f"\n{tag} Récupération du calendrier...", flush=True)
    try:
        matches = get_schedule(league, season_year)
    except Exception as e:
        print(f"{tag} ✗ Schedule : {e}")
        return False

    total = len(matches)
    print(f"{tag} {total} matchs terminés", flush=True)

    all_reds = []
    errors   = 0

    for i, match in enumerate(matches):
        label = f"{match['home_team'][:18]} vs {match['away_team'][:18]}"
        print(f"  {tag} [{i+1:>3}/{total}] {match['date_raw'][:8]} | {label:<38}", end=" ", flush=True)

        try:
            reds = get_red_cards(match)
            all_reds.extend(reds)
            n_red = len([r for r in reds if r["card_class"] == "red"])
            n_yr  = len([r for r in reds if r["card_class"] == "yellowRed"])
            print(f"red={n_red} yr={n_yr}")
            errors = 0
        except KeyboardInterrupt:
            print("\n⚠ Interrompu")
            _save(all_reds, out_path, tag)
            raise
        except Exception as e:
            errors += 1
            print(f"✗ {e}")
            if errors >= 8:
                print(f"{tag} ✗ 8 erreurs consécutives — abandon")
                _save(all_reds, out_path, tag)
                return False
            continue

        # Sauvegarde intermédiaire tous les 100 matchs
        if (i + 1) % 100 == 0:
            _save(all_reds, out_path, tag, verbose=False)

    _save(all_reds, out_path, tag)
    return True


def _save(rows: list, path: Path, tag: str, verbose: bool = True):
    if not rows:
        if verbose:
            print(f"{tag} ✓ 0 cartons rouges cette saison")
        pd.DataFrame(columns=[
            "event_id","player_id","player_name","player_pos",
            "is_home","time","reason","rescinded","card_class",
            "date","home_team","away_team",
        ]).to_csv(path, index=False)
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    if verbose:
        n_red = (df["card_class"] == "red").sum()
        n_yr  = (df["card_class"] == "yellowRed").sum()
        print(f"{tag} ✓ {len(df)} incidents : {n_red} rouges directs, {n_yr} jaune-rouges → {path.name}")


# ─── CLI ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues",  nargs="+", default=["E1", "F2"])
    ap.add_argument("--seasons",  nargs="+", type=int, default=list(range(2016, 2024)))
    ap.add_argument("--resume",   action="store_true")
    ap.add_argument("--sleep",    type=float, default=SLEEP)
    args = ap.parse_args()

    import scrape_tm_incidents as _self
    _self.SLEEP = args.sleep

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = load_checkpoint() if args.resume else set()

    todo = [
        (lg, sy)
        for lg in args.leagues
        for sy in args.seasons
        if lg in LEAGUE_TO_TM and f"{lg}_{sy}" not in done
    ]

    if not todo:
        print("✅ Tout déjà scrappé.")
        return

    print(f"Leagues : {args.leagues}")
    print(f"Saisons : {args.seasons}")
    print(f"À scraper : {len(todo)} saisons\n")

    for league, season_year in todo:
        key = f"{league}_{season_year}"
        try:
            ok = scrape_season(league, season_year)
        except KeyboardInterrupt:
            print("\n⛔ Arrêt.")
            save_checkpoint(done)
            break
        except Exception as e:
            print(f"✗ {key} : {e}")
            ok = False

        if ok:
            done.add(key)
            save_checkpoint(done)

    print(f"\n✅ Terminé. {len(done)} saisons dans le checkpoint.")


if __name__ == "__main__":
    main()
