"""
get_season_ids.py
-----------------
Récupère les season_ids Sofascore en parsant les pages web (pas l'API bloquée).
Exécuter une seule fois : python3 get_season_ids.py
"""
import json, sys, time, re
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent.parent / "data" / "sofascore" / "season_ids.json"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# Pages tournament Sofascore (URL publique, pas l'API)
TOURNAMENT_PAGES = {
    "E1": "https://www.sofascore.com/tournament/football/england/championship/18",
    "F2": "https://www.sofascore.com/tournament/football/france/ligue-2/182",
}

def extract_seasons_from_page(page, league: str, url: str) -> dict:
    """Visite la page tournament et extrait les season_ids du __NEXT_DATA__."""
    print(f"  Navigation vers {url}", flush=True)
    page.goto(url, wait_until="networkidle", timeout=30000)
    time.sleep(2)

    # Méthode 1 : __NEXT_DATA__ (Next.js)
    try:
        raw = page.locator("script#__NEXT_DATA__").inner_text(timeout=5000)
        data = json.loads(raw)
        # Cherche les seasons récursivement
        seasons = _find_seasons(data)
        if seasons:
            print(f"  __NEXT_DATA__ : {len(seasons)} saisons trouvées", flush=True)
            return seasons
    except Exception as e:
        print(f"  __NEXT_DATA__ échoué : {e}", flush=True)

    # Méthode 2 : chercher dans le JS de la page avec regex
    try:
        content = page.content()
        # Pattern : "id":XXXXX,"year":"YY/YY"
        matches = re.findall(r'"id"\s*:\s*(\d+)\s*,\s*"year"\s*:\s*"(\d{2}/\d{2})"', content)
        if matches:
            print(f"  Regex : {len(matches)} candidats trouvés", flush=True)
            return {f"{league}_{2000+int(y[:2])}": int(sid)
                    for sid, y in matches
                    if 16 <= int(y[:2]) <= 23}
    except Exception as e:
        print(f"  Regex échoué : {e}", flush=True)

    # Méthode 3 : intercepter les requêtes réseau
    return {}


def _find_seasons(obj, depth=0) -> dict:
    """Cherche récursivement les objets {id, year} dans le JSON."""
    if depth > 20:
        return {}
    result = {}
    if isinstance(obj, dict):
        if "year" in obj and "id" in obj:
            y = str(obj["year"])
            sid = obj["id"]
            m = re.match(r"(\d{2})/\d{2}", y)
            if m:
                full = 2000 + int(m.group(1))
                if 2016 <= full <= 2023:
                    result[full] = sid
        for v in obj.values():
            result.update(_find_seasons(v, depth+1))
    elif isinstance(obj, list):
        for item in obj:
            result.update(_find_seasons(item, depth+1))
    return result


cache = {}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(extra_http_headers={"User-Agent": UA})

    # Intercepte les réponses API pour capturer les seasons en cours de route
    captured = {}
    def handle_response(response):
        if "seasons" in response.url and "sofascore" in response.url:
            try:
                data = response.json()
                if data and data.get("seasons"):
                    captured["seasons"] = data["seasons"]
                    print(f"  [intercept] Capturé {len(data['seasons'])} saisons depuis {response.url}", flush=True)
            except Exception:
                pass
    page.on("response", handle_response)

    for league, url in TOURNAMENT_PAGES.items():
        print(f"\n=== {league} ===", flush=True)
        captured.clear()

        seasons_dict = extract_seasons_from_page(page, league, url)

        # Si interception réseau a capturé les saisons
        if not seasons_dict and captured.get("seasons"):
            for s in captured["seasons"]:
                y = s.get("year", "")
                m = re.match(r"(\d{2})/\d{2}", y)
                if m:
                    full = 2000 + int(m.group(1))
                    if 2016 <= full <= 2023:
                        seasons_dict[full] = s["id"]
            print(f"  Interception : {len(seasons_dict)} saisons", flush=True)

        if seasons_dict:
            for year, sid in sorted(seasons_dict.items()):
                print(f"    {year}/{(year+1)%100:02d} → id={sid}", flush=True)
                cache[f"{league}_{year}"] = sid
        else:
            print(f"  ECHEC total pour {league}", flush=True)

        time.sleep(2)

    browser.close()

print(f"\n=== Résultat final ===", flush=True)
print(json.dumps(cache, indent=2))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(cache, indent=2))
print(f"Sauvegardé → {OUT}", flush=True)
