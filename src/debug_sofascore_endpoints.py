"""
debug_sofascore_endpoints.py
----------------------------
Navigue sur une page Sofascore et capture toutes les requêtes API
pour identifier les endpoints actifs pour les événements d'une saison.

Usage : python3 debug_sofascore_endpoints.py
"""
import json, time
from playwright.sync_api import sync_playwright

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# Championship 2018/19 (season_id=17473) — on cherche les endpoints events
TARGET_URL = "https://www.sofascore.com/tournament/football/england/championship/18#id:17473"

captured = []

def handle_response(response):
    url = response.url
    if "api.sofascore.com" in url and ("event" in url or "season" in url):
        try:
            status = response.status
            captured.append({"url": url, "status": status})
        except Exception:
            pass

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(extra_http_headers={"User-Agent": UA})
    page.on("response", handle_response)

    print(f"Navigation vers {TARGET_URL}...", flush=True)
    page.goto(TARGET_URL, wait_until="networkidle", timeout=30000)
    time.sleep(3)

    print(f"\n{len(captured)} requêtes API capturées :\n")
    seen = set()
    for r in captured:
        url = r["url"]
        # Déduplique par pattern
        pattern = url.split("?")[0]
        if pattern not in seen:
            seen.add(pattern)
            print(f"  [{r['status']}] {url}")

    browser.close()
