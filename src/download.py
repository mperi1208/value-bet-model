"""
download.py
-----------
Télécharge automatiquement tous les CSV depuis football-data.co.uk.

URL format : https://www.football-data.co.uk/mmz4281/XXYY/{LEAGUE}.csv
  XXYY = derniers 2 chiffres de chaque année (ex: 2425 = saison 2024-25)

Usage :
    python 00_download.py                  # télécharge tout (10 ans)
    python 00_download.py --seasons 5      # 5 dernières saisons seulement
    python 00_download.py --update         # télécharge uniquement la saison en cours
"""

import os
import time
import argparse
import urllib.request
import urllib.error
import ssl

# Fix SSL sur Mac (certificate verify failed)
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# ── Configuration
BASE_URL  = 'https://www.football-data.co.uk/mmz4281'

DIV1 = ['F1', 'D1', 'SP1', 'E0', 'I1']
DIV2 = ['D2', 'E1', 'F2', 'I2', 'SP2']
ALL_LEAGUES = DIV1 + DIV2

# Saisons disponibles sur le site (de la plus récente à la plus ancienne)
# Format : (année_debut, année_fin) ex: (2024, 2025) -> "2425"
ALL_SEASONS = [
    (2024, 2025),
    (2023, 2024),
    (2022, 2023),
    (2021, 2022),
    (2020, 2021),
    (2019, 2020),
    (2018, 2019),
    (2017, 2018),
    (2016, 2017),
    (2015, 2016),
    (2014, 2015),
    (2013, 2014),
    (2012, 2013),
    (2011, 2012),
    (2010, 2011),
    (2009, 2010),
    (2008, 2009),
    (2007, 2008),
    (2006, 2007),
    (2005, 2006),
    (2004, 2005),
    (2003, 2004),
    (2002, 2003),
    (2001, 2002),
    (2000, 2001),
]


def season_code(y1: int, y2: int) -> str:
    """Ex: (2024, 2025) -> '2425'"""
    return f'{str(y1)[2:]}{str(y2)[2:]}'


def download_all(csv_root: str, n_seasons: int = 10,
                 update_only: bool = False, delay: float = 1.0):
    """
    Télécharge les fichiers CSV et les organise dans csv/{LEAGUE}/{LEAGUE}-{N}.csv
    où N=1 = saison la plus récente, N=2 = saison précédente, etc.

    Args:
        csv_root   : dossier de destination (ex: './csv')
        n_seasons  : nombre de saisons à télécharger (défaut: 10)
        update_only: si True, télécharge seulement la saison en cours (N=1)
        delay      : pause entre chaque requête en secondes (respecte le site)
    """
    seasons = ALL_SEASONS[:1] if update_only else ALL_SEASONS[:n_seasons]

    total      = len(seasons) * len(ALL_LEAGUES)
    downloaded = 0
    skipped    = 0
    errors     = 0

    print(f'Téléchargement : {len(seasons)} saisons × {len(ALL_LEAGUES)} ligues = {total} fichiers')
    print(f'Dossier : {os.path.abspath(csv_root)}')
    print()

    for season_idx, (y1, y2) in enumerate(seasons, start=1):
        code = season_code(y1, y2)
        print(f'── Saison {y1}-{str(y2)[2:]} (N={season_idx}) ──')

        for league in ALL_LEAGUES:
            # Dossier de destination
            league_dir = os.path.join(csv_root, league)
            os.makedirs(league_dir, exist_ok=True)

            # Fichier local : {LEAGUE}-{N}.csv (N=1 = plus récent)
            local_path = os.path.join(league_dir, f'{league}-{season_idx}.csv')

            # Si le fichier existe déjà et ce n'est pas la saison en cours → skip
            # (la saison en cours peut être incomplète, on la re-télécharge toujours)
            if os.path.exists(local_path) and season_idx > 1:
                print(f'  {league} N={season_idx} : déjà présent, skip')
                skipped += 1
                continue

            # URL
            url = f'{BASE_URL}/{code}/{league}.csv'

            try:
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'Mozilla/5.0 (compatible; value-bet-pipeline/1.0)'}
                )
                with urllib.request.urlopen(req, timeout=15, context=ssl_context) as response:
                    content = response.read()

                # Vérifier que c'est bien un CSV valide (pas une page d'erreur HTML)
                if b'<html' in content[:200].lower():
                    print(f'  {league} N={season_idx} ({y1}-{str(y2)[2:]}) : non disponible')
                    errors += 1
                    continue

                # Vérifier qu'il y a des données (plus de 2 lignes)
                lines = content.decode('utf-8', errors='ignore').strip().split('\n')
                if len(lines) < 3:
                    print(f'  {league} N={season_idx} ({y1}-{str(y2)[2:]}) : vide, skip')
                    errors += 1
                    continue

                with open(local_path, 'wb') as f:
                    f.write(content)

                print(f'  {league} N={season_idx} ({y1}-{str(y2)[2:]}) : '
                      f'✓ {len(lines)-1} matchs')
                downloaded += 1

            except urllib.error.HTTPError as e:
                print(f'  {league} N={season_idx} ({y1}-{str(y2)[2:]}) : HTTP {e.code}')
                errors += 1
            except Exception as e:
                print(f'  {league} N={season_idx} ({y1}-{str(y2)[2:]}) : erreur {e}')
                errors += 1

            # Pause pour ne pas surcharger le site
            time.sleep(delay)

        print()

    print('═' * 50)
    print(f'  Téléchargés : {downloaded}')
    print(f'  Déjà présents (skip) : {skipped}')
    print(f'  Non disponibles/erreurs : {errors}')
    print('═' * 50)

    # Résumé par ligue
    print('\nFichiers présents par ligue :')
    for league in ALL_LEAGUES:
        league_dir = os.path.join(csv_root, league)
        if os.path.exists(league_dir):
            files = [f for f in os.listdir(league_dir) if f.endswith('.csv')]
            tier = 'Div1' if league in DIV1 else 'Div2'
            print(f'  {league:>5} ({tier}) : {len(files)} saisons')


if __name__ == '__main__':
    # Chemin du dossier csv relatif à l'emplacement de ce script
    _HERE = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Télécharge les CSV depuis football-data.co.uk')
    parser.add_argument('--csv',     default=os.path.join(_HERE, 'csv'),
                        help='Dossier de destination (défaut: ./csv)')
    parser.add_argument('--seasons', default=25, type=int,
                        help='Nombre de saisons à télécharger (défaut: 25)')
    parser.add_argument('--update',  action='store_true',
                        help='Met à jour uniquement la saison en cours (N=1)')
    parser.add_argument('--delay',   default=1.0, type=float,
                        help='Pause entre requêtes en secondes (défaut: 1.0)')
    args = parser.parse_args()

    download_all(
        csv_root   = args.csv,
        n_seasons  = args.seasons,
        update_only= args.update,
        delay      = args.delay,
    )
