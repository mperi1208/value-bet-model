"""
load.py
-------
Charge tous les CSV depuis csv/{DIV}/{DIV}-{N}.csv.
Sépare div1 (entraînement) et div2 (value bet cible).
Gère les différences de colonnes entre fichiers récents et anciens.
"""

import os
import glob
import pandas as pd
import numpy as np

DIV1 = ['F1', 'D1', 'SP1', 'E0', 'I1']
DIV2 = ['D2', 'E1', 'F2', 'I2', 'SP2']
ALL_LEAGUES = DIV1 + DIV2

# Colonnes de base toujours présentes
BASE_COLS = [
    'Div', 'Date', 'HomeTeam', 'AwayTeam',
    'FTHG', 'FTAG', 'FTR',
    'HTHG', 'HTAG', 'HTR',
    'HS', 'AS', 'HST', 'AST',
    'HC', 'AC', 'HF', 'AF',
    'HY', 'AY', 'HR', 'AR',
    'Referee',
]

# Colonnes O/U : plusieurs noms selon l'époque du fichier
# On les normalise toutes vers Avg>2.5 et Max>2.5
OU_CANDIDATES = {
    'Avg>2.5': ['Avg>2.5', 'BbAv>2.5'],
    'Max>2.5': ['Max>2.5', 'BbMx>2.5'],
    'Avg<2.5': ['Avg<2.5', 'BbAv<2.5'],
    'Max<2.5': ['Max<2.5', 'BbMx<2.5'],
}


def load_all(csv_root: str) -> pd.DataFrame:
    frames = []

    for league in ALL_LEAGUES:
        pattern = os.path.join(csv_root, league, f'{league}-*.csv')
        files   = sorted(glob.glob(pattern))

        if not files:
            print(f'  Aucun fichier : {league}')
            continue

        tier = 1 if league in DIV1 else 2

        for fpath in files:
            try:
                # Tentative 1 : utf-8 strict
                # Tentative 2 : latin-1 (encodage Windows courant)
                # on_bad_lines='skip' ignore les lignes avec trop de colonnes
                # utf-8-sig en premier : 'utf-8' seul ne strippe pas le BOM
                # des fichiers récents → la colonne 'Div' devenait '﻿Div'
                # et les fichiers concernés perdaient leur ligue (bug audit)
                df_raw = None
                for encoding in ('utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1'):
                    try:
                        df_raw = pd.read_csv(
                            fpath,
                            low_memory=False,
                            encoding=encoding,
                            on_bad_lines='skip',
                        )
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception:
                        # Tokenization error : réessayer avec on_bad_lines warn
                        try:
                            df_raw = pd.read_csv(
                                fpath,
                                low_memory=False,
                                encoding=encoding,
                                on_bad_lines='warn',
                                engine='python',
                            )
                            break
                        except Exception:
                            continue
                if df_raw is None:
                    print(f'  Skip : {fpath}')
                    continue
                available = [c for c in BASE_COLS if c in df_raw.columns]
                df        = df_raw[available].copy()

                # Normalisation des colonnes O/U : fallback sur noms anciens
                for target_col, candidates in OU_CANDIDATES.items():
                    df[target_col] = np.nan
                    for src in candidates:
                        if src in df_raw.columns:
                            df[target_col] = pd.to_numeric(
                                df_raw[src], errors='coerce'
                            )
                            break  # on prend le premier disponible

                df['tier'] = tier
                frames.append(df)

            except Exception as e:
                print(f'  Erreur {fpath} : {e}')

    if not frames:
        raise ValueError('Aucun fichier chargé. Vérifie csv_root.')

    df_all = pd.concat(frames, ignore_index=True)
    df_all = _clean(df_all)

    # Numérotation des saisons par année calendaire réelle
    df_all['season_year'] = df_all['Date'].apply(
        lambda d: d.year if d.month >= 7 else d.year - 1
    )
    years_sorted = sorted(df_all['season_year'].unique(), reverse=True)
    year_to_id   = {y: i+1 for i, y in enumerate(years_sorted)}
    df_all['season'] = df_all['season_year'].map(year_to_id)

    n1 = (df_all['tier'] == 1).sum()
    n2 = (df_all['tier'] == 2).sum()
    n_seasons = df_all['season'].nunique()

    y_min = years_sorted[-1]; y_max = years_sorted[0]
    nan_total = df_all['Avg>2.5'].isna().sum()
    print(f'  Données : {len(df_all)} matchs | {n_seasons} saisons '
          f'({y_min}-{str(y_min+1)[2:]} → {y_max}-{str(y_max+1)[2:]}) '
          f'| {nan_total} NaN cotes')

    return df_all


def load_div1(csv_root: str) -> pd.DataFrame:
    df = load_all(csv_root)
    return df[df['tier'] == 1].copy().reset_index(drop=True)


def load_div2(csv_root: str) -> pd.DataFrame:
    df = load_all(csv_root)
    return df[df['tier'] == 2].copy().reset_index(drop=True)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date', 'FTR', 'FTHG', 'FTAG'])
    df['FTHG'] = df['FTHG'].astype(int)
    df['FTAG'] = df['FTAG'].astype(int)

    for c in ['HS','AS','HST','AST','HC','AC','HF','AF','HY','AY','HR','AR']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    return df.sort_values(['Div', 'Date']).reset_index(drop=True)


if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'csv'
    )
    df = load_all(root)
