"""
team_name_map.py
----------------
Mapping noms d'équipe → clé canonique football-data.co.uk.

Sources gérées : Sofascore, Transfermarkt, Understat
Clé cible : noms exacts utilisés dans football-data.co.uk (HomeTeam/AwayTeam).

Usage :
    from team_name_map import normalize_team
    fd_name = normalize_team("Nottingham Forest")  # → "Nott'm Forest"
"""

from difflib import get_close_matches

# ─── Mapping explicite (source → football-data) ───────────────────────────────
# Couvre les divergences connues pour E0, E1, F1, F2, D1, D2, SP1, SP2, I1, I2

_EXPLICIT: dict[str, str] = {
    # ── E0 / E1 ──────────────────────────────────────────────────────────────
    "Manchester City":          "Man City",
    "Manchester United":        "Man United",
    "Tottenham Hotspur":        "Tottenham",
    "Wolverhampton":            "Wolves",
    "Wolverhampton Wanderers":  "Wolves",
    "West Bromwich":            "West Brom",
    "West Bromwich Albion":     "West Brom",
    "Nottingham Forest":        "Nott'm Forest",
    "Sheffield Wednesday":      "Sheffield Weds",
    "Queens Park Rangers":      "QPR",
    "Burton Albion":            "Burton",
    "Huddersfield Town":        "Huddersfield",
    "Wigan Athletic":           "Wigan",
    "Rotherham United":         "Rotherham",
    "Bristol City":             "Bristol City",
    "Barnsley":                 "Barnsley",
    "Blackburn Rovers":         "Blackburn",
    "Birmingham City":          "Birmingham",
    "Leeds United":             "Leeds",
    "Cardiff City":             "Cardiff",
    "Derby County":             "Derby",
    "Preston North End":        "Preston",
    "Reading":                  "Reading",
    "Ipswich Town":             "Ipswich",
    "Norwich City":             "Norwich",
    "Newcastle United":         "Newcastle",
    "Brentford":                "Brentford",
    "Fulham":                   "Fulham",
    "Brighton & Hove Albion":   "Brighton",
    "Brighton":                 "Brighton",
    "Aston Villa":              "Aston Villa",
    "AFC Bournemouth":          "Bournemouth",
    "Bournemouth":              "Bournemouth",
    "Burnley":                  "Burnley",
    "Chelsea":                  "Chelsea",
    "Arsenal":                  "Arsenal",
    "Liverpool":                "Liverpool",
    "Everton":                  "Everton",
    "Leicester City":           "Leicester",
    "Leicester":                "Leicester",
    "Southampton":              "Southampton",
    "Crystal Palace":           "Crystal Palace",
    "Hull City":                "Hull",
    "Stoke City":               "Stoke",
    "Sunderland":               "Sunderland",
    "Swansea City":             "Swansea",
    "Watford":                  "Watford",
    "West Ham United":          "West Ham",
    "West Ham":                 "West Ham",
    "Middlesbrough":            "Middlesbrough",
    "Coventry City":            "Coventry",
    "Millwall":                 "Millwall",
    "Sunderland AFC":           "Sunderland",
    "Swansea":                  "Swansea",
    "Luton Town":               "Luton",
    "Luton":                    "Luton",
    "Blackpool":                "Blackpool",
    "Stoke":                    "Stoke",
    "Hull":                     "Hull",
    "Bristol City FC":          "Bristol City",
    "Sheffield United":         "Sheffield United",
    "Sheffield Utd":            "Sheffield United",
    "Peterborough United":      "Peterboro",
    "Swansea City AFC":         "Swansea",
    "Watford FC":               "Watford",
    # ── E1 TM extras ─────────────────────────────────────────────────────────
    "Charlton Athletic":        "Charlton",
    "Oxford United":            "Oxford",
    "Portsmouth FC":            "Portsmouth",
    "Wrexham AFC":              "Wrexham",
    "Middlesbrough FC":         "Middlesbrough",
    "Millwall FC":              "Millwall",
    "Wigan Athletic FC":        "Wigan",
    "Bolton Wanderers":         "Bolton",
    "Nottm Forest":             "Nott'm Forest",
    "Nottm Forest FC":          "Nott'm Forest",
    "Nott'm Forest":            "Nott'm Forest",
    "Plymouth Argyle":          "Plymouth",
    "Wycombe Wanderers":        "Wycombe",

    # ── F1 / F2 ──────────────────────────────────────────────────────────────
    "Paris Saint-Germain":      "Paris SG",
    "PSG":                      "Paris SG",
    "Saint-Étienne":            "St Etienne",
    "AS Saint-Étienne":         "St Etienne",
    "Olympique de Marseille":   "Marseille",
    "Olympique Lyonnais":       "Lyon",
    "Stade Rennais":            "Rennes",
    "Stade de Reims":           "Reims",
    "LOSC Lille":               "Lille",
    "Montpellier HSC":          "Montpellier",
    "Stade Brestois":           "Brest",
    "Stade Lavallois":          "Laval",
    "Le Havre AC":              "Le Havre",
    "RC Lens":                  "Lens",
    "Nîmes Olympique":          "Nimes",
    "Nimes Olympique":          "Nimes",
    "Valenciennes FC":          "Valenciennes",
    "AJ Auxerre":               "Auxerre",
    "FC Lorient":               "Lorient",
    "FC Nantes":                "Nantes",
    "OGC Nice":                 "Nice",
    "AS Monaco":                "Monaco",
    "EA Guingamp":              "Guingamp",
    "AS Nancy-Lorraine":        "Nancy",
    "FC Metz":                  "Metz",
    "SM Caen":                  "Caen",
    "Dijon FCO":                "Dijon",
    "Amiens SC":                "Amiens",
    "Clermont Foot":            "Clermont",
    "Toulouse FC":              "Toulouse",
    "Angers SCO":               "Angers",
    "Girondins de Bordeaux":    "Bordeaux",
    "Sporting Club de Bastia":  "Bastia",
    "Chamois Niortais":         "Niort",
    "Red Star FC":              "Red Star",
    "RC Strasbourg":            "Strasbourg",
    "Troyes AC":                "Troyes",
    "AC Ajaccio":               "Ajaccio",
    "Gazélec Ajaccio":          "Ajaccio GFCO",
    "GFC Ajaccio":              "Ajaccio GFCO",
    "FC Tours":                 "Tours",
    "Tours FC":                 "Tours",
    "US Orléans":               "Orleans",
    "Orleans":                  "Orleans",
    "Bourg-en-Bresse":          "Bourg Peronnas",
    "Bourg-Péronnas":           "Bourg Peronnas",
    "Sochaux":                  "Sochaux",
    "FC Sochaux":               "Sochaux",
    # ── F2 TM extras ─────────────────────────────────────────────────────────
    "Clermont Foot 63":             "Clermont",
    "ESTAC Troyes":                 "Troyes",
    "FC Annecy":                    "Annecy",
    "Grenoble Foot 38":             "Grenoble",
    "Le Mans FC":                   "Le Mans",
    "Pau FC":                       "Pau FC",
    "Rodez AF":                     "Rodez",
    "SC Bastia":                    "Bastia",
    "Stade Reims":                  "Reims",
    "US Boulogne":                  "Boulogne",
    "USL Dunkerque":                "Dunkerque",
    "FC Villefranche Beaujolais":   "Villefranche",
    "AS Béziers":                   "Beziers",
    "Quevilly-Rouen Métropole":     "Quevilly Rouen",
    "Entente SSG":                  "Entente SSG",
    "Paris FC":                     "Paris FC",
    "Châteauroux":                  "Chateauroux",
    "Le Havre AC":                  "Le Havre",
    "Valenciennes FC":              "Valenciennes",
    "Chamois Niortais FC":          "Niort",
    "FC Chambly Oise":              "Chambly",
    "FC Girondins Bordeaux":        "Bordeaux",
    "FC Sochaux-Montbéliard":       "Sochaux",
    "Football Bourg-en-Bresse Péronnas 01": "Bourg Peronnas",
    "RC Strasbourg Alsace":         "Strasbourg",
    "Stade Brestois 29":            "Brest",

    # ── D1 / D2 ──────────────────────────────────────────────────────────────
    "Borussia Dortmund":        "Dortmund",
    "Borussia M'gladbach":      "M'gladbach",
    "Borussia Mönchengladbach": "M'gladbach",
    "Eintracht Frankfurt":      "Ein Frankfurt",
    "1. FC Köln":               "FC Koln",
    "1. FC Koeln":              "FC Koln",
    "FC Schalke 04":            "Schalke 04",
    "Schalke":                  "Schalke 04",
    "Bayern München":           "Bayern Munich",
    "FC Bayern München":        "Bayern Munich",
    "SV Werder Bremen":         "Werder Bremen",
    "TSG Hoffenheim":           "Hoffenheim",
    "Bayer Leverkusen":         "Leverkusen",
    "VfL Wolfsburg":            "Wolfsburg",
    "Hertha BSC":               "Hertha",
    "FC Augsburg":              "Augsburg",
    "SC Freiburg":              "Freiburg",
    "1. FSV Mainz 05":          "Mainz",
    "Mainz 05":                 "Mainz",
    "RB Leipzig":               "RB Leipzig",
    "SV Darmstadt 98":          "Darmstadt",
    "FC Ingolstadt 04":         "Ingolstadt",
    "Hamburger SV":             "Hamburg",
    "VfL Bochum":               "Bochum",
    "1. FC Union Berlin":       "Union Berlin",
    "Union Berlin":             "Union Berlin",
    "FC St. Pauli":             "St Pauli",
    "Fortuna Düsseldorf":       "Dusseldorf",
    "1. FC Nürnberg":           "Nurnberg",
    "SpVgg Greuther Fürth":     "Greuther Furth",
    "SV Holstein Kiel":         "Holstein Kiel",
    "Heidenheim":               "Heidenheim",
    "1. FC Heidenheim":         "Heidenheim",
    "Hannover 96":              "Hannover",
    "Karlsruher SC":            "Karlsruhe",
    "SSV Jahn Regensburg":      "Regensburg",
    "Arminia Bielefeld":        "Bielefeld",
    "SV Sandhausen":            "Sandhausen",

    # ── SP1 / SP2 ────────────────────────────────────────────────────────────
    "Real Madrid":              "Real Madrid",
    "FC Barcelona":             "Barcelona",
    "Barcelona":                "Barcelona",
    "Atletico Madrid":          "Ath Madrid",
    "Atlético de Madrid":       "Ath Madrid",
    "Athletic Club":            "Ath Bilbao",
    "Athletic Bilbao":          "Ath Bilbao",
    "Villarreal CF":            "Villarreal",
    "Real Sociedad":            "Sociedad",
    "Real Betis":               "Betis",
    "Real Valladolid":          "Valladolid",
    "Sevilla FC":               "Sevilla",
    "RCD Espanyol":             "Espanol",
    "Espanyol":                 "Espanol",
    "Deportivo de La Coruña":   "La Coruna",
    "Getafe CF":                "Getafe",
    "Osasuna":                  "Osasuna",
    "CA Osasuna":               "Osasuna",
    "Celta de Vigo":            "Celta",
    "RC Celta":                 "Celta",
    "Cádiz CF":                 "Cadiz",
    "Levante UD":               "Levante",
    "Girona FC":                "Girona",
    "UD Almería":               "Almeria",
    "Elche CF":                 "Elche",
    "Rayo Vallecano":           "Vallecano",
    "Granada CF":               "Granada",
    "Valencia CF":              "Valencia",
    "Málaga CF":                "Malaga",
    "Deportivo Alavés":         "Alaves",
    "SD Huesca":                "Huesca",
    "UD Las Palmas":            "Las Palmas",
    "Mallorca":                 "Mallorca",
    "RCD Mallorca":             "Mallorca",
    "SD Eibar":                 "Eibar",
    "Zaragoza":                 "Zaragoza",

    # ── I1 / I2 ──────────────────────────────────────────────────────────────
    "Juventus":                 "Juventus",
    "Juventus FC":              "Juventus",
    "AC Milan":                 "Milan",
    "Inter":                    "Inter",
    "FC Internazionale":        "Inter",
    "Internazionale":           "Inter",
    "AS Roma":                  "Roma",
    "SSC Napoli":               "Napoli",
    "Lazio":                    "Lazio",
    "SS Lazio":                 "Lazio",
    "Atalanta":                 "Atalanta",
    "Atalanta BC":              "Atalanta",
    "Fiorentina":               "Fiorentina",
    "ACF Fiorentina":           "Fiorentina",
    "Sampdoria":                "Sampdoria",
    "UC Sampdoria":             "Sampdoria",
    "Torino FC":                "Torino",
    "Torino":                   "Torino",
    "Genoa CFC":                "Genoa",
    "Genoa":                    "Genoa",
    "Udinese Calcio":           "Udinese",
    "Udinese":                  "Udinese",
    "Bologna FC":               "Bologna",
    "Bologna":                  "Bologna",
    "Cagliari Calcio":          "Cagliari",
    "Cagliari":                 "Cagliari",
    "Hellas Verona":            "Verona",
    "Verona":                   "Verona",
    "Empoli FC":                "Empoli",
    "Empoli":                   "Empoli",
    "Sassuolo":                 "Sassuolo",
    "US Sassuolo":              "Sassuolo",
    "Crotone":                  "Crotone",
    "FC Crotone":               "Crotone",
    "Benevento Calcio":         "Benevento",
    "Venezia FC":               "Venezia",
    "Spezia Calcio":            "Spezia",
    "US Lecce":                 "Lecce",
    "Lecce":                    "Lecce",
    "US Salernitana":           "Salernitana",
    "Frosinone Calcio":         "Frosinone",
    "Monza":                    "Monza",
    "AC Monza":                 "Monza",
    "Frosinone":                "Frosinone",
}

# ─── Lookup rapide (lowercase → football-data) ────────────────────────────────
_LOWER_EXPLICIT: dict[str, str] = {k.lower(): v for k, v in _EXPLICIT.items()}

# Pool de noms football-data pour fallback fuzzy (à peupler dynamiquement)
_FD_POOL: list[str] = sorted(set(_EXPLICIT.values()))


def normalize_team(name: str, fd_pool: list[str] | None = None) -> str:
    """
    Retourne le nom football-data.co.uk correspondant à `name`.

    1. Lookup direct dans _EXPLICIT
    2. Lookup lowercase
    3. Fuzzy match dans fd_pool (si fourni ou fallback sur _FD_POOL)
    4. Retourne `name` tel quel si aucun match

    Parameters
    ----------
    name : str
        Nom source (Sofascore, Transfermarkt, Understat…)
    fd_pool : list[str], optional
        Liste des noms football-data pour cette saison/division.
        Si fourni, remplace _FD_POOL pour le fuzzy match.
    """
    if not name:
        return name

    # 1. Exact
    if name in _EXPLICIT:
        return _EXPLICIT[name]

    # 2. Lowercase
    lc = name.lower()
    if lc in _LOWER_EXPLICIT:
        return _LOWER_EXPLICIT[lc]

    # 3. Fuzzy
    pool = fd_pool if fd_pool else _FD_POOL
    matches = get_close_matches(name, pool, n=1, cutoff=0.72)
    if matches:
        return matches[0]

    # 4. Identité
    return name


def build_fd_pool(df_fd) -> list[str]:
    """
    Extrait les noms d'équipes uniques d'un DataFrame football-data
    (colonnes HomeTeam / AwayTeam).
    """
    teams = set(df_fd["HomeTeam"].dropna()) | set(df_fd["AwayTeam"].dropna())
    return sorted(teams)
