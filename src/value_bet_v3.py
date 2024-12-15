#!/usr/bin/env python3
"""
value_bet_v3.py — Pipeline enrichi avec features de marché
============================================================
Ajouts vs v2 :
  - Features de marché : dispersion cotes, écart Pinnacle vs marché,
    marge implicite, max/avg ratio
  - xG proxy : shots-on-target ratio rolling, conversion rate
  - Interaction features : marché × stats
  - Deux modes de calibration : Platt (défaut) ou beta calibration
  - Support Pinnacle odds (P>2.5, P<2.5, PSH/D/A)

Usage :
    python3 value_bet_v3.py --csv ./csv --market under --edge 0.03
    python3 value_bet_v3.py --csv ./csv --market draw --edge 0.04
    python3 value_bet_v3.py --csv ./csv --market under --edge 0.02 --no-ensemble
"""

import os
import sys
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

DIV1 = ["F1", "D1", "SP1", "E0", "I1"]
DIV2 = ["D2", "E1", "F2", "I2", "SP2"]

BASE_COLS = [
    "Div", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST", "HC", "AC",
    "HF", "AF", "HY", "AY", "HR", "AR",
]

# Toutes les colonnes de cotes qu'on veut charger
ODDS_LOAD = [
    # 1X2 individual bookmakers
    "B365H", "B365D", "B365A",
    "BWH", "BWD", "BWA",
    "IWH", "IWD", "IWA",
    "PSH", "PSD", "PSA",       # Pinnacle
    "WHH", "WHD", "WHA",
    "VCH", "VCD", "VCA",
    # 1X2 aggregates
    "AvgH", "AvgD", "AvgA",
    "MaxH", "MaxD", "MaxA",
    # O/U individual
    "B365>2.5", "B365<2.5",
    "P>2.5", "P<2.5",          # Pinnacle O/U
    # O/U aggregates
    "Avg>2.5", "Avg<2.5",
    "Max>2.5", "Max<2.5",
    # Legacy columns
    "BbAvH", "BbAvD", "BbAvA",
    "BbMxH", "BbMxD", "BbMxA",
    "BbAv>2.5", "BbAv<2.5",
    "BbMx>2.5", "BbMx<2.5",
    "BbOU",
]

RANDOM_STATE = 42


# ═══════════════════════════════════════════════════════════════
# 0. DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_data(csv_root: str, leagues: List[str]) -> pd.DataFrame:
    frames = []
    for league in leagues:
        pattern = os.path.join(csv_root, league, f"{league}-*.csv")
        for fpath in sorted(glob.glob(pattern)):
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    df_raw = pd.read_csv(fpath, low_memory=False,
                                         encoding=enc, on_bad_lines="skip")
                    break
                except Exception:
                    continue
            else:
                continue

            available = [c for c in BASE_COLS if c in df_raw.columns]
            df = df_raw[available].copy()

            # Charger toutes les colonnes de cotes disponibles
            for col in ODDS_LOAD:
                if col in df_raw.columns:
                    df[col] = pd.to_numeric(df_raw[col], errors="coerce")

            # Normaliser les anciens noms
            legacy_map = {
                "BbAvH": "AvgH", "BbAvD": "AvgD", "BbAvA": "AvgA",
                "BbMxH": "MaxH", "BbMxD": "MaxD", "BbMxA": "MaxA",
                "BbAv>2.5": "Avg>2.5", "BbAv<2.5": "Avg<2.5",
                "BbMx>2.5": "Max>2.5", "BbMx<2.5": "Max<2.5",
            }
            for old, new in legacy_map.items():
                if old in df.columns and (new not in df.columns or df[new].isna().all()):
                    df[new] = df[old]

            df["_file"] = os.path.basename(fpath)
            frames.append(df)

    if not frames:
        raise ValueError(f"Aucun CSV trouvé sous {csv_root}")

    df_all = pd.concat(frames, ignore_index=True)
    df_all["Date"] = pd.to_datetime(df_all["Date"], format="mixed",
                                     dayfirst=True, errors="coerce")
    df_all = df_all.dropna(subset=["Date", "FTR", "FTHG", "FTAG"])
    df_all["FTHG"] = df_all["FTHG"].astype(int)
    df_all["FTAG"] = df_all["FTAG"].astype(int)

    for c in ["HS", "AS", "HST", "AST", "HC", "AC",
              "HF", "AF", "HY", "AY", "HR", "AR"]:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    df_all = df_all.sort_values(["Div", "Date"]).reset_index(drop=True)

    df_all["season_year"] = df_all["Date"].apply(
        lambda d: d.year if d.month >= 7 else d.year - 1
    )
    years = sorted(df_all["season_year"].unique(), reverse=True)
    df_all["season"] = df_all["season_year"].map(
        {y: i + 1 for i, y in enumerate(years)}
    )

    # Stats de disponibilité des cotes
    pinnacle_pct = df_all["PSH"].notna().mean() if "PSH" in df_all.columns else 0
    ou_pinn_pct = df_all["P>2.5"].notna().mean() if "P>2.5" in df_all.columns else 0

    print(f"  Chargé: {len(df_all)} matchs | {df_all['season'].nunique()} saisons "
          f"| {df_all['Div'].nunique()} ligues")
    print(f"  Cotes Pinnacle 1X2: {pinnacle_pct:.0%} | O/U: {ou_pinn_pct:.0%}")

    return df_all


# ═══════════════════════════════════════════════════════════════
# 1. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

# --- 1a. Team rolling stats (même logique que v2) ---

def _team_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    home = df[["Div", "season", "Date", "HomeTeam",
               "FTHG", "FTAG", "FTR"]].copy()
    home.columns = ["Div", "season", "Date", "team",
                    "scored", "conceded", "FTR"]
    home["is_home"] = True
    home["idx"] = df.index

    # xG proxy columns
    for col_pair in [("HS", "HST"), ("AS", "AST")]:
        if col_pair[0] in df.columns and col_pair[1] in df.columns:
            pass  # handled below

    away = df[["Div", "season", "Date", "AwayTeam",
               "FTAG", "FTHG", "FTR"]].copy()
    away.columns = ["Div", "season", "Date", "team",
                    "scored", "conceded", "FTR"]
    away["is_home"] = False
    away["idx"] = df.index

    # Add shots data
    if "HS" in df.columns:
        home["shots"] = df["HS"]
        home["sot"] = df["HST"]
        home["shots_against"] = df["AS"]
        home["sot_against"] = df["AST"]
        away["shots"] = df["AS"]
        away["sot"] = df["AST"]
        away["shots_against"] = df["HS"]
        away["sot_against"] = df["HST"]

    # Add fouls
    if "HF" in df.columns:
        home["fouls"] = df["HF"]
        away["fouls"] = df["AF"]

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["Div", "season", "team", "Date"]).reset_index(drop=True)

    long["total"] = long["scored"] + long["conceded"]
    long["is_under"] = (long["total"] < 2.5).astype(float)
    long["is_draw"] = (long["FTR"] == "D").astype(float)
    long["win"] = (
        ((long["is_home"]) & (long["FTR"] == "H")) |
        ((~long["is_home"]) & (long["FTR"] == "A"))
    ).astype(float)

    # xG proxy: SOT conversion rate, shot accuracy
    if "shots" in long.columns:
        long["sot_ratio"] = (long["sot"] / long["shots"].replace(0, np.nan)).fillna(0)
        long["conversion"] = (long["scored"] / long["sot"].replace(0, np.nan)).fillna(0)
        long["defensive_sot"] = long["sot_against"]  # SOT concédés

    grp = ["Div", "season", "team"]
    features = {}

    stat_cols = ["scored", "conceded", "total", "is_under", "is_draw", "win"]
    if "shots" in long.columns:
        stat_cols += ["sot_ratio", "conversion", "sot", "defensive_sot"]

    for w in [5, 10]:
        for col in stat_cols:
            if col not in long.columns:
                continue
            feat_name = f"{col}_r{w}"
            shifted = long.groupby(grp, group_keys=False)[col].shift(1)
            features[feat_name] = (
                shifted
                .groupby([long["Div"], long["season"], long["team"]])
                .transform(lambda x: x.rolling(w, min_periods=max(2, w // 2)).mean())
            )

    # Variance buts 10 matchs
    shifted_total = long.groupby(grp, group_keys=False)["total"].shift(1)
    features["total_var_10"] = (
        shifted_total
        .groupby([long["Div"], long["season"], long["team"]])
        .transform(lambda x: x.rolling(10, min_periods=3).std())
    )

    feat_df = pd.DataFrame(features, index=long.index)
    feat_df["idx"] = long["idx"]
    feat_df["is_home"] = long["is_home"]

    home_feat = feat_df[feat_df["is_home"]].drop(columns="is_home").set_index("idx")
    away_feat = feat_df[~feat_df["is_home"]].drop(columns="is_home").set_index("idx")

    df = df.join(home_feat.add_prefix("h_"), how="left")
    df = df.join(away_feat.add_prefix("a_"), how="left")

    return df


# --- 1b. Market features ---

def _market_features(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """
    Features extraites des cotes des bookmakers.
    Principe : le DÉSACCORD entre bookmakers = signal de mispricing.
    """
    df = df.copy()

    if market == "under":
        # Collecter toutes les cotes under disponibles
        under_cols = [c for c in ["B365<2.5", "P<2.5", "Avg<2.5"] if c in df.columns]
        over_cols = [c for c in ["B365>2.5", "P>2.5", "Avg>2.5"] if c in df.columns]

        if len(under_cols) >= 2:
            under_odds = df[under_cols]
            # Dispersion = std des cotes / mean
            df["mkt_dispersion_under"] = under_odds.std(axis=1) / under_odds.mean(axis=1)
            # Range normalisé
            df["mkt_range_under"] = (under_odds.max(axis=1) - under_odds.min(axis=1)) / under_odds.mean(axis=1)
        else:
            df["mkt_dispersion_under"] = 0
            df["mkt_range_under"] = 0

        # Pinnacle vs marché
        if "P<2.5" in df.columns and "Avg<2.5" in df.columns:
            df["pinnacle_vs_avg_under"] = (
                (1 / df["P<2.5"].replace(0, np.nan)) -
                (1 / df["Avg<2.5"].replace(0, np.nan))
            )
        else:
            df["pinnacle_vs_avg_under"] = 0

        # Max / Avg ratio (proxy de "quelqu'un offre beaucoup plus")
        if "Max<2.5" in df.columns and "Avg<2.5" in df.columns:
            df["max_avg_ratio_under"] = df["Max<2.5"] / df["Avg<2.5"].replace(0, np.nan)
        else:
            df["max_avg_ratio_under"] = 1.0

        # Marge O/U (proxy de l'incertitude du marché)
        if "Avg>2.5" in df.columns and "Avg<2.5" in df.columns:
            df["ou_margin"] = (
                1 / df["Avg>2.5"].replace(0, np.nan) +
                1 / df["Avg<2.5"].replace(0, np.nan)
            ) - 1
        else:
            df["ou_margin"] = np.nan

        # No-vig probs
        if "Avg>2.5" in df.columns and "Avg<2.5" in df.columns:
            ro = 1 / df["Avg>2.5"].replace(0, np.nan)
            ru = 1 / df["Avg<2.5"].replace(0, np.nan)
            m = ro + ru
            df["nv_over"] = ro / m
            df["nv_under"] = ru / m
        else:
            df["nv_over"] = np.nan
            df["nv_under"] = np.nan

    elif market == "draw":
        # Collecter cotes draw
        draw_cols = [c for c in ["B365D", "BWD", "IWD", "PSD", "WHD", "VCD", "AvgD"]
                     if c in df.columns]

        if len(draw_cols) >= 2:
            draw_odds = df[draw_cols]
            df["mkt_dispersion_draw"] = draw_odds.std(axis=1) / draw_odds.mean(axis=1)
            df["mkt_range_draw"] = (draw_odds.max(axis=1) - draw_odds.min(axis=1)) / draw_odds.mean(axis=1)
        else:
            df["mkt_dispersion_draw"] = 0
            df["mkt_range_draw"] = 0

        # Pinnacle vs marché
        if "PSD" in df.columns and "AvgD" in df.columns:
            df["pinnacle_vs_avg_draw"] = (
                (1 / df["PSD"].replace(0, np.nan)) -
                (1 / df["AvgD"].replace(0, np.nan))
            )
        else:
            df["pinnacle_vs_avg_draw"] = 0

        # Max / Avg
        if "MaxD" in df.columns and "AvgD" in df.columns:
            df["max_avg_ratio_draw"] = df["MaxD"] / df["AvgD"].replace(0, np.nan)
        else:
            df["max_avg_ratio_draw"] = 1.0

        # Marge 1X2
        if all(c in df.columns for c in ["AvgH", "AvgD", "AvgA"]):
            df["match_margin"] = (
                1 / df["AvgH"].replace(0, np.nan) +
                1 / df["AvgD"].replace(0, np.nan) +
                1 / df["AvgA"].replace(0, np.nan)
            ) - 1
        else:
            df["match_margin"] = np.nan

        # No-vig probs
        if all(c in df.columns for c in ["AvgH", "AvgD", "AvgA"]):
            rh = 1 / df["AvgH"].replace(0, np.nan)
            rd = 1 / df["AvgD"].replace(0, np.nan)
            ra = 1 / df["AvgA"].replace(0, np.nan)
            m = rh + rd + ra
            df["nv_home"] = rh / m
            df["nv_draw"] = rd / m
            df["nv_away"] = ra / m
        else:
            df["nv_home"] = np.nan
            df["nv_draw"] = np.nan
            df["nv_away"] = np.nan

    return df


# --- 1c. Ranking ---

def _ranking(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    h_ranks, a_ranks = {}, {}
    for (div, season), grp in df.groupby(["Div", "season"], sort=False):
        grp = grp.sort_values("Date")
        stats = {}
        for team in pd.concat([grp["HomeTeam"], grp["AwayTeam"]]).unique():
            stats[team] = {"pts": 0, "gd": 0}
        for idx, row in grp.iterrows():
            sorted_t = sorted(stats.items(), key=lambda x: (-x[1]["pts"], -x[1]["gd"]))
            rank_map = {t: i + 1 for i, (t, _) in enumerate(sorted_t)}
            h_ranks[idx] = rank_map.get(row["HomeTeam"])
            a_ranks[idx] = rank_map.get(row["AwayTeam"])
            h, a = row["HomeTeam"], row["AwayTeam"]
            hg, ag = row["FTHG"], row["FTAG"]
            if row["FTR"] == "H":
                stats[h]["pts"] += 3
            elif row["FTR"] == "A":
                stats[a]["pts"] += 3
            else:
                stats[h]["pts"] += 1; stats[a]["pts"] += 1
            stats[h]["gd"] += (hg - ag); stats[a]["gd"] += (ag - hg)
    df["h_rank"] = pd.Series(h_ranks)
    df["a_rank"] = pd.Series(a_ranks)
    return df


# --- 1d. Build all features ---

def build_features(df: pd.DataFrame, market: str) -> Tuple[pd.DataFrame, List[str]]:
    print("  [a] Rolling stats...")
    df = _team_rolling(df)
    print("  [b] Rankings...")
    df = _ranking(df)
    print("  [c] Market features...")
    df = _market_features(df, market)

    # Targets
    df["total_goals"] = df["FTHG"] + df["FTAG"]
    df["under_25"] = (df["total_goals"] < 2.5).astype(int)
    df["draw"] = (df["FTR"] == "D").astype(int)

    # Combined features
    for w in [5, 10]:
        df[f"comb_under_r{w}"] = (
            df.get(f"h_is_under_r{w}", pd.Series(0.5, index=df.index)).fillna(0.5) +
            df.get(f"a_is_under_r{w}", pd.Series(0.5, index=df.index)).fillna(0.5)
        ) / 2
        df[f"comb_draw_r{w}"] = (
            df.get(f"h_is_draw_r{w}", pd.Series(0.26, index=df.index)).fillna(0.26) +
            df.get(f"a_is_draw_r{w}", pd.Series(0.26, index=df.index)).fillna(0.26)
        ) / 2
        df[f"comb_goals_r{w}"] = (
            df.get(f"h_total_r{w}", pd.Series(2.7, index=df.index)).fillna(2.7) +
            df.get(f"a_total_r{w}", pd.Series(2.7, index=df.index)).fillna(2.7)
        ) / 2
        df[f"goal_diff_gap_r{w}"] = abs(
            (df.get(f"h_scored_r{w}", pd.Series(1.3, index=df.index)).fillna(1.3) -
             df.get(f"h_conceded_r{w}", pd.Series(1.3, index=df.index)).fillna(1.3)) -
            (df.get(f"a_scored_r{w}", pd.Series(1.3, index=df.index)).fillna(1.3) -
             df.get(f"a_conceded_r{w}", pd.Series(1.3, index=df.index)).fillna(1.3))
        )

    df["rank_diff"] = df["a_rank"].fillna(10) - df["h_rank"].fillna(10)

    # xG proxy combined features
    if "h_sot_r5" in df.columns:
        df["comb_sot_r5"] = (df["h_sot_r5"].fillna(4) + df["a_sot_r5"].fillna(4)) / 2
        df["comb_sot_r10"] = (df["h_sot_r10"].fillna(4) + df["a_sot_r10"].fillna(4)) / 2
        df["comb_conversion_r5"] = (
            df["h_conversion_r5"].fillna(0.3) + df["a_conversion_r5"].fillna(0.3)
        ) / 2
        df["comb_sot_ratio_r5"] = (
            df["h_sot_ratio_r5"].fillna(0.35) + df["a_sot_ratio_r5"].fillna(0.35)
        ) / 2
        # Defensive: SOT concédés
        df["comb_def_sot_r5"] = (
            df["h_defensive_sot_r5"].fillna(4) + df["a_defensive_sot_r5"].fillna(4)
        ) / 2

    # ── Feature selection par marché ──
    if market == "under":
        feature_cols = [
            # Rolling stats (core)
            "h_scored_r5", "h_conceded_r5", "h_is_under_r5",
            "a_scored_r5", "a_conceded_r5", "a_is_under_r5",
            "h_is_under_r10", "a_is_under_r10",
            # Combined
            "comb_under_r5", "comb_under_r10",
            "comb_goals_r5", "comb_goals_r10",
            "goal_diff_gap_r5",
            # Variance
            "h_total_var_10", "a_total_var_10",
            # Ranking
            "rank_diff",
            # xG proxy
            "comb_sot_r5", "comb_conversion_r5", "comb_def_sot_r5",
            "comb_sot_ratio_r5",
            # MARKET FEATURES (new)
            "mkt_dispersion_under",
            "mkt_range_under",
            "pinnacle_vs_avg_under",
            "max_avg_ratio_under",
            "ou_margin",
        ]
        target = "under_25"
        odds_col = "Avg<2.5"
        prob_bookie_col = "nv_under"

    elif market == "draw":
        feature_cols = [
            # Rolling stats
            "h_is_draw_r5", "a_is_draw_r5",
            "h_win_r5", "a_win_r5",
            "h_scored_r5", "a_scored_r5",
            "h_is_draw_r10", "a_is_draw_r10",
            # Combined
            "comb_draw_r5", "comb_draw_r10",
            "goal_diff_gap_r5", "goal_diff_gap_r10",
            "comb_goals_r5",
            # Ranking
            "rank_diff",
            # xG proxy
            "comb_sot_r5", "comb_conversion_r5",
            "comb_sot_ratio_r5",
            # MARKET FEATURES (new)
            "mkt_dispersion_draw",
            "mkt_range_draw",
            "pinnacle_vs_avg_draw",
            "max_avg_ratio_draw",
            "match_margin",
        ]
        target = "draw"
        odds_col = "AvgD"
        prob_bookie_col = "nv_draw"
    else:
        raise ValueError(f"Market inconnu: {market}")

    # Filter features that actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    df["_target"] = df[target]
    df["_odds"] = df[odds_col] if odds_col in df.columns else np.nan
    df["_prob_bookie"] = df[prob_bookie_col] if prob_bookie_col in df.columns else np.nan

    n_before = len(df)
    df = df.dropna(subset=["_target", "_odds", "_prob_bookie"]).copy()
    print(f"  Features: {len(feature_cols)} cols | {len(df)} matchs "
          f"(dropped {n_before - len(df)} sans cotes)")

    return df, feature_cols


# ═══════════════════════════════════════════════════════════════
# 2. MODEL
# ═══════════════════════════════════════════════════════════════

def _make_xgb():
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=15,
        reg_alpha=0.5,
        reg_lambda=3.0,
        gamma=1.0,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _make_logreg():
    return LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE)


def train_fold(X_train, y_train, X_val, y_val, use_ensemble=True):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(np.nan_to_num(X_train, nan=0.0))
    X_val_sc = scaler.transform(np.nan_to_num(X_val, nan=0.0))
    X_train_clean = np.nan_to_num(X_train, nan=0.0)
    X_val_clean = np.nan_to_num(X_val, nan=0.0)

    xgb_model = _make_xgb()
    xgb_model.fit(X_train_clean, y_train,
                  eval_set=[(X_val_clean, y_val)], verbose=False)
    xgb_val = xgb_model.predict_proba(X_val_clean)[:, 1]

    if use_ensemble:
        lr_model = _make_logreg()
        lr_model.fit(X_train_sc, y_train)
        lr_val = lr_model.predict_proba(X_val_sc)[:, 1]
        blend_val = 0.6 * xgb_val + 0.4 * lr_val
    else:
        lr_model = None
        blend_val = xgb_val

    # Platt calibration
    calib = LogisticRegression(C=1e10, max_iter=1000)
    calib.fit(blend_val.reshape(-1, 1), y_val)

    return {
        "xgb": xgb_model, "lr": lr_model, "scaler": scaler,
        "calibrator": calib, "use_ensemble": use_ensemble,
    }


def predict_fold(model, X):
    X_clean = np.nan_to_num(X, nan=0.0)
    xgb_p = model["xgb"].predict_proba(X_clean)[:, 1]

    if model["use_ensemble"] and model["lr"] is not None:
        X_sc = model["scaler"].transform(X_clean)
        lr_p = model["lr"].predict_proba(X_sc)[:, 1]
        blend = 0.6 * xgb_p + 0.4 * lr_p
    else:
        blend = xgb_p

    return model["calibrator"].predict_proba(blend.reshape(-1, 1))[:, 1]


# ═══════════════════════════════════════════════════════════════
# 3. WALK-FORWARD
# ═══════════════════════════════════════════════════════════════

def walk_forward(df, feature_cols, min_train_seasons=5, use_ensemble=True):
    n_seasons = df["season"].max()
    all_preds = []

    for test_s in range(1, n_seasons - min_train_seasons):
        val_s = test_s + 1
        train_start = test_s + 2

        train = df[df["season"] >= train_start].dropna(subset=feature_cols + ["_target"])
        val = df[df["season"] == val_s].dropna(subset=feature_cols + ["_target"])
        test = df[df["season"] == test_s].dropna(subset=feature_cols + ["_target"])

        if len(train) < 500 or len(val) < 100 or len(test) < 50:
            continue

        model = train_fold(
            train[feature_cols].values, train["_target"].values,
            val[feature_cols].values, val["_target"].values,
            use_ensemble=use_ensemble,
        )
        probs = predict_fold(model, test[feature_cols].values)

        auc = roc_auc_score(test["_target"], probs) if len(np.unique(test["_target"])) > 1 else 0.5
        brier = brier_score_loss(test["_target"], probs)

        test_out = test.copy()
        test_out["prob_model"] = probs
        all_preds.append(test_out)

        sy = test["season_year"].iloc[0]
        print(f"  Fold {len(all_preds):>2} | {sy} | "
              f"tr={len(train):>5} val={len(val):>4} te={len(test):>4} | "
              f"AUC={auc:.3f} Brier={brier:.4f}")

        # Feature importance du dernier fold
        if len(all_preds) == 1:
            imp = pd.Series(
                model["xgb"].feature_importances_, index=feature_cols
            ).sort_values(ascending=False)
            print(f"  Top features: {' | '.join(f'{f}={v:.3f}' for f, v in imp.head(7).items())}")

    if not all_preds:
        raise ValueError("Aucun fold valide")

    df_wf = pd.concat(all_preds, ignore_index=True)
    overall_auc = roc_auc_score(df_wf["_target"], df_wf["prob_model"])
    overall_brier = brier_score_loss(df_wf["_target"], df_wf["prob_model"])

    print(f"\n  Total: {len(df_wf)} matchs | AUC={overall_auc:.4f} | Brier={overall_brier:.4f}")

    # Calibration check global
    for lo, hi in [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8)]:
        mask = (df_wf["prob_model"] >= lo) & (df_wf["prob_model"] < hi)
        if mask.sum() > 50:
            actual = df_wf.loc[mask, "_target"].mean()
            pred = df_wf.loc[mask, "prob_model"].mean()
            print(f"    Calib [{lo:.1f}-{hi:.1f}): n={mask.sum():>5} | "
                  f"pred={pred:.3f} actual={actual:.3f} delta={actual-pred:+.3f}")

    return df_wf


# ═══════════════════════════════════════════════════════════════
# 4. BACKTEST
# ═══════════════════════════════════════════════════════════════

def backtest(df_wf, edge_min=0.03, kelly_frac=0.25):
    df = df_wf.copy()
    df["edge"] = df["prob_model"] - df["_prob_bookie"]
    vb = df[df["edge"] >= edge_min].copy().reset_index(drop=True)

    if len(vb) == 0:
        print(f"\n  Aucun value bet (edge >= {edge_min:.0%})")
        return None

    n_vb = len(vb)
    print(f"\n  Value bets: {n_vb}/{len(df)} ({n_vb/len(df):.1%})")

    # Flat
    vb["profit_flat"] = np.where(vb["_target"] == 1, vb["_odds"] - 1, -1.0)
    vb["cum_flat"] = vb["profit_flat"].cumsum()
    roi_flat = vb["profit_flat"].sum() / n_vb * 100

    # Kelly
    vb["kelly_stake"] = np.clip(
        kelly_frac * (vb["prob_model"] * vb["_odds"] - 1) / (vb["_odds"] - 1),
        0, 0.05,
    )
    vb["profit_kelly"] = np.where(
        vb["_target"] == 1,
        vb["kelly_stake"] * (vb["_odds"] - 1),
        -vb["kelly_stake"],
    )
    vb["cum_kelly"] = vb["profit_kelly"].cumsum()
    roi_kelly = vb["profit_kelly"].sum() / vb["kelly_stake"].sum() * 100 if vb["kelly_stake"].sum() > 0 else 0

    # Calibration on value bets
    actual_wr = vb["_target"].mean()
    pred_wr = vb["prob_model"].mean()
    avg_odds = vb["_odds"].mean()

    print(f"\n  {'='*55}")
    print(f"  RESULTS | {n_vb} bets")
    print(f"  {'='*55}")
    print(f"  Win rate:    {actual_wr:.1%} (breakeven: {1/avg_odds:.1%})")
    print(f"  Avg odds:    {avg_odds:.2f}")
    print(f"  Avg edge:    {vb['edge'].mean():.1%}")
    print(f"  ROI flat:    {roi_flat:+.2f}%")
    print(f"  ROI Kelly:   {roi_kelly:+.2f}%")
    print(f"  Calib delta: pred={pred_wr:.1%} actual={actual_wr:.1%} ({actual_wr-pred_wr:+.1%})")
    print(f"\n  Par ligue:")
    for div, g in vb.groupby("Div"):
        r = g["profit_flat"].sum() / len(g) * 100
        wr = g["_target"].mean()
        print(f"    {div:>4}: {len(g):>4} bets | win {wr:.0%} | ROI {r:+.1f}%")

    return vb


def test_significance(vb, n_boot=10_000):
    profits = vb["profit_flat"].values
    n = len(profits)
    t_stat, p_val = sp_stats.ttest_1samp(profits, 0)
    rng = np.random.default_rng(RANDOM_STATE)
    boots = [rng.choice(profits, n, replace=True).mean() for _ in range(n_boot)]
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    sig = p_val < 0.05 and ci_lo > 0

    print(f"\n  Significance ({n} bets):")
    print(f"    t={t_stat:+.3f} p={p_val:.4f} "
          f"{'✅' if p_val < 0.05 else '⚠'}")
    print(f"    IC95=[{ci_lo:+.4f}, {ci_hi:+.4f}] "
          f"{'✅ edge confirmé' if sig else '⚠ edge non confirmé'}")
    return {"t": t_stat, "p": p_val, "ci_lo": ci_lo, "ci_hi": ci_hi, "sig": sig}


def optimize_edge(df_wf):
    rows = []
    for t in [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
        mask = (df_wf["prob_model"] - df_wf["_prob_bookie"]) >= t
        vb = df_wf[mask]
        if len(vb) < 20:
            continue
        profit = np.where(vb["_target"] == 1, vb["_odds"] - 1, -1.0)
        rows.append({
            "edge": f"{t:.0%}", "n": len(vb),
            "win": f"{vb['_target'].mean():.1%}",
            "roi": f"{profit.sum()/len(vb)*100:+.1f}%",
            "profit": f"{profit.sum():+.1f}u",
        })
    print(f"\n  Edge optimization:")
    print(pd.DataFrame(rows).to_string(index=False))


# ═══════════════════════════════════════════════════════════════
# 5. PLOTS
# ═══════════════════════════════════════════════════════════════

def plot_diagnostic(df_wf, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Calibration
    ax = axes[0, 0]
    bins = pd.cut(df_wf["prob_model"], bins=15)
    cal = df_wf.groupby(bins, observed=False).agg(
        pred=("prob_model", "mean"), actual=("_target", "mean"),
        n=("_target", "count"),
    ).dropna()
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.scatter(cal["pred"], cal["actual"],
               s=cal["n"] / cal["n"].max() * 300, alpha=0.7, color="steelblue")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Calibration")

    # PnL curves
    ax = axes[0, 1]
    for t in [0.02, 0.03, 0.05]:
        mask = (df_wf["prob_model"] - df_wf["_prob_bookie"]) >= t
        vb = df_wf[mask]
        if len(vb) > 10:
            pnl = np.cumsum(np.where(vb["_target"] == 1, vb["_odds"] - 1, -1.0))
            ax.plot(pnl, label=f"≥{t:.0%} (n={len(vb)})", alpha=0.8)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_title("Cumulative P&L"); ax.legend(fontsize=8)

    # Edge vs actual ROI
    ax = axes[1, 0]
    tmp = df_wf.copy()
    tmp["edge"] = tmp["prob_model"] - tmp["_prob_bookie"]
    tmp["pfl"] = np.where(tmp["_target"] == 1, tmp["_odds"] - 1, -1.0)
    eb = pd.cut(tmp["edge"], bins=10)
    er = tmp.groupby(eb, observed=False).agg(
        em=("edge", "mean"), roi=("pfl", "mean"), n=("pfl", "count")
    ).dropna()
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in er["roi"]]
    ax.bar(range(len(er)), er["roi"] * 100, color=colors)
    ax.set_xticks(range(len(er)))
    ax.set_xticklabels([f"{e:.2f}" for e in er["em"]], rotation=45, fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("ROI by edge bucket"); ax.set_ylabel("ROI (%)")

    # Prob distributions
    ax = axes[1, 1]
    ax.hist(df_wf["prob_model"], bins=30, alpha=0.5, label="Model",
            color="steelblue", density=True)
    ax.hist(df_wf["_prob_bookie"], bins=30, alpha=0.5, label="Bookie",
            color="darkorange", density=True)
    ax.set_title("Probability distributions"); ax.legend()

    plt.suptitle("Value Bet v3 — Diagnostic", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 {save_path}")


def export_results(vb, path):
    cols = [c for c in ["Div", "Date", "HomeTeam", "AwayTeam", "season_year",
                         "FTHG", "FTAG", "_target", "_odds", "_prob_bookie",
                         "prob_model", "edge", "profit_flat", "cum_flat",
                         "kelly_stake", "profit_kelly", "cum_kelly"] if c in vb.columns]
    out = vb[cols].rename(columns={"_target": "result", "_odds": "odds",
                                    "_prob_bookie": "prob_bookie"})
    out.to_csv(path, index=False)
    print(f"  💾 {path} ({len(out)} rows)")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--market", default="under", choices=["under", "draw"])
    p.add_argument("--leagues", default=None)
    p.add_argument("--edge", default=0.03, type=float)
    p.add_argument("--min-train", default=5, type=int)
    p.add_argument("--no-ensemble", action="store_true")
    p.add_argument("--kelly", default=0.25, type=float)
    a = p.parse_args()

    leagues = a.leagues.split(",") if a.leagues else (DIV2 if a.market == "under" else DIV1)

    print(f"\n{'='*60}")
    print(f"  VALUE BET v3 | {a.market.upper()} | {','.join(leagues)}")
    print(f"{'='*60}")

    print("\n[1/5] Loading...")
    df = load_data(a.csv, leagues)

    print("\n[2/5] Features...")
    df_feat, feat_cols = build_features(df, a.market)

    print(f"\n[3/5] Walk-forward (min_train={a.min_train})...")
    df_wf = walk_forward(df_feat, feat_cols, a.min_train, not a.no_ensemble)

    print(f"\n[4/5] Backtest (edge >= {a.edge:.0%})...")
    vb = backtest(df_wf, a.edge, a.kelly)

    if vb is not None:
        test_significance(vb)
        optimize_edge(df_wf)
        out_dir = os.path.dirname(os.path.abspath(a.csv))
        print(f"\n[5/5] Export...")
        export_results(vb, os.path.join(out_dir, f"v3_{a.market}.csv"))
        plot_diagnostic(df_wf, os.path.join(out_dir, f"v3_{a.market}_diag.png"))
    else:
        out_dir = os.path.dirname(os.path.abspath(a.csv))
        optimize_edge(df_wf)
        plot_diagnostic(df_wf, os.path.join(out_dir, f"v3_{a.market}_diag.png"))

    print(f"\n✅ Done")


if __name__ == "__main__":
    main()
