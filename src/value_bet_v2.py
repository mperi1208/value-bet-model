#!/usr/bin/env python3
"""
value_bet_v2.py — Pipeline value bet from scratch
===================================================
Architecture repensée :
  - Calibration Platt (sigmoïde) au lieu d'isotonique
  - Features réduites et ciblées (~20)
  - Pas de scale_pos_weight
  - Walk-forward strict : train | val (calibration) | test
  - Métriques : ROI flat, CLV (closing line value), Kelly
  - Stacking optionnel : XGBoost + Ridge logistic

Usage :
    python value_bet_v2.py --csv ./csv --market under
    python value_bet_v2.py --csv ./csv --market draw
    python value_bet_v2.py --csv ./csv --market under --edge 0.03
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
from sklearn.calibration import CalibratedClassifierCV
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

# Colonnes de base attendues
BASE_COLS = [
    "Div", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST", "HC", "AC",
    "HY", "AY", "HR", "AR",
]

# Colonnes de cotes à normaliser
ODDS_COLS_MAP = {
    "AvgH": ["AvgH", "BbAvH"],
    "AvgD": ["AvgD", "BbAvD"],
    "AvgA": ["AvgA", "BbAvA"],
    "Avg>2.5": ["Avg>2.5", "BbAv>2.5"],
    "Avg<2.5": ["Avg<2.5", "BbAv<2.5"],
    "MaxH": ["MaxH", "BbMxH"],
    "MaxD": ["MaxD", "BbMxD"],
    "MaxA": ["MaxA", "BbMxA"],
    "Max>2.5": ["Max>2.5", "BbMx>2.5"],
    "Max<2.5": ["Max<2.5", "BbMx<2.5"],
}

RANDOM_STATE = 42


# ═══════════════════════════════════════════════════════════════
# 0. DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_data(csv_root: str, leagues: List[str]) -> pd.DataFrame:
    """Charge tous les CSV, normalise colonnes, trie par date."""
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
                print(f"  Skip: {fpath}")
                continue

            # Colonnes de base
            available = [c for c in BASE_COLS if c in df_raw.columns]
            df = df_raw[available].copy()

            # Normalisation des cotes
            for target_col, candidates in ODDS_COLS_MAP.items():
                df[target_col] = np.nan
                for src in candidates:
                    if src in df_raw.columns:
                        df[target_col] = pd.to_numeric(df_raw[src], errors="coerce")
                        break

            # Métadonnées
            df["_file"] = os.path.basename(fpath)
            frames.append(df)

    if not frames:
        raise ValueError(f"Aucun CSV trouvé sous {csv_root}")

    df_all = pd.concat(frames, ignore_index=True)

    # Nettoyage
    df_all["Date"] = pd.to_datetime(df_all["Date"], format="mixed",
                                     dayfirst=True, errors="coerce")
    df_all = df_all.dropna(subset=["Date", "FTR", "FTHG", "FTAG"])
    df_all["FTHG"] = df_all["FTHG"].astype(int)
    df_all["FTAG"] = df_all["FTAG"].astype(int)
    for c in ["HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"]:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    df_all = df_all.sort_values(["Div", "Date"]).reset_index(drop=True)

    # Saisons
    df_all["season_year"] = df_all["Date"].apply(
        lambda d: d.year if d.month >= 7 else d.year - 1
    )
    years = sorted(df_all["season_year"].unique(), reverse=True)
    df_all["season"] = df_all["season_year"].map(
        {y: i + 1 for i, y in enumerate(years)}
    )

    print(f"  Chargé: {len(df_all)} matchs | {df_all['season'].nunique()} saisons "
          f"| {df_all['Div'].nunique()} ligues")
    return df_all


# ═══════════════════════════════════════════════════════════════
# 1. FEATURE ENGINEERING — LEAN
# ═══════════════════════════════════════════════════════════════

def _team_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features par équipe, en long format puis pivot.
    Principe : shift(1) systématique → pas de leakage.
    Fenêtres : 5 et 10 matchs seulement (pas de 3 — trop bruyant).
    """
    df = df.copy()

    # Long format
    home = df[["Div", "season", "Date", "HomeTeam",
               "FTHG", "FTAG", "FTR"]].copy()
    home.columns = ["Div", "season", "Date", "team",
                    "scored", "conceded", "FTR"]
    home["is_home"] = True
    home["idx"] = df.index

    away = df[["Div", "season", "Date", "AwayTeam",
               "FTAG", "FTHG", "FTR"]].copy()
    away.columns = ["Div", "season", "Date", "team",
                    "scored", "conceded", "FTR"]
    away["is_home"] = False
    away["idx"] = df.index

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["Div", "season", "team", "Date"]).reset_index(drop=True)

    # Derived columns
    long["total"] = long["scored"] + long["conceded"]
    long["is_under"] = (long["total"] < 2.5).astype(float)
    long["is_draw"] = (long["FTR"] == "D").astype(float)
    long["win"] = (
        ((long["is_home"]) & (long["FTR"] == "H")) |
        ((~long["is_home"]) & (long["FTR"] == "A"))
    ).astype(float)

    grp = ["Div", "season", "team"]
    features = {}

    for w in [5, 10]:
        for col in ["scored", "conceded", "total", "is_under", "is_draw", "win"]:
            feat_name = f"{col}_r{w}"
            shifted = long.groupby(grp, group_keys=False)[col].shift(1)
            features[feat_name] = (
                shifted
                .groupby([long["Div"], long["season"], long["team"]])
                .transform(lambda x: x.rolling(w, min_periods=max(2, w // 2)).mean())
            )

    # Variance des buts sur 10 matchs
    shifted_total = long.groupby(grp, group_keys=False)["total"].shift(1)
    features["total_var_10"] = (
        shifted_total
        .groupby([long["Div"], long["season"], long["team"]])
        .transform(lambda x: x.rolling(10, min_periods=3).std())
    )

    feat_df = pd.DataFrame(features, index=long.index)
    feat_df["idx"] = long["idx"]
    feat_df["is_home"] = long["is_home"]

    # Pivot home/away
    home_feat = feat_df[feat_df["is_home"]].drop(columns="is_home").set_index("idx")
    away_feat = feat_df[~feat_df["is_home"]].drop(columns="is_home").set_index("idx")

    df = df.join(home_feat.add_prefix("h_"), how="left")
    df = df.join(away_feat.add_prefix("a_"), how="left")

    return df


def _ranking_features(df: pd.DataFrame) -> pd.DataFrame:
    """Classement dynamique — simplifié."""
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
            stats[h]["gd"] += (hg - ag)
            stats[a]["gd"] += (ag - hg)

    df["h_rank"] = pd.Series(h_ranks)
    df["a_rank"] = pd.Series(a_ranks)
    return df


def _implied_probs(df: pd.DataFrame) -> pd.DataFrame:
    """No-vig probabilities pour les 3 marchés."""
    df = df.copy()

    # 1X2
    for cols, prefix in [
        (["AvgH", "AvgD", "AvgA"], "nv_"),
    ]:
        if all(c in df.columns for c in cols):
            raws = [1.0 / df[c].replace(0, np.nan) for c in cols]
            margin = sum(raws)
            df[f"{prefix}home"] = raws[0] / margin
            df[f"{prefix}draw"] = raws[1] / margin
            df[f"{prefix}away"] = raws[2] / margin

    # O/U 2.5
    for over_c, under_c, prefix in [
        ("Avg>2.5", "Avg<2.5", "nv_ou_"),
    ]:
        if over_c in df.columns and under_c in df.columns:
            ro = 1.0 / df[over_c].replace(0, np.nan)
            ru = 1.0 / df[under_c].replace(0, np.nan)
            m = ro + ru
            df[f"{prefix}over"] = ro / m
            df[f"{prefix}under"] = ru / m

    return df


def build_features(df: pd.DataFrame, market: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Pipeline features complet. Retourne (df, feature_cols).
    market: 'under' ou 'draw'
    """
    df = _team_rolling_features(df)
    df = _ranking_features(df)
    df = _implied_probs(df)

    # Targets
    df["total_goals"] = df["FTHG"] + df["FTAG"]
    df["under_25"] = (df["total_goals"] < 2.5).astype(int)
    df["draw"] = (df["FTR"] == "D").astype(int)

    # Combined features
    for w in [5, 10]:
        df[f"comb_under_r{w}"] = (
            df[f"h_is_under_r{w}"].fillna(0.5) +
            df[f"a_is_under_r{w}"].fillna(0.5)
        ) / 2
        df[f"comb_draw_r{w}"] = (
            df[f"h_is_draw_r{w}"].fillna(0.26) +
            df[f"a_is_draw_r{w}"].fillna(0.26)
        ) / 2
        df[f"comb_goals_r{w}"] = (
            df[f"h_total_r{w}"].fillna(2.7) +
            df[f"a_total_r{w}"].fillna(2.7)
        ) / 2
        df[f"goal_diff_gap_r{w}"] = abs(
            (df[f"h_scored_r{w}"].fillna(1.3) - df[f"h_conceded_r{w}"].fillna(1.3)) -
            (df[f"a_scored_r{w}"].fillna(1.3) - df[f"a_conceded_r{w}"].fillna(1.3))
        )

    df["rank_diff"] = df["a_rank"].fillna(10) - df["h_rank"].fillna(10)
    df["rank_sum"] = df["h_rank"].fillna(10) + df["a_rank"].fillna(10)

    # Feature selection selon le marché
    if market == "under":
        feature_cols = [
            # Rolling 5
            "h_scored_r5", "h_conceded_r5", "h_is_under_r5",
            "a_scored_r5", "a_conceded_r5", "a_is_under_r5",
            # Rolling 10
            "h_scored_r10", "h_conceded_r10", "h_is_under_r10",
            "a_scored_r10", "a_conceded_r10", "a_is_under_r10",
            # Combined
            "comb_under_r5", "comb_under_r10",
            "comb_goals_r5", "comb_goals_r10",
            "goal_diff_gap_r5",
            # Variance
            "h_total_var_10", "a_total_var_10",
            # Ranking
            "rank_diff", "rank_sum",
        ]
        target = "under_25"
        odds_col = "Avg<2.5"
        prob_bookie_col = "nv_ou_under"

    elif market == "draw":
        feature_cols = [
            # Rolling 5
            "h_is_draw_r5", "a_is_draw_r5",
            "h_win_r5", "a_win_r5",
            "h_scored_r5", "a_scored_r5",
            # Rolling 10
            "h_is_draw_r10", "a_is_draw_r10",
            "h_win_r10", "a_win_r10",
            "h_scored_r10", "a_scored_r10",
            # Combined
            "comb_draw_r5", "comb_draw_r10",
            "goal_diff_gap_r5", "goal_diff_gap_r10",
            "comb_goals_r5",
            # Ranking
            "rank_diff", "rank_sum",
        ]
        target = "draw"
        odds_col = "AvgD"
        prob_bookie_col = "nv_draw"
    else:
        raise ValueError(f"Market inconnu: {market}")

    # Vérifier que toutes les features existent
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  ⚠ Features manquantes: {missing}")
        feature_cols = [c for c in feature_cols if c in df.columns]

    df["_target"] = df[target]
    df["_odds"] = df[odds_col]
    df["_prob_bookie"] = df[prob_bookie_col]

    n_before = len(df)
    df = df.dropna(subset=["_target", "_odds", "_prob_bookie"]).copy()
    print(f"  Features: {len(feature_cols)} cols | {len(df)} matchs "
          f"(dropped {n_before - len(df)} sans cotes)")

    return df, feature_cols


# ═══════════════════════════════════════════════════════════════
# 2. MODEL — XGBoost + Platt calibration
# ═══════════════════════════════════════════════════════════════

def _make_xgb():
    """XGBoost conservateur — pas de scale_pos_weight."""
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,           # moins profond = moins d'overfitting
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=15,   # plus conservateur
        reg_alpha=0.5,
        reg_lambda=3.0,
        gamma=1.0,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _make_logreg():
    """Logistic regression comme baseline / ensemble member."""
    return LogisticRegression(
        C=0.1, max_iter=1000, random_state=RANDOM_STATE
    )


def train_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    use_ensemble: bool = True
) -> dict:
    """
    Entraîne XGBoost + (optionnel) LogReg, calibre via Platt sur val.
    Retourne un dict avec les modèles et le scaler.
    """
    # Scaler pour LogReg
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(np.nan_to_num(X_train, nan=0.0))
    X_val_sc = scaler.transform(np.nan_to_num(X_val, nan=0.0))

    # XGBoost
    xgb_model = _make_xgb()
    xgb_model.fit(
        np.nan_to_num(X_train, nan=0.0), y_train,
        eval_set=[(np.nan_to_num(X_val, nan=0.0), y_val)],
        verbose=False,
    )

    # Probas brutes sur val
    xgb_val_probs = xgb_model.predict_proba(
        np.nan_to_num(X_val, nan=0.0)
    )[:, 1]

    if use_ensemble:
        # LogReg
        lr_model = _make_logreg()
        lr_model.fit(X_train_sc, y_train)
        lr_val_probs = lr_model.predict_proba(X_val_sc)[:, 1]

        # Blend brut (avant calibration)
        blend_val = 0.7 * xgb_val_probs + 0.3 * lr_val_probs
    else:
        lr_model = None
        blend_val = xgb_val_probs

    # Calibration Platt sur le blend
    # On utilise une LogReg sur les probas brutes → probas calibrées
    calib_model = LogisticRegression(C=1e10, max_iter=1000)  # pas de régul
    calib_model.fit(blend_val.reshape(-1, 1), y_val)

    return {
        "xgb": xgb_model,
        "lr": lr_model,
        "scaler": scaler,
        "calibrator": calib_model,
        "use_ensemble": use_ensemble,
    }


def predict_fold(model_dict: dict, X: np.ndarray) -> np.ndarray:
    """Prédit les probas calibrées."""
    X_clean = np.nan_to_num(X, nan=0.0)

    xgb_probs = model_dict["xgb"].predict_proba(X_clean)[:, 1]

    if model_dict["use_ensemble"] and model_dict["lr"] is not None:
        X_sc = model_dict["scaler"].transform(X_clean)
        lr_probs = model_dict["lr"].predict_proba(X_sc)[:, 1]
        blend = 0.7 * xgb_probs + 0.3 * lr_probs
    else:
        blend = xgb_probs

    # Calibration Platt
    calibrated = model_dict["calibrator"].predict_proba(
        blend.reshape(-1, 1)
    )[:, 1]

    return calibrated


# ═══════════════════════════════════════════════════════════════
# 3. WALK-FORWARD
# ═══════════════════════════════════════════════════════════════

def walk_forward(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_train_seasons: int = 5,
    use_ensemble: bool = True,
) -> pd.DataFrame:
    """
    Walk-forward strict :
      - Train: saisons [test+2, ...]  (toutes les saisons avant val)
      - Val:   saison test+1           (calibration uniquement)
      - Test:  saison test             (évaluation)

    season=1 = plus récente, season=N = plus ancienne.
    """
    n_seasons = df["season"].max()
    all_preds = []
    fold_metrics = []

    for test_s in range(1, n_seasons - min_train_seasons):
        val_s = test_s + 1
        train_start = test_s + 2

        train = df[df["season"] >= train_start].dropna(
            subset=feature_cols + ["_target"]
        )
        val = df[df["season"] == val_s].dropna(
            subset=feature_cols + ["_target"]
        )
        test = df[df["season"] == test_s].dropna(
            subset=feature_cols + ["_target"]
        )

        if len(train) < 500 or len(val) < 100 or len(test) < 50:
            continue

        X_train = train[feature_cols].values
        y_train = train["_target"].values
        X_val = val[feature_cols].values
        y_val = val["_target"].values
        X_test = test[feature_cols].values
        y_test = test["_target"].values

        model = train_fold(X_train, y_train, X_val, y_val,
                           use_ensemble=use_ensemble)
        probs = predict_fold(model, X_test)

        # Métriques du fold
        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
        brier = brier_score_loss(y_test, probs)

        test_out = test.copy()
        test_out["prob_model"] = probs
        all_preds.append(test_out)

        sy = test["season_year"].iloc[0] if "season_year" in test.columns else test_s
        fold_metrics.append({
            "fold": len(fold_metrics) + 1,
            "season": sy,
            "train_n": len(train),
            "val_n": len(val),
            "test_n": len(test),
            "auc": auc,
            "brier": brier,
        })

        print(f"  Fold {fold_metrics[-1]['fold']:>2} | season={sy} | "
              f"train={len(train):>5} val={len(val):>4} test={len(test):>4} | "
              f"AUC={auc:.3f} Brier={brier:.4f}")

    if not all_preds:
        raise ValueError("Aucun fold valide dans le walk-forward")

    df_wf = pd.concat(all_preds, ignore_index=True)
    metrics_df = pd.DataFrame(fold_metrics)

    overall_auc = roc_auc_score(df_wf["_target"], df_wf["prob_model"])
    print(f"\n  Walk-forward total: {len(df_wf)} matchs | "
          f"AUC={overall_auc:.4f} | "
          f"Brier={brier_score_loss(df_wf['_target'], df_wf['prob_model']):.4f}")

    return df_wf


# ═══════════════════════════════════════════════════════════════
# 4. BACKTEST — EDGE DETECTION + ROI + CLV
# ═══════════════════════════════════════════════════════════════

def backtest(
    df_wf: pd.DataFrame,
    edge_min: float = 0.03,
    kelly_fraction: float = 0.25,
) -> Optional[pd.DataFrame]:
    """
    Détecte les value bets, calcule ROI flat et Kelly.
    Calcule le CLV (closing line value) comme métrique primaire.
    """
    df = df_wf.copy()
    df["edge"] = df["prob_model"] - df["_prob_bookie"]

    # Value bets
    vb = df[df["edge"] >= edge_min].copy().reset_index(drop=True)
    n_vb = len(vb)
    n_total = len(df)

    if n_vb == 0:
        print(f"\n  Aucun value bet avec edge >= {edge_min:.0%}")
        return None

    print(f"\n  Value bets: {n_vb}/{n_total} ({n_vb/n_total:.1%}) | "
          f"edge >= {edge_min:.0%}")

    # ROI flat stake
    vb["profit_flat"] = np.where(
        vb["_target"] == 1,
        vb["_odds"] - 1,
        -1.0,
    )
    vb["cum_profit_flat"] = vb["profit_flat"].cumsum()
    roi_flat = vb["profit_flat"].sum() / n_vb * 100

    # ROI Kelly fractionnaire
    vb["kelly_stake"] = np.clip(
        kelly_fraction * (
            (vb["prob_model"] * vb["_odds"] - 1) / (vb["_odds"] - 1)
        ),
        0, 0.05,  # cap à 5% du bankroll par pari
    )
    vb["profit_kelly"] = np.where(
        vb["_target"] == 1,
        vb["kelly_stake"] * (vb["_odds"] - 1),
        -vb["kelly_stake"],
    )
    vb["cum_profit_kelly"] = vb["profit_kelly"].cumsum()
    roi_kelly = vb["profit_kelly"].sum() / vb["kelly_stake"].sum() * 100

    # CLV (Closing Line Value)
    # Le modèle bat-il la ligne ? prob_model > prob_bookie en moyenne ?
    vb["clv"] = vb["prob_model"] - vb["_prob_bookie"]
    mean_clv = vb["clv"].mean()

    # Calibration check sur les value bets
    actual_wr = vb["_target"].mean()
    predicted_wr = vb["prob_model"].mean()
    avg_odds = vb["_odds"].mean()
    breakeven_wr = 1 / avg_odds

    # Win rate
    win_rate = actual_wr

    print(f"\n  {'='*55}")
    print(f"  BACKTEST RESULTS | {n_vb} paris")
    print(f"  {'='*55}")
    print(f"  Win rate:      {win_rate:.1%} (breakeven: {breakeven_wr:.1%})")
    print(f"  Avg odds:      {avg_odds:.2f}")
    print(f"  Avg edge:      {vb['edge'].mean():.1%}")
    print(f"  ROI flat:      {roi_flat:+.2f}%")
    print(f"  ROI Kelly:     {roi_kelly:+.2f}%")
    print(f"  CLV moyen:     {mean_clv:+.3f}")
    print(f"  Calibration:   predicted={predicted_wr:.1%} vs actual={actual_wr:.1%} "
          f"(delta={actual_wr - predicted_wr:+.1%})")

    # Par ligue
    print(f"\n  Par ligue:")
    for div, g in vb.groupby("Div"):
        r = g["profit_flat"].sum() / len(g) * 100
        wr = g["_target"].mean()
        pred_wr = g["prob_model"].mean()
        print(f"    {div:>4}: {len(g):>4} paris | win {wr:.0%} | "
              f"pred {pred_wr:.0%} | ROI {r:+.1f}%")

    return vb


# ═══════════════════════════════════════════════════════════════
# 5. SIGNIFICANCE TESTS
# ═══════════════════════════════════════════════════════════════

def test_significance(vb: pd.DataFrame, n_bootstrap: int = 10_000) -> dict:
    """T-test + bootstrap IC 95% sur les profits flat."""
    profits = vb["profit_flat"].values
    n = len(profits)
    roi_mean = profits.mean()

    t_stat, p_value = sp_stats.ttest_1samp(profits, popmean=0)

    rng = np.random.default_rng(RANDOM_STATE)
    boot_means = np.array([
        rng.choice(profits, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    print(f"\n  Significativité ({n} paris):")
    print(f"    ROI moyen/pari: {roi_mean:+.4f}u")
    print(f"    t-stat:         {t_stat:+.3f}")
    print(f"    p-value:        {p_value:.4f} "
          f"{'✅ sig. 5%' if p_value < 0.05 else '⚠ non sig.'}")
    print(f"    IC 95%:         [{ci_low:+.4f}, {ci_high:+.4f}]")
    sig = p_value < 0.05 and ci_low > 0
    print(f"    {'✅ Edge confirmé' if sig else '⚠ Edge non confirmé'}")

    return {
        "n": n, "roi_mean": roi_mean, "t_stat": t_stat,
        "p_value": p_value, "ci_low": ci_low, "ci_high": ci_high,
        "significant": sig,
    }


# ═══════════════════════════════════════════════════════════════
# 6. EDGE THRESHOLD OPTIMIZATION
# ═══════════════════════════════════════════════════════════════

def optimize_edge(df_wf: pd.DataFrame,
                  thresholds: Optional[List[float]] = None) -> pd.DataFrame:
    """Teste plusieurs seuils d'edge."""
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]

    rows = []
    for t in thresholds:
        mask = (df_wf["prob_model"] - df_wf["_prob_bookie"]) >= t
        vb = df_wf[mask]
        if len(vb) < 20:
            continue
        profit = np.where(vb["_target"] == 1, vb["_odds"] - 1, -1.0)
        roi = profit.sum() / len(vb) * 100
        wr = vb["_target"].mean()
        rows.append({
            "edge_min": f"{t:.0%}",
            "n_bets": len(vb),
            "win_rate": f"{wr:.1%}",
            "roi": f"{roi:+.1f}%",
            "profit": f"{profit.sum():+.1f}u",
        })

    result = pd.DataFrame(rows)
    print(f"\n  Optimisation seuil d'edge:")
    print(result.to_string(index=False))
    return result


# ═══════════════════════════════════════════════════════════════
# 7. CALIBRATION DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════

def calibration_diagnostic(df_wf: pd.DataFrame, save_path: str):
    """
    Diagnostic visuel complet: calibration, PnL, edge distribution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Calibration curve
    ax = axes[0, 0]
    bins = pd.cut(df_wf["prob_model"], bins=15)
    cal = df_wf.groupby(bins, observed=False).agg(
        pred=("prob_model", "mean"),
        actual=("_target", "mean"),
        n=("_target", "count"),
    ).dropna()
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Parfait")
    ax.scatter(cal["pred"], cal["actual"],
               s=cal["n"] / cal["n"].max() * 300,
               alpha=0.7, color="steelblue", label="Modèle")
    ax.set_xlabel("Prob prédite")
    ax.set_ylabel("Fréquence réelle")
    ax.set_title("Courbe de calibration")
    ax.legend()

    # 2. Profit cumulé (si value bets existent)
    ax = axes[0, 1]
    for edge_t in [0.02, 0.03, 0.05]:
        mask = (df_wf["prob_model"] - df_wf["_prob_bookie"]) >= edge_t
        vb = df_wf[mask].copy()
        if len(vb) > 10:
            profit = np.where(vb["_target"] == 1, vb["_odds"] - 1, -1.0)
            cum = np.cumsum(profit)
            ax.plot(cum, label=f"edge≥{edge_t:.0%} (n={len(vb)})", alpha=0.8)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("Profit cumulé par seuil")
    ax.set_xlabel("Pari n°")
    ax.set_ylabel("Unités")
    ax.legend(fontsize=8)

    # 3. Edge vs ROI réalisé par bucket
    ax = axes[1, 0]
    df_wf_tmp = df_wf.copy()
    df_wf_tmp["edge"] = df_wf_tmp["prob_model"] - df_wf_tmp["_prob_bookie"]
    df_wf_tmp["profit_flat"] = np.where(
        df_wf_tmp["_target"] == 1, df_wf_tmp["_odds"] - 1, -1.0
    )
    edge_bins = pd.cut(df_wf_tmp["edge"], bins=10)
    edge_roi = df_wf_tmp.groupby(edge_bins, observed=False).agg(
        edge_mean=("edge", "mean"),
        roi=("profit_flat", "mean"),
        n=("profit_flat", "count"),
    ).dropna()
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in edge_roi["roi"]]
    ax.bar(range(len(edge_roi)), edge_roi["roi"] * 100, color=colors, alpha=0.8)
    ax.set_xticks(range(len(edge_roi)))
    ax.set_xticklabels([f"{e:.2f}" for e in edge_roi["edge_mean"]],
                       rotation=45, fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("ROI par bucket d'edge")
    ax.set_xlabel("Edge moyen")
    ax.set_ylabel("ROI (%)")

    # 4. Distribution prob_model vs prob_bookie
    ax = axes[1, 1]
    ax.hist(df_wf["prob_model"], bins=30, alpha=0.5, label="Modèle",
            color="steelblue", density=True)
    ax.hist(df_wf["_prob_bookie"], bins=30, alpha=0.5, label="Bookie",
            color="darkorange", density=True)
    ax.set_title("Distribution des probabilités")
    ax.set_xlabel("Probabilité")
    ax.legend()

    plt.suptitle("Diagnostic complet — Value Bet v2", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Diagnostic sauvegardé → {save_path}")


# ═══════════════════════════════════════════════════════════════
# 8. EXPORT
# ═══════════════════════════════════════════════════════════════

def export_results(vb: pd.DataFrame, save_path: str):
    """Export CSV des value bets."""
    cols = []
    for c in ["Div", "Date", "HomeTeam", "AwayTeam", "season_year",
              "FTHG", "FTAG", "total_goals", "_target", "_odds",
              "_prob_bookie", "prob_model", "edge",
              "profit_flat", "cum_profit_flat",
              "kelly_stake", "profit_kelly", "cum_profit_kelly"]:
        if c in vb.columns:
            cols.append(c)

    df_out = vb[cols].copy()
    rename = {
        "_target": "result", "_odds": "odds",
        "_prob_bookie": "prob_bookie",
    }
    df_out = df_out.rename(columns={k: v for k, v in rename.items()
                                     if k in df_out.columns})
    df_out.to_csv(save_path, index=False)
    print(f"  💾 Export → {save_path} ({len(df_out)} lignes)")


# ═══════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Value Bet Pipeline v2 — From Scratch"
    )
    parser.add_argument("--csv", required=True, help="Dossier racine des CSV")
    parser.add_argument("--market", default="under",
                        choices=["under", "draw"],
                        help="Marché cible (under/draw)")
    parser.add_argument("--leagues", default=None,
                        help="Ligues (comma-sep, ex: D2,E1). Défaut: Div2 pour under, Div1 pour draw")
    parser.add_argument("--edge", default=0.03, type=float,
                        help="Seuil d'edge minimum")
    parser.add_argument("--min-train", default=5, type=int,
                        help="Nombre minimum de saisons d'entraînement")
    parser.add_argument("--no-ensemble", action="store_true",
                        help="XGBoost seul (pas d'ensemble avec LogReg)")
    parser.add_argument("--kelly", default=0.25, type=float,
                        help="Fraction de Kelly (0.25 = quarter Kelly)")
    args = parser.parse_args()

    # Ligues par défaut
    if args.leagues:
        leagues = args.leagues.split(",")
    elif args.market == "under":
        leagues = DIV2
    else:
        leagues = DIV1

    print("\n" + "=" * 60)
    print(f"  VALUE BET v2 | Market: {args.market.upper()} | "
          f"Leagues: {','.join(leagues)}")
    print("=" * 60)

    # 1. Load
    print("\n[1/5] Chargement des données...")
    df = load_data(args.csv, leagues)

    # 2. Features
    print("\n[2/5] Feature engineering...")
    df_feat, feature_cols = build_features(df, args.market)

    # 3. Walk-forward
    print(f"\n[3/5] Walk-forward (min_train={args.min_train})...")
    df_wf = walk_forward(
        df_feat, feature_cols,
        min_train_seasons=args.min_train,
        use_ensemble=not args.no_ensemble,
    )

    # 4. Backtest
    print(f"\n[4/5] Backtest (edge >= {args.edge:.0%})...")
    vb = backtest(df_wf, edge_min=args.edge, kelly_fraction=args.kelly)

    if vb is not None:
        test_significance(vb)
        optimize_edge(df_wf)

        # 5. Export
        print(f"\n[5/5] Export & diagnostic...")
        out_dir = os.path.dirname(os.path.abspath(args.csv))
        export_results(vb, os.path.join(out_dir, f"vb2_{args.market}.csv"))
        calibration_diagnostic(
            df_wf,
            os.path.join(out_dir, f"vb2_{args.market}_diagnostic.png"),
        )
    else:
        # Quand même générer le diagnostic
        print(f"\n[5/5] Diagnostic (pas de value bets)...")
        out_dir = os.path.dirname(os.path.abspath(args.csv))
        calibration_diagnostic(
            df_wf,
            os.path.join(out_dir, f"vb2_{args.market}_diagnostic.png"),
        )
        optimize_edge(df_wf)

    print(f"\n✅ Terminé")


if __name__ == "__main__":
    main()
