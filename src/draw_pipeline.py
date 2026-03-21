"""
draw_pipeline.py
================
Value bet detection — Match Draw market (Div1)
XGBoost + sigmoid calibration (cv=3) + walk-forward validation

Note: this pipeline uses Platt scaling (sigmoid) with k-fold cross-validation
(cv=3), which avoids the overfitting observed with isotonic calibration on
small calibration sets (spurious +8.81% ROI in earlier versions).
See README for the full analysis.

Usage:
    python3 draw_pipeline.py --data-dir "/path/to/csv/"
    python3 draw_pipeline.py --data-dir "./csv" --train-window 5 --edge 0.05
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

LEAGUE_MAP = {
    "E0":  "Premier League",
    "SP1": "La Liga",
    "D1":  "Bundesliga",
    "I1":  "Serie A",
    "F1":  "Ligue 1",
}

if "--data-dir" in sys.argv:
    idx      = sys.argv.index("--data-dir")
    DATA_DIR = sys.argv[idx + 1]
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# TRAIN_WINDOW paramétrable via CLI (v2)
if "--train-window" in sys.argv:
    idx          = sys.argv.index("--train-window")
    TRAIN_WINDOW = int(sys.argv[idx + 1])
else:
    TRAIN_WINDOW = 3

if "--edge" in sys.argv:
    idx      = sys.argv.index("--edge")
    MIN_EDGE = float(sys.argv[idx + 1])
else:
    MIN_EDGE = 0.04

STAKE        = 10
ROLLING_WINDOWS = [3, 5, 10]
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_all_data() -> pd.DataFrame:
    frames = []
    for league in LEAGUE_MAP:
        for fp in sorted(glob.glob(os.path.join(DATA_DIR, league, "*.csv"))):
            if "_enriched" in os.path.basename(fp):
                continue
            try:
                df = pd.read_csv(fp, encoding="latin1")
                df = df.dropna(subset=["FTHG", "FTAG", "FTR"])
                df["league"]   = league
                df["filename"] = os.path.basename(fp)
                frames.append(df)
            except Exception:
                pass

    if not frames:
        raise FileNotFoundError(
            f"No CSV files found under {DATA_DIR}/E0/, {DATA_DIR}/I1/, etc."
        )
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────
# 2. TARGET & IMPLIED PROBABILITY
# ─────────────────────────────────────────────

def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["draw"] = (df["FTR"] == "D").astype(int)

    if "AvgH" in df.columns and "AvgD" in df.columns and "AvgA" in df.columns:
        raw_h = 1 / df["AvgH"].replace(0, np.nan)
        raw_d = 1 / df["AvgD"].replace(0, np.nan)
        raw_a = 1 / df["AvgA"].replace(0, np.nan)
        margin = raw_h + raw_d + raw_a
        df["implied_prob_draw"] = raw_d / margin
        df["odds_draw"]         = df["AvgD"]
    else:
        df["implied_prob_draw"] = np.nan
        df["odds_draw"]         = np.nan

    return df


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Correction v2 : suppression des colonnes HS/AS/HST/AST/HC/AC.
    Ces stats sont mesurées PENDANT le match → leakage post-match.
    Seules les stats de l'historique des équipes sont utilisées.
    """
    df = build_target(df_raw)

    if "Date" in df.columns:
        df["_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.sort_values(["league", "filename", "_date"]).reset_index(drop=True)
    else:
        df = df.sort_values(["league", "filename"]).reset_index(drop=True)

    # Format long : une ligne par (match, équipe)
    home = df[["league", "filename", "_date", "HomeTeam", "FTHG", "FTAG", "FTR"]].copy()
    home.columns = ["league", "filename", "_date", "team", "scored", "conceded", "FTR"]
    home["is_home"]  = True
    home["orig_idx"] = df.index

    away = df[["league", "filename", "_date", "AwayTeam", "FTAG", "FTHG", "FTR"]].copy()
    away.columns = ["league", "filename", "_date", "team", "scored", "conceded", "FTR"]
    away["is_home"]  = False
    away["orig_idx"] = df.index

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["league", "filename", "team", "_date"]).reset_index(drop=True)

    long["is_draw"]   = (long["FTR"] == "D").astype(float)
    long["is_win"]    = (
        (long["is_home"] & (long["FTR"] == "H")) |
        (~long["is_home"] & (long["FTR"] == "A"))
    ).astype(float)
    long["is_loss"]   = (
        (long["is_home"] & (long["FTR"] == "A")) |
        (~long["is_home"] & (long["FTR"] == "H"))
    ).astype(float)
    long["goal_diff"] = long["scored"] - long["conceded"]

    grp      = ["league", "filename", "team"]
    raw_cols = ["is_draw", "is_win", "is_loss", "scored", "conceded", "goal_diff"]
    col_map  = {
        "is_draw":   "draw_rate",
        "is_win":    "win_rate",
        "is_loss":   "loss_rate",
        "scored":    "avg_scored",
        "conceded":  "avg_conceded",
        "goal_diff": "avg_goal_diff",
    }

    # Rolling stats vectorisées (shift(1) pour éviter le data leakage)
    for w in ROLLING_WINDOWS:
        for col in raw_cols:
            long[f"_{col}_{w}"] = (
                long.groupby(grp, group_keys=False)[col]
                .transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean())
            )

    roll_cols   = [f"_{col}_{w}" for w in ROLLING_WINDOWS for col in raw_cols]
    long_sub    = long[["orig_idx", "is_home"] + roll_cols]

    home_rename = {f"_{col}_{w}": f"home_{name}_{w}"
                   for w in ROLLING_WINDOWS for col, name in col_map.items()}
    away_rename = {f"_{col}_{w}": f"away_{name}_{w}"
                   for w in ROLLING_WINDOWS for col, name in col_map.items()}

    home_stats = (long_sub[long_sub["is_home"]]
                  .drop(columns="is_home").set_index("orig_idx")
                  .rename(columns=home_rename))
    away_stats = (long_sub[~long_sub["is_home"]]
                  .drop(columns="is_home").set_index("orig_idx")
                  .rename(columns=away_rename))

    df = df.join(home_stats).join(away_stats)

    for w in ROLLING_WINDOWS:
        df[f"combined_draw_rate_{w}"] = (
            (df[f"home_draw_rate_{w}"].fillna(0) + df[f"away_draw_rate_{w}"].fillna(0)) / 2
        )
        df[f"goal_diff_gap_{w}"] = (
            df[f"home_avg_goal_diff_{w}"].fillna(0) - df[f"away_avg_goal_diff_{w}"].fillna(0)
        ).abs()
        df[f"combined_goals_{w}"] = (
            df[f"home_avg_scored_{w}"].fillna(0) + df[f"away_avg_scored_{w}"].fillna(0)
        )

    df["draw_target"] = df["draw"]

    le = LabelEncoder()
    df["league_enc"] = le.fit_transform(df["league"])

    feat_cols = (
        [f"home_{name}_{w}" for w in ROLLING_WINDOWS for name in col_map.values()] +
        [f"away_{name}_{w}" for w in ROLLING_WINDOWS for name in col_map.values()] +
        [f"combined_draw_rate_{w}" for w in ROLLING_WINDOWS] +
        [f"goal_diff_gap_{w}"      for w in ROLLING_WINDOWS] +
        [f"combined_goals_{w}"     for w in ROLLING_WINDOWS] +
        ["league_enc"]
    )
    keep = ["league", "filename", "draw_target", "implied_prob_draw", "odds_draw"] + feat_cols
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def get_feature_cols(df: pd.DataFrame) -> list:
    drop = {"draw_target", "implied_prob_draw", "odds_draw", "league", "filename"}
    return [c for c in df.columns if c not in drop and not c.startswith("_")]


# ─────────────────────────────────────────────
# 4. MODEL
# ─────────────────────────────────────────────

def build_model() -> CalibratedClassifierCV:
    """
    Correction v2 : random_state fixé dans CalibratedClassifierCV.
    """
    base = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=6,
        gamma=1.5,
        reg_alpha=0.2,
        reg_lambda=1.5,
        scale_pos_weight=2.0,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    # cv=3 avec random_state pour reproductibilité
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


# ─────────────────────────────────────────────
# 5. WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────

def walk_forward(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward par ligue séparément.
    Saison 1 = plus récente, N = plus ancienne (convention football-data.co.uk).
    Pour chaque saison test i, entraîne sur saisons [i+1 .. i+TRAIN_WINDOW].
    """
    import re
    def _season_num(fname):
        m = re.search(r'-(\d+)\.csv$', fname)
        return int(m.group(1)) if m else 0

    feat_cols = get_feature_cols(df_feat)
    all_preds = []
    leagues   = sorted(df_feat["league"].unique())

    print(f'  Walk-forward : TRAIN_WINDOW={TRAIN_WINDOW}, ligues={leagues}')

    for league in leagues:
        league_df = df_feat[df_feat["league"] == league]
        # filenames[0] = saison 1 (récente), filenames[-1] = saison N (ancienne)
        filenames = sorted(league_df["filename"].unique(), key=_season_num)

        for i in range(len(filenames) - TRAIN_WINDOW):
            test_file   = filenames[i]
            train_files = filenames[i + 1: i + 1 + TRAIN_WINDOW]

            train = league_df[league_df["filename"].isin(train_files)].dropna(subset=feat_cols)
            test  = league_df[league_df["filename"] == test_file].dropna(subset=feat_cols)

            if len(train) < 100 or len(test) < 20:
                continue

            X_train = train[feat_cols].values
            y_train = train["draw_target"].values
            X_test  = test[feat_cols].values

            model = build_model()
            model.fit(X_train, y_train)
            prob  = model.predict_proba(X_test)[:, 1]

            preds = test[["league", "filename", "draw_target",
                           "implied_prob_draw", "odds_draw"]].copy()
            preds["prob_draw"] = prob
            preds["test_fold"] = test_file
            all_preds.append(preds)

    if not all_preds:
        raise ValueError(
            f"Not enough files for walk-forward (need > {TRAIN_WINDOW} per league)."
        )
    return pd.concat(all_preds, ignore_index=True)


# ─────────────────────────────────────────────
# 6. VALUE BET DETECTION & SIMULATION
# ─────────────────────────────────────────────

def detect_value_bets(preds: pd.DataFrame) -> pd.DataFrame:
    df = preds.dropna(subset=["implied_prob_draw", "odds_draw"]).copy()
    df["edge"]      = df["prob_draw"] - df["implied_prob_draw"]
    df["value_bet"] = df["edge"] >= MIN_EDGE
    return df


def simulate_flat_staking(value_bets: pd.DataFrame) -> pd.DataFrame:
    bets = value_bets[value_bets["value_bet"]].copy().reset_index(drop=True)
    bets["payout"] = np.where(
        bets["draw_target"] == 1,
        (bets["odds_draw"] - 1) * STAKE,
        -STAKE
    )
    bets["cumulative_pnl"]      = bets["payout"].cumsum()
    bets["cumulative_invested"] = (np.arange(1, len(bets) + 1)) * STAKE
    bets["roi"]                 = bets["cumulative_pnl"] / bets["cumulative_invested"]
    return bets


# ─────────────────────────────────────────────
# 7. PLOTS & SUMMARY
# ─────────────────────────────────────────────

def calibration_plot(preds: pd.DataFrame, n_bins: int = 10):
    preds = preds.dropna(subset=["prob_draw"]).copy()
    preds["bin"] = pd.cut(preds["prob_draw"], bins=n_bins, labels=False)
    cal = preds.groupby("bin").agg(
        mean_pred=("prob_draw",   "mean"),
        actual=   ("draw_target", "mean"),
        count=    ("draw_target", "count"),
    ).dropna()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.scatter(cal["mean_pred"], cal["actual"],
               s=cal["count"] / cal["count"].max() * 300,
               alpha=0.8, color="steelblue", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Actual draw rate")
    ax.set_title("Calibration curve — Draw model")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(DATA_DIR, "draw_calibration.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def pnl_plot(bets: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(bets.index, bets["cumulative_pnl"], color="steelblue")
    axes[0].axhline(0, color="grey", linewidth=0.8)
    axes[0].set_title("Cumulative P&L — Draw value bets (flat stake)")
    axes[0].set_xlabel("Bet #")
    axes[0].set_ylabel("P&L (units)")
    axes[1].plot(bets.index, bets["roi"] * 100, color="darkorange")
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].set_title("Rolling ROI (%)")
    axes[1].set_xlabel("Bet #")
    axes[1].set_ylabel("ROI (%)")
    plt.tight_layout()
    out = os.path.join(DATA_DIR, "draw_pnl.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def print_summary(bets: pd.DataFrame, preds: pd.DataFrame):
    n_total  = len(preds)
    n_bets   = len(bets)
    wins     = (bets["draw_target"] == 1).sum() if n_bets else 0
    win_rate = wins / n_bets if n_bets else 0
    pnl      = bets["payout"].sum() if n_bets else 0
    roi      = pnl / (n_bets * STAKE) if n_bets else 0
    coverage = n_bets / n_total if n_total else 0

    print("\n" + "=" * 52)
    print("  DRAW VALUE BET — BACKTEST SUMMARY")
    print("=" * 52)
    print(f"  Matches evaluated    : {n_total:,}")
    print(f"  Value bets placed    : {n_bets:,}  ({coverage:.1%} of matches)")
    print(f"  Win rate (draw)      : {win_rate:.1%}")
    print(f"  Total P&L            : {pnl:+.1f} units")
    print(f"  ROI                  : {roi:+.2%}")
    print(f"  Min edge threshold   : {MIN_EDGE:.0%}")
    print("=" * 52)


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def main():
    print(f"Data directory : {DATA_DIR}")
    print(f"Train window   : {TRAIN_WINDOW} | Min edge : {MIN_EDGE:.0%}\n")

    print("Loading data...")
    df_raw = load_all_data()
    print(f"\n  {len(df_raw):,} matches — "
          f"{df_raw['league'].nunique()} leagues — "
          f"{df_raw['filename'].nunique()} files\n")

    print("Building features...")
    df_feat = build_features(df_raw)
    feat_cols = get_feature_cols(df_feat)
    print(f"  {df_feat.shape[0]:,} rows × {len(feat_cols)} features\n")

    print("Running walk-forward validation...")
    preds = walk_forward(df_feat)

    print("\nDetecting value bets...")
    preds_vb = detect_value_bets(preds)
    bets     = simulate_flat_staking(preds_vb)

    print_summary(bets, preds_vb)

    print("\nGenerating plots...")
    calibration_plot(preds)
    if len(bets):
        pnl_plot(bets)

    preds.to_csv(os.path.join(DATA_DIR, "draw_predictions.csv"), index=False)
    if len(bets):
        bets.to_csv(os.path.join(DATA_DIR, "draw_value_bets.csv"), index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
