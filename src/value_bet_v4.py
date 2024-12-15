#!/usr/bin/env python3
"""
value_bet_v4.py — Pipeline avec xG Understat
==============================================
Intègre les xG scrapés par scrape_understat.py dans le pipeline v3.

Nouvelles features xG :
  - h_xg_r5 / a_xg_r5 : xG rolling 5 matchs par équipe
  - h_xga_r5 / a_xga_r5 : xG against rolling 5
  - h_xg_diff_r5 : xG - xGA rolling (domination offensive)
  - comb_xg_r5 : xG combiné des deux équipes (proxy attaque)
  - xg_overperf_r5 : buts réels - xG rolling (sur/sous-performance)
  - comb_xg_r10, h_xg_r10, etc. : même chose sur 10 matchs

Marché cible : DRAW sur Div1 (données xG disponibles)
Peut aussi tourner en mode UNDER sur Div1.

Usage :
    python3 value_bet_v4.py --csv ./csv --xg understat_xg.csv --market draw --edge 0.03
    python3 value_bet_v4.py --csv ./csv --xg understat_xg.csv --market under --edge 0.03
"""

import os, glob, argparse, warnings
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DIV1 = ["F1", "D1", "SP1", "E0", "I1"]
DIV2 = ["D2", "E1", "F2", "I2", "SP2"]

BASE_COLS = [
    "Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
]
ODDS_LOAD = [
    "B365H","B365D","B365A","BWH","BWD","BWA","IWH","IWD","IWA",
    "PSH","PSD","PSA","WHH","WHD","WHA","VCH","VCD","VCA",
    "AvgH","AvgD","AvgA","MaxH","MaxD","MaxA",
    "B365>2.5","B365<2.5","P>2.5","P<2.5",
    "Avg>2.5","Avg<2.5","Max>2.5","Max<2.5",
    "BbAvH","BbAvD","BbAvA","BbMxH","BbMxD","BbMxA",
    "BbAv>2.5","BbAv<2.5","BbMx>2.5","BbMx<2.5",
]
RANDOM_STATE = 42


def load_matches(csv_root, leagues):
    frames = []
    for league in leagues:
        for fpath in sorted(glob.glob(os.path.join(csv_root, league, f"{league}-*.csv"))):
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    df_raw = pd.read_csv(fpath, low_memory=False, encoding=enc, on_bad_lines="skip")
                    break
                except Exception: continue
            else: continue
            available = [c for c in BASE_COLS if c in df_raw.columns]
            df = df_raw[available].copy()
            for col in ODDS_LOAD:
                if col in df_raw.columns:
                    df[col] = pd.to_numeric(df_raw[col], errors="coerce")
            legacy = {"BbAvH":"AvgH","BbAvD":"AvgD","BbAvA":"AvgA",
                      "BbMxH":"MaxH","BbMxD":"MaxD","BbMxA":"MaxA",
                      "BbAv>2.5":"Avg>2.5","BbAv<2.5":"Avg<2.5",
                      "BbMx>2.5":"Max>2.5","BbMx<2.5":"Max<2.5"}
            for old, new in legacy.items():
                if old in df.columns and (new not in df.columns or df[new].isna().all()):
                    df[new] = df[old]
            frames.append(df)
    if not frames: raise ValueError(f"Aucun CSV sous {csv_root}")
    df_all = pd.concat(frames, ignore_index=True)
    df_all["Date"] = pd.to_datetime(df_all["Date"], format="mixed", dayfirst=True, errors="coerce")
    df_all = df_all.dropna(subset=["Date","FTR","FTHG","FTAG"])
    df_all["FTHG"] = df_all["FTHG"].astype(int); df_all["FTAG"] = df_all["FTAG"].astype(int)
    for c in ["HS","AS","HST","AST","HC","AC","HF","AF","HY","AY","HR","AR"]:
        if c in df_all.columns: df_all[c] = pd.to_numeric(df_all[c], errors="coerce")
    df_all = df_all.sort_values(["Div","Date"]).reset_index(drop=True)
    df_all["season_year"] = df_all["Date"].apply(lambda d: d.year if d.month >= 7 else d.year - 1)
    years = sorted(df_all["season_year"].unique(), reverse=True)
    df_all["season"] = df_all["season_year"].map({y: i+1 for i, y in enumerate(years)})
    print(f"  Matchs: {len(df_all)} | {df_all['season'].nunique()} saisons")
    return df_all


def _normalize_name(name):
    return (name.strip().lower().replace("'","").replace(".","").replace("-"," ")
            .replace("fc ","").replace(" fc","").replace("cf ","").replace(" cf",""))

def load_and_merge_xg(df_matches, xg_path):
    if not os.path.exists(xg_path):
        print(f"  ⚠ Fichier xG non trouvé: {xg_path}"); return df_matches
    xg = pd.read_csv(xg_path)
    xg["date"] = pd.to_datetime(xg["date"])
    print(f"  xG data: {len(xg)} matchs")
    df = df_matches.copy()
    df["_date"] = df["Date"].dt.normalize(); xg["_date"] = xg["date"].dt.normalize()
    merged = df.merge(
        xg[["_date","home_fd","away_fd","xG_home","xG_away","league_fd"]],
        left_on=["_date","HomeTeam","AwayTeam"],
        right_on=["_date","home_fd","away_fd"], how="left",
    ).drop(columns=["home_fd","away_fd","league_fd"], errors="ignore")
    n1 = merged["xG_home"].notna().sum()
    print(f"  xG merge (exact): {n1}/{len(merged)} ({n1/len(merged):.0%})")

    if n1/len(merged) < 0.7:
        print("  Tentative merge normalisé...")
        df["_hn"] = df["HomeTeam"].apply(_normalize_name)
        df["_an"] = df["AwayTeam"].apply(_normalize_name)
        xg["_hn"] = xg["home_fd"].apply(_normalize_name)
        xg["_an"] = xg["away_fd"].apply(_normalize_name)
        m2 = df.merge(xg[["_date","_hn","_an","xG_home","xG_away"]],
                       on=["_date","_hn","_an"], how="left", suffixes=("","_n"))
        if "xG_home_n" in m2.columns:
            mask = merged["xG_home"].isna() & m2["xG_home_n"].notna()
            merged.loc[mask, "xG_home"] = m2.loc[mask, "xG_home_n"]
            merged.loc[mask, "xG_away"] = m2.loc[mask, "xG_away_n"]
        n2 = merged["xG_home"].notna().sum()
        print(f"  xG merge (normalisé): {n2}/{len(merged)} ({n2/len(merged):.0%})")
        # Debug unmapped
        unm = merged[merged["xG_home"].isna() & (merged["season_year"]>=2017)]
        if len(unm)>0:
            teams = sorted(pd.concat([unm["HomeTeam"],unm["AwayTeam"]]).unique()[:15])
            print(f"  ⚠ Non matchés: {teams}")

    return merged.drop(columns=["_date"], errors="ignore")


def _team_rolling(df):
    df = df.copy()
    home = df[["Div","season","Date","HomeTeam","FTHG","FTAG","FTR"]].copy()
    home.columns = ["Div","season","Date","team","scored","conceded","FTR"]
    home["is_home"]=True; home["idx"]=df.index
    away = df[["Div","season","Date","AwayTeam","FTAG","FTHG","FTR"]].copy()
    away.columns = ["Div","season","Date","team","scored","conceded","FTR"]
    away["is_home"]=False; away["idx"]=df.index
    has_xg = "xG_home" in df.columns and df["xG_home"].notna().any()
    if has_xg:
        home["xg"]=df["xG_home"]; home["xga"]=df["xG_away"]
        away["xg"]=df["xG_away"]; away["xga"]=df["xG_home"]
    if "HST" in df.columns:
        home["sot"]=df["HST"]; away["sot"]=df["AST"]
    long = pd.concat([home,away],ignore_index=True)
    long = long.sort_values(["Div","season","team","Date"]).reset_index(drop=True)
    long["total"]=long["scored"]+long["conceded"]
    long["is_under"]=(long["total"]<2.5).astype(float)
    long["is_draw"]=(long["FTR"]=="D").astype(float)
    long["win"]=(((long["is_home"])&(long["FTR"]=="H"))|((~long["is_home"])&(long["FTR"]=="A"))).astype(float)
    if has_xg:
        long["xg_diff"]=long["xg"]-long["xga"]
        long["xg_overperf"]=long["scored"]-long["xg"]
    grp=["Div","season","team"]; features={}
    stat_cols=["scored","conceded","total","is_under","is_draw","win"]
    if has_xg: stat_cols+=["xg","xga","xg_diff","xg_overperf"]
    if "sot" in long.columns: stat_cols+=["sot"]
    for w in [5,10]:
        for col in stat_cols:
            if col not in long.columns: continue
            shifted=long.groupby(grp,group_keys=False)[col].shift(1)
            features[f"{col}_r{w}"]=(
                shifted.groupby([long["Div"],long["season"],long["team"]])
                .transform(lambda x: x.rolling(w,min_periods=max(2,w//2)).mean()))
    shifted_t=long.groupby(grp,group_keys=False)["total"].shift(1)
    features["total_var_10"]=(shifted_t.groupby([long["Div"],long["season"],long["team"]])
        .transform(lambda x: x.rolling(10,min_periods=3).std()))
    feat_df=pd.DataFrame(features,index=long.index)
    feat_df["idx"]=long["idx"]; feat_df["is_home"]=long["is_home"]
    hf=feat_df[feat_df["is_home"]].drop(columns="is_home").set_index("idx")
    af=feat_df[~feat_df["is_home"]].drop(columns="is_home").set_index("idx")
    df=df.join(hf.add_prefix("h_"),how="left"); df=df.join(af.add_prefix("a_"),how="left")
    return df

def _ranking(df):
    df=df.copy(); h_ranks,a_ranks={},{}
    for (div,season),grp in df.groupby(["Div","season"],sort=False):
        grp=grp.sort_values("Date")
        stats={t:{"pts":0,"gd":0} for t in pd.concat([grp["HomeTeam"],grp["AwayTeam"]]).unique()}
        for idx,row in grp.iterrows():
            st=sorted(stats.items(),key=lambda x:(-x[1]["pts"],-x[1]["gd"]))
            rm={t:i+1 for i,(t,_) in enumerate(st)}
            h_ranks[idx]=rm.get(row["HomeTeam"]); a_ranks[idx]=rm.get(row["AwayTeam"])
            h,a=row["HomeTeam"],row["AwayTeam"]; hg,ag=row["FTHG"],row["FTAG"]
            if row["FTR"]=="H":stats[h]["pts"]+=3
            elif row["FTR"]=="A":stats[a]["pts"]+=3
            else:stats[h]["pts"]+=1;stats[a]["pts"]+=1
            stats[h]["gd"]+=hg-ag;stats[a]["gd"]+=ag-hg
    df["h_rank"]=pd.Series(h_ranks);df["a_rank"]=pd.Series(a_ranks)
    return df

def _market_features(df,market):
    df=df.copy()
    if market=="under":
        uc=[c for c in ["B365<2.5","P<2.5","Avg<2.5"] if c in df.columns]
        if len(uc)>=2:
            uo=df[uc];df["mkt_disp_u"]=uo.std(axis=1)/uo.mean(axis=1)
            df["mkt_range_u"]=(uo.max(axis=1)-uo.min(axis=1))/uo.mean(axis=1)
        else:df["mkt_disp_u"]=0;df["mkt_range_u"]=0
        if "P<2.5" in df.columns and "Avg<2.5" in df.columns:
            df["pinn_vs_avg_u"]=(1/df["P<2.5"].replace(0,np.nan))-(1/df["Avg<2.5"].replace(0,np.nan))
        else:df["pinn_vs_avg_u"]=0
        if "Max<2.5" in df.columns and "Avg<2.5" in df.columns:
            df["max_avg_u"]=df["Max<2.5"]/df["Avg<2.5"].replace(0,np.nan)
        else:df["max_avg_u"]=1.0
        if "Avg>2.5" in df.columns and "Avg<2.5" in df.columns:
            ro=1/df["Avg>2.5"].replace(0,np.nan);ru=1/df["Avg<2.5"].replace(0,np.nan)
            m=ro+ru;df["nv_over"]=ro/m;df["nv_under"]=ru/m;df["ou_margin"]=m-1
        else:df["nv_over"]=np.nan;df["nv_under"]=np.nan;df["ou_margin"]=np.nan
    elif market=="draw":
        dc=[c for c in ["B365D","BWD","IWD","PSD","WHD","VCD","AvgD"] if c in df.columns]
        if len(dc)>=2:
            do=df[dc];df["mkt_disp_d"]=do.std(axis=1)/do.mean(axis=1)
            df["mkt_range_d"]=(do.max(axis=1)-do.min(axis=1))/do.mean(axis=1)
        else:df["mkt_disp_d"]=0;df["mkt_range_d"]=0
        if "PSD" in df.columns and "AvgD" in df.columns:
            df["pinn_vs_avg_d"]=(1/df["PSD"].replace(0,np.nan))-(1/df["AvgD"].replace(0,np.nan))
        else:df["pinn_vs_avg_d"]=0
        if "MaxD" in df.columns and "AvgD" in df.columns:
            df["max_avg_d"]=df["MaxD"]/df["AvgD"].replace(0,np.nan)
        else:df["max_avg_d"]=1.0
        if all(c in df.columns for c in ["AvgH","AvgD","AvgA"]):
            rh=1/df["AvgH"].replace(0,np.nan);rd=1/df["AvgD"].replace(0,np.nan)
            ra=1/df["AvgA"].replace(0,np.nan);m=rh+rd+ra
            df["nv_home"]=rh/m;df["nv_draw"]=rd/m;df["nv_away"]=ra/m;df["match_margin"]=m-1
        else:df["nv_home"]=np.nan;df["nv_draw"]=np.nan;df["nv_away"]=np.nan;df["match_margin"]=np.nan
    return df

def build_features(df,market):
    has_xg="xG_home" in df.columns and df["xG_home"].notna().any()
    print(f"  [a] Rolling {'+ xG' if has_xg else ''}...")
    df=_team_rolling(df)
    print("  [b] Rankings...")
    df=_ranking(df)
    print("  [c] Market...")
    df=_market_features(df,market)
    df["total_goals"]=df["FTHG"]+df["FTAG"]
    df["under_25"]=(df["total_goals"]<2.5).astype(int)
    df["draw"]=(df["FTR"]=="D").astype(int)

    for w in [5,10]:
        s=lambda col,d: df.get(f"h_{col}_r{w}",pd.Series(d,index=df.index)).fillna(d)
        a=lambda col,d: df.get(f"a_{col}_r{w}",pd.Series(d,index=df.index)).fillna(d)
        df[f"comb_under_r{w}"]=(s("is_under",.5)+a("is_under",.5))/2
        df[f"comb_draw_r{w}"]=(s("is_draw",.26)+a("is_draw",.26))/2
        df[f"comb_goals_r{w}"]=(s("total",2.7)+a("total",2.7))/2
        df[f"gdiff_gap_r{w}"]=abs((s("scored",1.3)-s("conceded",1.3))-(a("scored",1.3)-a("conceded",1.3)))
        if has_xg:
            df[f"comb_xg_r{w}"]=(s("xg",1.3)+a("xg",1.3))/2
            df[f"comb_xga_r{w}"]=(s("xga",1.3)+a("xga",1.3))/2
            df[f"comb_xg_diff_r{w}"]=(s("xg_diff",0)+a("xg_diff",0))/2
            df[f"comb_xg_overperf_r{w}"]=(s("xg_overperf",0)+a("xg_overperf",0))/2
            df[f"xg_gap_r{w}"]=abs(s("xg",1.3)-a("xg",1.3))
    df["rank_diff"]=df["a_rank"].fillna(10)-df["h_rank"].fillna(10)
    if "h_sot_r5" in df.columns:
        df["comb_sot_r5"]=(df["h_sot_r5"].fillna(4)+df["a_sot_r5"].fillna(4))/2

    if market=="under":
        feature_cols=["h_scored_r5","h_conceded_r5","h_is_under_r5",
            "a_scored_r5","a_conceded_r5","a_is_under_r5","h_is_under_r10","a_is_under_r10",
            "comb_under_r5","comb_under_r10","comb_goals_r5","comb_goals_r10","gdiff_gap_r5",
            "h_total_var_10","a_total_var_10","rank_diff","comb_sot_r5",
            "mkt_disp_u","mkt_range_u","pinn_vs_avg_u","max_avg_u","ou_margin"]
        if has_xg:
            feature_cols+=["comb_xg_r5","comb_xg_r10","comb_xga_r5","comb_xga_r10",
                "comb_xg_diff_r5","comb_xg_overperf_r5","xg_gap_r5"]
        target="under_25";odds_col="Avg<2.5";pbc="nv_under"
    elif market=="draw":
        feature_cols=["h_is_draw_r5","a_is_draw_r5","h_win_r5","a_win_r5",
            "h_scored_r5","a_scored_r5","h_is_draw_r10","a_is_draw_r10",
            "comb_draw_r5","comb_draw_r10","gdiff_gap_r5","gdiff_gap_r10",
            "comb_goals_r5","rank_diff","comb_sot_r5",
            "mkt_disp_d","mkt_range_d","pinn_vs_avg_d","max_avg_d","match_margin"]
        if has_xg:
            feature_cols+=["comb_xg_r5","comb_xg_r10","comb_xga_r5",
                "comb_xg_diff_r5","comb_xg_diff_r10","comb_xg_overperf_r5",
                "xg_gap_r5","xg_gap_r10"]
        target="draw";odds_col="AvgD";pbc="nv_draw"

    feature_cols=[c for c in feature_cols if c in df.columns]
    df["_target"]=df[target]; df["_odds"]=df.get(odds_col,np.nan); df["_prob_bookie"]=df.get(pbc,np.nan)
    n0=len(df); df=df.dropna(subset=["_target","_odds","_prob_bookie"]).copy()

    if has_xg:
        nxg=df["xG_home"].notna().sum()
        print(f"  xG coverage: {nxg}/{len(df)} ({nxg/len(df):.0%})")
        df_xg=df[df["xG_home"].notna()].copy()
        if len(df_xg)>2000:
            print(f"  → Filtrage xG: {len(df)} → {len(df_xg)}")
            df=df_xg

    print(f"  Total: {len(feature_cols)} features | {len(df)} matchs")
    return df, feature_cols


def _make_xgb():
    return xgb.XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,
        subsample=0.8,colsample_bytree=0.7,min_child_weight=15,
        reg_alpha=0.5,reg_lambda=3.0,gamma=1.0,
        eval_metric="logloss",random_state=RANDOM_STATE,n_jobs=-1)

def train_fold(Xtr,ytr,Xv,yv,ens=True):
    sc=StandardScaler();Xtr_s=sc.fit_transform(np.nan_to_num(Xtr,nan=0.0))
    Xv_s=sc.transform(np.nan_to_num(Xv,nan=0.0))
    Xtr_c=np.nan_to_num(Xtr,nan=0.0);Xv_c=np.nan_to_num(Xv,nan=0.0)
    xm=_make_xgb();xm.fit(Xtr_c,ytr,eval_set=[(Xv_c,yv)],verbose=False)
    xp=xm.predict_proba(Xv_c)[:,1]; lr=None
    if ens:
        lr=LogisticRegression(C=0.1,max_iter=1000,random_state=RANDOM_STATE)
        lr.fit(Xtr_s,ytr); blend=0.6*xp+0.4*lr.predict_proba(Xv_s)[:,1]
    else: blend=xp
    cal=LogisticRegression(C=1e10,max_iter=1000)
    cal.fit(blend.reshape(-1,1),yv)
    return {"xgb":xm,"lr":lr,"sc":sc,"cal":cal,"ens":ens}

def predict_fold(m,X):
    Xc=np.nan_to_num(X,nan=0.0); xp=m["xgb"].predict_proba(Xc)[:,1]
    if m["ens"] and m["lr"]:
        blend=0.6*xp+0.4*m["lr"].predict_proba(m["sc"].transform(Xc))[:,1]
    else: blend=xp
    return m["cal"].predict_proba(blend.reshape(-1,1))[:,1]

def walk_forward(df,fcols,min_tr=3,ens=True):
    ns=df["season"].max(); preds=[]
    for ts in range(1,ns-min_tr):
        tr=df[df["season"]>=ts+2].dropna(subset=fcols+["_target"])
        vl=df[df["season"]==ts+1].dropna(subset=fcols+["_target"])
        te=df[df["season"]==ts].dropna(subset=fcols+["_target"])
        if len(tr)<300 or len(vl)<80 or len(te)<50: continue
        m=train_fold(tr[fcols].values,tr["_target"].values,vl[fcols].values,vl["_target"].values,ens)
        p=predict_fold(m,te[fcols].values)
        auc=roc_auc_score(te["_target"],p) if len(np.unique(te["_target"]))>1 else 0.5
        brier=brier_score_loss(te["_target"],p)
        t=te.copy();t["prob_model"]=p;preds.append(t)
        sy=te["season_year"].iloc[0]
        print(f"  Fold {len(preds):>2} | {sy} | tr={len(tr):>5} vl={len(vl):>4} te={len(te):>4} | AUC={auc:.3f} Brier={brier:.4f}")
        if len(preds)==1:
            imp=pd.Series(m["xgb"].feature_importances_,index=fcols).sort_values(ascending=False)
            print(f"  Top: {' | '.join(f'{f}={v:.3f}' for f,v in imp.head(8).items())}")
    if not preds: raise ValueError("No valid folds")
    dw=pd.concat(preds,ignore_index=True)
    oa=roc_auc_score(dw["_target"],dw["prob_model"])
    print(f"\n  Total: {len(dw)} matchs | AUC={oa:.4f}")
    for lo,hi in [(.2,.3),(.3,.4),(.4,.5),(.5,.6),(.6,.7)]:
        mk=(dw["prob_model"]>=lo)&(dw["prob_model"]<hi)
        if mk.sum()>50:
            print(f"    [{lo:.1f}-{hi:.1f}): n={mk.sum():>5} pred={dw.loc[mk,'prob_model'].mean():.3f} "
                  f"actual={dw.loc[mk,'_target'].mean():.3f} Δ={dw.loc[mk,'_target'].mean()-dw.loc[mk,'prob_model'].mean():+.3f}")
    return dw

def backtest(dw,edge_min=0.03):
    dw=dw.copy();dw["edge"]=dw["prob_model"]-dw["_prob_bookie"]
    vb=dw[dw["edge"]>=edge_min].copy().reset_index(drop=True)
    if len(vb)==0:print(f"\n  No VB (edge>={edge_min:.0%})");return None
    n=len(vb); vb["pfl"]=np.where(vb["_target"]==1,vb["_odds"]-1,-1.0)
    vb["cum_fl"]=vb["pfl"].cumsum(); roi=vb["pfl"].sum()/n*100
    wr=vb["_target"].mean();po=vb["prob_model"].mean();ao=vb["_odds"].mean()
    print(f"\n  {'='*55}\n  RESULTS | {n} bets\n  {'='*55}")
    print(f"  Win: {wr:.1%} (BE: {1/ao:.1%}) | Odds: {ao:.2f} | Edge: {vb['edge'].mean():.1%}")
    print(f"  ROI flat: {roi:+.2f}% | Calib: pred={po:.1%} actual={wr:.1%} ({wr-po:+.1%})")
    print(f"\n  Par ligue:")
    for d,g in vb.groupby("Div"):
        r=g["pfl"].sum()/len(g)*100; print(f"    {d:>4}: {len(g):>4} | win {g['_target'].mean():.0%} | ROI {r:+.1f}%")
    return vb

def significance(vb):
    p=vb["pfl"].values;n=len(p);t,pv=sp_stats.ttest_1samp(p,0)
    rng=np.random.default_rng(42);bs=[rng.choice(p,n,True).mean() for _ in range(10000)]
    cl,ch=np.percentile(bs,[2.5,97.5]);sig=pv<0.05 and cl>0
    print(f"\n  Sig: t={t:+.3f} p={pv:.4f} IC=[{cl:+.4f},{ch:+.4f}] {'✅ EDGE' if sig else '⚠ no edge'}")
    return sig

def optimize_edge(dw):
    rows=[]
    for t in [.01,.02,.03,.04,.05,.07,.10]:
        mk=(dw["prob_model"]-dw["_prob_bookie"])>=t; vb=dw[mk]
        if len(vb)<20:continue
        pf=np.where(vb["_target"]==1,vb["_odds"]-1,-1.0)
        rows.append({"edge":f"{t:.0%}","n":len(vb),"win":f"{vb['_target'].mean():.1%}",
                      "roi":f"{pf.sum()/len(vb)*100:+.1f}%","profit":f"{pf.sum():+.1f}u"})
    print(f"\n  Edge optim:");print(pd.DataFrame(rows).to_string(index=False))

def plot_diag(dw,path):
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    ax=axes[0,0];bins=pd.cut(dw["prob_model"],bins=15)
    cal=dw.groupby(bins,observed=False).agg(p=("prob_model","mean"),a=("_target","mean"),n=("_target","count")).dropna()
    ax.plot([0,1],[0,1],"k--",alpha=.5);ax.scatter(cal["p"],cal["a"],s=cal["n"]/cal["n"].max()*300,alpha=.7,color="steelblue")
    ax.set_title("Calibration")
    ax=axes[0,1]
    for t in [.02,.03,.05]:
        mk=(dw["prob_model"]-dw["_prob_bookie"])>=t;vb=dw[mk]
        if len(vb)>10:ax.plot(np.cumsum(np.where(vb["_target"]==1,vb["_odds"]-1,-1.0)),label=f"≥{t:.0%} (n={len(vb)})",alpha=.8)
    ax.axhline(0,color="grey",ls="--",lw=.8);ax.set_title("P&L");ax.legend(fontsize=8)
    ax=axes[1,0];tmp=dw.copy();tmp["edge"]=tmp["prob_model"]-tmp["_prob_bookie"]
    tmp["pfl"]=np.where(tmp["_target"]==1,tmp["_odds"]-1,-1.0)
    eb=pd.cut(tmp["edge"],bins=10);er=tmp.groupby(eb,observed=False).agg(em=("edge","mean"),roi=("pfl","mean"),n=("pfl","count")).dropna()
    c=["#4CAF50" if v>=0 else "#F44336" for v in er["roi"]]
    ax.bar(range(len(er)),er["roi"]*100,color=c);ax.set_xticks(range(len(er)))
    ax.set_xticklabels([f"{e:.2f}" for e in er["em"]],rotation=45,fontsize=7)
    ax.axhline(0,color="k",lw=.8);ax.set_title("ROI by edge")
    ax=axes[1,1];ax.hist(dw["prob_model"],bins=30,alpha=.5,label="Model",color="steelblue",density=True)
    ax.hist(dw["_prob_bookie"],bins=30,alpha=.5,label="Bookie",color="darkorange",density=True)
    ax.set_title("Distributions");ax.legend()
    plt.suptitle("Value Bet v4 (xG)",fontsize=14,fontweight="bold")
    plt.tight_layout();plt.savefig(path,dpi=150,bbox_inches="tight");plt.close()
    print(f"\n  📊 {path}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True);p.add_argument("--xg",default="understat_xg.csv")
    p.add_argument("--market",default="draw",choices=["under","draw"])
    p.add_argument("--leagues",default=None);p.add_argument("--edge",default=0.03,type=float)
    p.add_argument("--min-train",default=3,type=int);p.add_argument("--no-ensemble",action="store_true")
    a=p.parse_args()
    leagues=a.leagues.split(",") if a.leagues else DIV1
    print(f"\n{'='*60}\n  VALUE BET v4 (xG) | {a.market.upper()} | {','.join(leagues)}\n{'='*60}")
    print("\n[1/5] Loading matches...");df=load_matches(a.csv,leagues)
    print("\n[2/5] Merging xG...");df=load_and_merge_xg(df,a.xg)
    print("\n[3/5] Features...");df_feat,fcols=build_features(df,a.market)
    print(f"\n[4/5] Walk-forward (min_train={a.min_train})...")
    dw=walk_forward(df_feat,fcols,a.min_train,not a.no_ensemble)
    print(f"\n[5/5] Backtest (edge>={a.edge:.0%})...");vb=backtest(dw,a.edge)
    if vb is not None: significance(vb)
    optimize_edge(dw)
    out=os.path.dirname(os.path.abspath(a.csv))
    plot_diag(dw,os.path.join(out,f"v4_{a.market}_xg_diag.png"))
    print(f"\n✅ Done")

if __name__=="__main__": main()
