"""
04_backtest.py
--------------
Détecte les value bets et simule le ROI sur le jeu de test.

Un value bet existe quand :
    prob_modèle > prob_implicite_bookmaker + edge_min

Corrections v2 :
  - Ajout test de significativité statistique du ROI (bootstrap + t-test)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats


# ═══════════════════════════════════════════════════════════════
# 1. DÉTECTION DES VALUE BETS
# ═══════════════════════════════════════════════════════════════

def detect_value_bets(df: pd.DataFrame,
                      edge_min: float = 0.03,
                      prob_col: str = 'prob_model',
                      bookie_col: str = 'prob_bookie') -> pd.DataFrame:
    df = df.copy()
    df['edge'] = df[prob_col] - df[bookie_col]
    value_bets = df[df['edge'] >= edge_min].copy()
    print(f'  Value bets edge≥{edge_min:.0%} : {len(value_bets)}/{len(df)} ({len(value_bets)/len(df):.1%})')
    return value_bets


# ═══════════════════════════════════════════════════════════════
# 2. SIMULATION ROI
# ═══════════════════════════════════════════════════════════════

def simulate_roi(value_bets: pd.DataFrame,
                 odds_col: str = 'Avg>2.5',
                 target_col: str = 'over_25',
                 stake: float = 1.0) -> pd.DataFrame:
    vb = value_bets.copy().reset_index(drop=True)
    vb['profit'] = np.where(
        vb[target_col] == 1,
        (vb[odds_col] - 1) * stake,
        -stake
    )
    vb['cumulative_profit'] = vb['profit'].cumsum()
    total_staked = len(vb) * stake
    vb['roi_pct'] = vb['cumulative_profit'] / total_staked * 100
    return vb


def print_summary(vb: pd.DataFrame,
                  odds_col: str = 'Avg>2.5',
                  target_col: str = 'over_25'):
    wins     = vb[target_col].sum()
    total    = len(vb)
    profit   = vb['profit'].sum()
    roi      = profit / total * 100
    avg_odds = vb[odds_col].mean()
    edge_m   = vb['edge'].mean()

    print(f'\n{"═"*50}')
    print(f'  BACKTEST VALUE BET | {total} paris | {wins/total:.1%} win')
    print(f'  Cote moy {avg_odds:.2f} | Edge moy {edge_m:.1%}')
    print(f'  Profit : {profit:+.1f}u | ROI : {roi:+.1f}%')
    print(f'{"═"*50}')
    print(f'  Par ligue :')
    for div, g in vb.groupby('Div'):
        r = g['profit'].sum() / len(g) * 100
        w = g[target_col].mean()
        print(f'    {div:>4} : {len(g):>4} paris | win {w:.0%} | ROI {r:+.1f}%')


# ═══════════════════════════════════════════════════════════════
# 3. SIGNIFICATIVITÉ STATISTIQUE (v2)
# ═══════════════════════════════════════════════════════════════

def test_significance(vb: pd.DataFrame,
                      n_bootstrap: int = 10_000,
                      random_state: int = 42) -> dict:
    """
    Teste si le ROI est significativement différent de 0.

    Deux approches :
      1. t-test sur les profits unitaires (H0 : mean = 0)
      2. Bootstrap IC 95% du ROI moyen

    Returns:
        dict avec t-stat, p-value, IC bootstrap 95%, ROI moyen.
    """
    profits = vb['profit'].values
    n       = len(profits)
    roi_mean = profits.mean()

    # t-test
    t_stat, p_value = scipy_stats.ttest_1samp(profits, popmean=0)

    # Bootstrap
    rng = np.random.default_rng(random_state)
    boot_means = np.array([
        rng.choice(profits, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    print(f'\n  Significativité statistique ({n} paris) :')
    print(f'    ROI moyen / pari : {roi_mean:+.4f}u')
    print(f'    t-stat           : {t_stat:+.3f}')
    print(f'    p-value          : {p_value:.4f} {"✅ sig. 5%" if p_value < 0.05 else "⚠ non sig."}')
    print(f'    IC 95% bootstrap : [{ci_low:+.4f}, {ci_high:+.4f}]')
    print(f'    {"✅ IC entièrement positif" if ci_low > 0 else "⚠ IC contient 0"}')

    return {
        'n': n,
        'roi_mean': roi_mean,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'significant': p_value < 0.05 and ci_low > 0,
    }


# ═══════════════════════════════════════════════════════════════
# 4. VISUALISATIONS
# ═══════════════════════════════════════════════════════════════

def plot_results(vb: pd.DataFrame, save_path: str = 'backtest.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    market_label = 'Under 2.5' if 'under' in str(save_path) else 'Over 2.5'
    fig.suptitle(f'Backtest Value Bet — {market_label}', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(vb['cumulative_profit'].values, color='#2196F3', linewidth=2)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.fill_between(range(len(vb)), vb['cumulative_profit'].values, alpha=0.1, color='#2196F3')
    ax.set_title('Profit cumulé (unités)')
    ax.set_xlabel('Pari n°')
    ax.set_ylabel('Unités')

    ax = axes[0, 1]
    ax.hist(vb['edge'], bins=25, color='#4CAF50', edgecolor='white', linewidth=0.5)
    ax.set_title('Distribution des edges')
    ax.set_xlabel('Edge (prob_modèle − prob_bookie)')
    ax.set_ylabel('Nb matchs')

    ax = axes[1, 0]
    target_col_plot = [c for c in vb.columns if c in ['over_25', 'under_25']][0]
    colors = vb[target_col_plot].map({1: '#4CAF50', 0: '#F44336'})
    ax.scatter(vb['prob_bookie'], vb['prob_model'], c=colors, alpha=0.4, s=20)
    ax.plot([0.3, 0.9], [0.3, 0.9], 'k--', linewidth=0.8, label='prob égales')
    ax.set_xlabel('Prob bookie')
    ax.set_ylabel('Prob modèle')
    ax.set_title('Modèle vs Bookie\n(vert=win réel, rouge=lose)')
    ax.legend(fontsize=8)

    if 'Div' in vb.columns:
        ax = axes[1, 1]
        roi_div = vb.groupby('Div').apply(
            lambda g: g['profit'].sum() / len(g) * 100
        ).sort_values()
        colors_bar = ['#4CAF50' if v >= 0 else '#F44336' for v in roi_div]
        roi_div.plot(kind='barh', ax=ax, color=colors_bar)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title('ROI par championnat (%)')
        ax.set_xlabel('ROI (%)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\n📊 Graphiques sauvegardés → {save_path}')
    plt.close()


def export_value_bets(vb: pd.DataFrame,
                      save_path: str = 'value_bets.csv',
                      odds_col: str = 'Avg>2.5',
                      target_col: str = 'over_25') -> pd.DataFrame:
    cols_to_keep = []
    for c in ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'season_year']:
        if c in vb.columns: cols_to_keep.append(c)
    for c in ['FTHG', 'FTAG', 'total_goals', target_col]:
        if c in vb.columns: cols_to_keep.append(c)
    for c in [odds_col, 'prob_bookie', 'prob_model', 'edge']:
        if c in vb.columns: cols_to_keep.append(c)
    for c in ['profit', 'cumulative_profit', 'roi_pct', 'match_importance']:
        if c in vb.columns: cols_to_keep.append(c)

    df_export = vb[cols_to_keep].copy()
    rename_map = {
        target_col:          'result_reel',
        odds_col:            'cote',
        'prob_bookie':       'prob_bookie_%',
        'prob_model':        'prob_modele_%',
        'edge':              'edge_%',
        'profit':            'profit_u',
        'cumulative_profit': 'profit_cumul_u',
        'roi_pct':           'ROI_%',
        'total_goals':       'buts_total',
        'FTHG':              'buts_domicile',
        'FTAG':              'buts_exterieur',
        'season_year':       'saison',
    }
    df_export = df_export.rename(columns={k: v for k, v in rename_map.items() if k in df_export.columns})

    for c in ['prob_bookie_%', 'prob_modele_%', 'edge_%']:
        if c in df_export.columns:
            df_export[c] = (df_export[c] * 100).round(1)
    for c in ['profit_u', 'profit_cumul_u', 'ROI_%']:
        if c in df_export.columns:
            df_export[c] = df_export[c].round(2)
    if 'result_reel' in df_export.columns:
        df_export['resultat'] = df_export['result_reel'].map({1: 'WIN ✅', 0: 'LOSE ❌'})

    df_export.to_csv(save_path, index=False)
    print(f'💾 Value bets exportés → {save_path} ({len(df_export)} lignes)')
    return df_export


# ═══════════════════════════════════════════════════════════════
# 5. OPTIMISATION DU SEUIL D'EDGE
# ═══════════════════════════════════════════════════════════════

def optimize_edge_threshold(df: pd.DataFrame,
                             edge_range=None,
                             odds_col: str = 'Avg>2.5',
                             target_col: str = 'over_25'):
    if edge_range is None:
        edge_range = [0.03, 0.05, 0.07, 0.09, 0.10, 0.12, 0.14]
    rows = []
    for edge in edge_range:
        vb = detect_value_bets(df, edge_min=edge)
        if len(vb) < 10:
            continue
        sim = simulate_roi(vb, odds_col=odds_col, target_col=target_col)
        rows.append({
            'edge':   f'{edge:.0%}',
            'paris':  len(sim),
            'ROI':    f"{sim['profit'].sum()/len(sim)*100:+.1f}%",
            'profit': f"{sim['profit'].sum():+.1f}u",
        })
    print('\n  Optimisation edge :')
    print(pd.DataFrame(rows).to_string(index=False))
