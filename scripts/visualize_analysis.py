"""
6-panel visualization of trajectory-only model analysis findings.
Output: results/images/model_analysis.png

Usage: python scripts/visualize_analysis.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

ANALYSIS_DIR = Path('/home/satria/Project/ATLAS/results/analysis')
OUTPUT_DIR   = Path('/home/satria/Project/ATLAS/results/images')

sns.set_theme(style='whitegrid', font_scale=1.0)
plt.rcParams['font.family'] = 'sans-serif'

# Consistent colour map for outcomes
OUTCOME_COLORS = {
    'TP': '#2E8B57',   # green
    'TN': '#4A90D9',   # blue
    'FP': '#E05C5C',   # red
    'FN': '#F5A623',   # orange
}
FAIL_COLORS = {
    'slow_creep_fn':         '#D94F4F',
    'short_traj_fn':         '#F5A623',
    'ambiguous_fn':          '#888888',
    'brief_stop_fn':         '#9B59B6',
    'close_ped_fp':          '#E67E22',
    'ambiguous_fp':          '#95A5A6',
    'high_approach_stops_fp':'#3498DB',
}


def load_data():
    with open(ANALYSIS_DIR / 'package.json') as f:
        pkg = json.load(f)
    with open(ANALYSIS_DIR / 'all_events.json') as f:
        all_events = json.load(f)
    df = pd.DataFrame(all_events)
    return pkg, df


# ── panel helpers ─────────────────────────────────────────────────────────────

def panel_ap(ax, pkg):
    """Panel A: AP bar chart."""
    bl   = pkg['baseline_ap']
    names  = ['APv\n(violation)', 'APn\n(compliance)', 'mAP']
    values = [bl['APv'], bl['APn'], bl['mAP']]
    colors = ['#E05C5C', '#2E8B57', '#4A90D9']

    bars = ax.barh(names, values, color=colors, edgecolor='white',
                   linewidth=0.8, height=0.55)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

    ax.axvline(0.5, color='#999999', linestyle='--', linewidth=1.2,
               label='Random baseline (0.5)')
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Average Precision', fontsize=9)
    ax.set_title('Model Performance', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.invert_yaxis()


def panel_confusion(ax, pkg):
    """Panel B: 2×2 confusion matrix."""
    conf = pkg['confusion']
    n_viol = conf['TP'] + conf['FN']
    n_comp = conf['FP'] + conf['TN']

    mat = np.array([
        [conf['TP'] / n_viol, conf['FN'] / n_viol],
        [conf['FP'] / n_comp, conf['TN'] / n_comp],
    ])
    raw = np.array([
        [conf['TP'], conf['FN']],
        [conf['FP'], conf['TN']],
    ])

    im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['GT: Violation', 'GT: Compliance'], fontsize=9)
    ax.set_yticklabels(['Pred: Violation', 'Pred: Compliance'], fontsize=9)

    for i in range(2):
        for j in range(2):
            pct = mat[i, j] * 100
            color = 'white' if mat[i, j] < 0.35 or mat[i, j] > 0.65 else 'black'
            ax.text(j, i, f'{raw[i, j]:,}\n({pct:.0f}%)',
                    ha='center', va='center', fontsize=10,
                    fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label='Row-normalised rate')
    ax.set_title(
        f'Confusion Matrix\nSensitivity={pkg["sensitivity"]:.2f}  '
        f'Specificity={pkg["specificity"]:.2f}',
        fontsize=11, fontweight='bold',
    )


def panel_failures(ax, pkg):
    """Panel C: failure mode breakdown."""
    fn = pkg['fn_categories']
    fp = pkg['fp_categories']

    labels, counts, colors, group = [], [], [], []
    for k, v in sorted(fn.items(), key=lambda x: -x[1]):
        labels.append(k.replace('_fn', '').replace('_', '\n'))
        counts.append(v); colors.append(FAIL_COLORS.get(k, '#888'))
        group.append('FN')
    for k, v in sorted(fp.items(), key=lambda x: -x[1]):
        labels.append(k.replace('_fp', '').replace('_', '\n'))
        counts.append(v); colors.append(FAIL_COLORS.get(k, '#888'))
        group.append('FP')

    y = np.arange(len(labels))
    bars = ax.barh(y, counts, color=colors, edgecolor='white',
                   linewidth=0.5, height=0.65)
    for bar, val in zip(bars, counts):
        ax.text(val + 2, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', fontsize=8)

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Event count', fontsize=9)
    ax.set_title('Failure Mode Breakdown', fontsize=11, fontweight='bold')

    # Divider line between FN and FP sections
    n_fn = len(fn)
    ax.axhline(n_fn - 0.5, color='#999', linestyle=':', linewidth=1)
    ax.text(ax.get_xlim()[1] * 0.98, n_fn * 0.35,
            'FN\n(missed violations)', ha='right', va='center',
            fontsize=8, color='#555', style='italic')
    ax.text(ax.get_xlim()[1] * 0.98, n_fn + len(fp) * 0.6,
            'FP\n(false alarms)', ha='right', va='center',
            fontsize=8, color='#555', style='italic')


def panel_ablation(ax, pkg):
    """Panel D: feature ablation waterfall."""
    bl  = pkg['baseline_ap']['APv']
    abl = pkg['ablations']

    names  = ['Baseline\n(full)', 'Speed\nzeroed', 'Ped\nzeroed', 'Position\nzeroed']
    values = [bl,
              abl['speed_zeroed']['APv'],
              abl['ped_zeroed']['APv'],
              abl['position_zeroed']['APv']]
    colors = ['#4A90D9', '#F5A623', '#E05C5C', '#D94F4F']

    bars = ax.bar(names, values, color=colors, edgecolor='white',
                  linewidth=0.8, width=0.55)
    for bar, val, base in zip(bars, values, [bl]*4):
        delta = base - val
        label = f'{val:.3f}' if delta == 0 else f'{val:.3f}\n(−{delta:.3f})'
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                label, ha='center', va='bottom', fontsize=8.5,
                fontweight='bold')

    ax.set_ylim(0, 0.75)
    ax.set_ylabel('APv', fontsize=9)
    ax.set_title('Feature Ablation (APv)', fontsize=11, fontweight='bold')
    ax.axhline(bl, color='#4A90D9', linestyle='--', linewidth=1, alpha=0.5)



def panel_score_vs_speed(ax, df):
    """Panel E: prediction score vs mean speed, coloured by outcome."""
    # Clip speed at 99th pct for readability
    clip = df['mean_speed'].quantile(0.99)
    sub  = df[df['mean_speed'].notna() & (df['mean_speed'] <= clip)].copy()

    for outcome in ['TN', 'TP', 'FN', 'FP']:
        s = sub[sub['outcome'] == outcome]
        ax.scatter(s['mean_speed'], s['score'],
                   c=OUTCOME_COLORS[outcome], label=outcome,
                   s=6, alpha=0.35, linewidths=0)

    # Decision boundary
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1,
               label='Decision boundary (0.5)')

    # Ambiguity band shading
    ax.axvspan(1.0, 2.0, alpha=0.07, color='gray')
    ax.text(1.5, 0.97, '1–2 m/s\nambiguity band',
            ha='center', va='top', fontsize=7.5, color='#555', style='italic')

    ax.set_xlabel('Mean approach speed (m/s)', fontsize=9)
    ax.set_ylabel('P(violation) score', fontsize=9)
    ax.set_title('Score vs Speed by Outcome', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, markerscale=3, loc='upper right',
              framealpha=0.9)
    ax.set_ylim(-0.05, 1.05)


def panel_feature_dists(ax, df):
    """Panel F: box plots of mean_speed by outcome."""
    clip = df['mean_speed'].quantile(0.99)
    sub  = df[df['mean_speed'].notna() & (df['mean_speed'] <= clip)].copy()

    order  = ['TP', 'TN', 'FP', 'FN']
    labels = ['TP\n(correct viol.)', 'TN\n(correct comp.)',
              'FP\n(false alarm)', 'FN\n(missed viol.)']
    palette = {o: OUTCOME_COLORS[o] for o in order}

    sns.boxplot(data=sub, x='outcome', y='mean_speed',
                order=order, hue='outcome', palette=palette, legend=False,
                width=0.5, fliersize=2, ax=ax,
                linewidth=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlabel('')
    ax.set_ylabel('Mean approach speed (m/s)', fontsize=9)
    ax.set_title('Speed Distribution by Outcome', fontsize=11, fontweight='bold')

    # Medians annotation
    for i, o in enumerate(order):
        med = sub[sub['outcome'] == o]['mean_speed'].median()
        ax.text(i, med + 0.08, f'{med:.2f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='#333')


# ── main ──────────────────────────────────────────────────────────────────────

PANELS = [
    ('ap',            panel_ap,            (6, 4),  True,  False),
    ('confusion',     panel_confusion,     (5, 4),  True,  False),
    ('failures',      panel_failures,      (6, 5),  True,  False),
    ('ablation',      panel_ablation,      (6, 4),  True,  False),
    ('score_speed',   panel_score_vs_speed,(7, 5),  False, True),
    ('speed_dist',    panel_feature_dists, (7, 5),  False, True),
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pkg, df = load_data()

    for name, fn, figsize, uses_pkg, uses_df in PANELS:
        fig, ax = plt.subplots(figsize=figsize)
        if uses_pkg and uses_df:
            fn(ax, pkg, df)
        elif uses_pkg:
            fn(ax, pkg)
        else:
            fn(ax, df)
        plt.tight_layout()
        out = OUTPUT_DIR / f'analysis_{name}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved → {out}")


if __name__ == '__main__':
    main()
