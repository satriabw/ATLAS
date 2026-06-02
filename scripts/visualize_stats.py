"""
Simple dataset statistics dashboard — white background, seaborn style.
Output: results/images/dataset_stats.png

Usage: python scripts/visualize_stats.py
"""
import pickle, re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

DATA_ROOT   = Path('/home/satria/Project/ATLAS')
PARQUET_DIR = DATA_ROOT / 'data/processed/interactions'
STATS_JSON  = DATA_ROOT / 'results/summary/stats.json'
OUTPUT_DIR  = DATA_ROOT / 'results/images'

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.05)

VIO_COLOR  = '#E05C5C'
COMP_COLOR = '#4A9E6F'


def to_arr(val):
    arr = np.asarray(val)
    if arr.dtype == object:
        return np.stack(arr.tolist()).astype(np.float32)
    return arr.astype(np.float32)


def load_per_event_stats():
    events = []
    for split in ('train', 'test'):
        pkl = DATA_ROOT / 'data/raw/labels' / f'{split}_labels.pkl'
        if not pkl.exists():
            continue
        with open(pkl, 'rb') as f:
            strings, _ = pickle.load(f)
        for s in strings:
            m = re.match(r'V(\d+)I(\d+)S(\d)D\d+R\d+A(\d)', s)
            if not m:
                continue
            events.append((f"video_{int(m.group(1)):03d}", int(m.group(2)),
                           'BOT' if m.group(3) == '1' else 'TOP', int(m.group(4))))

    parquets, rows = {}, []
    for vid, tid, roi, ann in events:
        if vid not in parquets:
            p = PARQUET_DIR / f'{vid}_interactions.parquet'
            parquets[vid] = pd.read_parquet(p) if p.exists() else None
        df = parquets[vid]
        if df is None:
            continue
        g = df[(df['v_track_id'] == tid) & (df['roi'] == roi)]
        if g.empty:
            continue
        # Deduplicate frames so repeated vehicle rows don't inflate arc length
        f_parts  = [np.asarray(r['frames'], dtype=np.int64).ravel() for _, r in g.iterrows()]
        sp_parts = [np.asarray(r['v_speed'], dtype=np.float32).ravel() for _, r in g.iterrows()]
        lc_parts = [to_arr(r['v_loc_planar']) for _, r in g.iterrows()]
        all_f    = np.concatenate(f_parts)
        all_sp_  = np.concatenate(sp_parts)
        all_lc   = np.vstack(lc_parts)
        order    = np.argsort(all_f)
        _, keep  = np.unique(all_f[order], return_index=True)
        idx      = order[keep]
        all_sp   = all_sp_[idx]
        all_loc  = all_lc[idx]
        arc      = float(np.sqrt(np.sum(np.diff(all_loc, axis=0) ** 2, axis=1)).sum())
        rows.append({
            'class':        'Violation' if ann == 0 else 'Compliance',
            'mean_speed':   float(np.mean(all_sp)),
            'arc_length_m': arc,
        })
    return pd.DataFrame(rows)


def panel_class_balance(ax, df):
    counts = df['class'].value_counts()
    _, _, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        colors=[VIO_COLOR, COMP_COLOR],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 11},
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight('bold')
    ax.set_title(f'Class Balance  (n={len(df):,})', fontsize=12, fontweight='bold')


def panel_speed_dist(ax, df):
    clip = df['mean_speed'].quantile(0.99)
    for cls, color in [('Violation', VIO_COLOR), ('Compliance', COMP_COLOR)]:
        vals = df[df['class'] == cls]['mean_speed']
        vals = vals[vals <= clip]
        sns.kdeplot(vals, ax=ax, color=color, fill=True, alpha=0.28,
                    linewidth=2, label=cls)
        med = vals.median()
        ax.axvline(med, color=color, linestyle='--', linewidth=1.2, alpha=0.8)
        ax.text(med + 0.06, ax.get_ylim()[1] * 0.85, f'{med:.1f}',
                color=color, fontsize=8)
    ax.set_xlabel('Mean approach speed (m/s)')
    ax.set_ylabel('Density')
    ax.set_title('Speed Distribution by Class', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)


def panel_arc_length(ax, df):
    clip_arc = df['arc_length_m'].quantile(0.98)
    for cls, color in [('Violation', VIO_COLOR), ('Compliance', COMP_COLOR)]:
        vals = df[df['class'] == cls]['arc_length_m']
        vals = vals[(vals >= 0) & (vals <= clip_arc)]
        sns.kdeplot(vals, ax=ax, color=color, fill=True, alpha=0.28,
                    linewidth=2, label=cls)
        med = vals.median()
        ax.axvline(med, color=color, linestyle='--', linewidth=1.2, alpha=0.8)
        ax.text(med + 2, ax.get_ylim()[1] * 0.82, f'{med:.0f}m',
                color=color, fontsize=8)
    ax.set_xlabel('Trajectory arc length (m)')
    ax.set_ylabel('Density')
    ax.set_xlim(left=0)
    ax.set_title('Trajectory Length by Class', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)


def panel_violation_rate(ax, bins_data):
    labels = list(bins_data.keys())
    rates  = [bins_data[k]['violation_rate'] * 100 for k in labels]
    totals = [bins_data[k]['total'] for k in labels]
    colors = [VIO_COLOR if r == max(rates) else '#888888' for r in rates]

    bars = ax.barh(labels, rates, color=colors, edgecolor='white', linewidth=0.5, height=0.6)
    for bar, rate, total in zip(bars, rates, totals):
        ax.text(rate + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{rate:.1f}%  (n={total:,})',
                va='center', fontsize=8, color='#444444')

    ax.set_xlabel('Violation Rate (%)')
    ax.set_title('Violation Rate by Speed Bin', fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(rates) * 1.45)
    ax.invert_yaxis()
    ax.text(0.97, 0.05,
            '"Slow creep" peaks at 1-2 m/s',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color='#333333', style='italic')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading per-event statistics...")
    df = load_per_event_stats()
    print(f"Loaded {len(df)} events")

    with open(STATS_JSON) as f:
        stats = json.load(f)
    bins_data = stats['violation_rate_by_speed_bin']

    panels = [
        ('class_balance',   panel_class_balance,  (5, 5),  lambda ax: panel_class_balance(ax, df)),
        ('speed_dist',      panel_speed_dist,      (7, 5),  lambda ax: panel_speed_dist(ax, df)),
        ('arc_length',      panel_arc_length,      (7, 5),  lambda ax: panel_arc_length(ax, df)),
        ('violation_rate',  panel_violation_rate,  (7, 5),  lambda ax: panel_violation_rate(ax, bins_data)),
    ]

    for name, _, figsize, draw in panels:
        fig, ax = plt.subplots(figsize=figsize)
        draw(ax)
        plt.tight_layout()
        out = OUTPUT_DIR / f'stats_{name}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {out}")


if __name__ == '__main__':
    main()
