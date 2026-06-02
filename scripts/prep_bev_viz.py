"""
BEV trajectory visualization: violation vs compliance side-by-side.
Crosswalk ROI is projected from pixel coords via the camera model.
Only pedestrians within CLOSE_PED_DMIN metres of the vehicle are shown.

Usage: python scripts/prep_bev_viz.py
"""
import sys, pickle, re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path

sys.path.insert(0, '/home/satria/Project/crosswalk-ws/modules/calibration')
from camera_model import CameraModel

DATA_ROOT    = Path('/home/satria/Project/ATLAS')
PARQUET_DIR  = DATA_ROOT / 'data/processed/interactions'
LABELS_PATH  = DATA_ROOT / 'data/raw/labels/train_labels.pkl'
CAM_YML      = Path('/home/satria/Project/crosswalk-ws/data/calibration/camera_model.yml')
OUTPUT_DIR   = DATA_ROOT / 'results/images'
JSON_PATH    = DATA_ROOT / 'results/summary/trajectory_examples.json'

MAX_VID_NUM  = 50        # odd-numbered videos up to this
DMIN_THRESH  = 5.0       # event must have at least one ped within this distance
CLOSE_PED    = 6.0       # only show pedestrians closer than this in BEV

# Pixel-space ROI polygons (from Track2Data config)
POLY_TOP_PX = np.array([[0.0, 120.0], [1200.0, 120.0], [1200.0, 480.0], [0.0, 480.0]])
POLY_BOT_PX = np.array([[120.0, 480.0], [1100.0, 240.0], [1100.0, 600.0], [240.0, 1000.0]])


def build_camera_model():
    return CameraModel.load_from_yml(str(CAM_YML))


def project_roi(cam, poly_px):
    """Project a pixel-space polygon to world (planar) coordinates."""
    world = cam.project_to_ground(poly_px)
    if world.shape[0] == 2:
        world = world.T
    return world  # (N, 2)


def parse_label(s):
    m = re.match(r'V(\d+)I(\d+)S(\d)D\d+R\d+A(\d)', s)
    if not m:
        return None
    n = int(m.group(1))
    return f"video_{n:03d}", int(m.group(2)), 'BOT' if m.group(3) == '1' else 'TOP', int(m.group(4)), n


def to_arr(val):
    arr = np.asarray(val)
    if arr.dtype == object:
        return np.stack(arr.tolist()).astype(np.float32)
    return arr.astype(np.float32)


def load_event(df, v_track_id, roi):
    """
    Returns (v_loc, v_sp, ped_data) where ped_data = {pid: (locs, dmin)}.
    """
    group = df[(df['v_track_id'] == v_track_id) & (df['roi'] == roi)]
    if group.empty:
        return None

    v_locs, v_speeds, all_frames = [], [], []
    ped_data = {}

    for _, row in group.iterrows():
        f    = np.asarray(row['frames'], dtype=np.int64).ravel()
        sp   = np.asarray(row['v_speed'], dtype=np.float32).ravel()
        loc  = to_arr(row['v_loc_planar'])
        ploc = to_arr(row['p_loc_planar'])
        pid  = row['p_track_id']
        dmin = float(np.asarray(row['d_min']).ravel().min())

        all_frames.append(f)
        v_locs.append(loc)
        v_speeds.append(sp)

        if pid not in ped_data:
            ped_data[pid] = ([], dmin)
        ped_data[pid][0].append(ploc)

    all_f   = np.concatenate(all_frames)
    all_loc = np.vstack(v_locs)
    all_sp  = np.concatenate(v_speeds)
    order   = np.argsort(all_f)
    _, keep = np.unique(all_f[order], return_index=True)
    idx     = order[keep]

    v_loc = all_loc[idx]
    v_sp  = all_sp[idx]
    ped_final = {pid: (np.vstack(locs), dmin) for pid, (locs, dmin) in ped_data.items()}
    return v_loc, v_sp, ped_final


def event_min_dmin(df, v_track_id, roi):
    g = df[(df['v_track_id'] == v_track_id) & (df['roi'] == roi)]
    if g.empty:
        return float('inf')
    return min(float(np.asarray(r['d_min']).ravel().min()) for _, r in g.iterrows())


def score_violation(v_sp):
    return float(np.mean(v_sp))


def score_compliance(v_sp):
    n = len(v_sp)
    first = v_sp[:max(n // 3, 1)]
    last  = v_sp[max(n * 2 // 3, 1):]
    return float(np.mean(first) - np.min(last))


def collect_scored(labels, score_fn, parquets):
    results = []
    for vid, tid, roi, ann, _ in labels:
        if vid not in parquets:
            continue
        if event_min_dmin(parquets[vid], tid, roi) > DMIN_THRESH:
            continue
        data = load_event(parquets[vid], tid, roi)
        if data is None or len(data[1]) < 10:
            continue
        v_loc, v_sp, ped_data = data
        dm = min(d for _, d in ped_data.values())
        results.append((score_fn(v_sp), vid, tid, roi, ann, v_loc, v_sp, ped_data, dm))
    return sorted(results, reverse=True)


def plot_event(ax, label_type, result, roi_world):
    _, vid, tid, roi, _, v_loc, v_sp, ped_data, dm = result
    accent = '#D94F4F' if label_type == 'Violation' else '#2E8B57'
    ax.set_facecolor('#f8f9fa')

    # Draw crosswalk ROI polygon
    if roi_world is not None:
        poly_patch = MplPolygon(roi_world, closed=True,
                                facecolor='#FFD700', alpha=0.18,
                                edgecolor='#B8860B', linewidth=1.5, linestyle='--', zorder=1)
        ax.add_patch(poly_patch)
        # Label
        cx, cy = roi_world[:, 0].mean(), roi_world[:, 1].mean()
        ax.text(cx, cy, 'Crosswalk', ha='center', va='center',
                fontsize=7, color='#8B6914', alpha=0.8, style='italic')

    # Vehicle trajectory colored by speed
    sp_min, sp_max = v_sp.min(), v_sp.max()
    sp_range = sp_max - sp_min + 1e-6
    for i in range(len(v_loc) - 1):
        t = (v_sp[i] - sp_min) / sp_range
        ax.plot(v_loc[i:i+2, 0], v_loc[i:i+2, 1],
                color=plt.cm.RdYlGn_r(t), linewidth=2.5,
                solid_capstyle='round', zorder=3)

    # Start / end markers
    ax.scatter(*v_loc[0],  color='royalblue', s=80, zorder=6, marker='o',
               edgecolors='white', linewidths=1, label='Start')
    ax.scatter(*v_loc[-1], color='white',     s=80, zorder=6, marker='s',
               edgecolors=accent, linewidths=1.5, label='End')

    # Direction arrow at midpoint
    mid = len(v_loc) // 2
    if mid + 1 < len(v_loc):
        ax.annotate('', xy=v_loc[mid + 1], xytext=v_loc[mid],
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5), zorder=5)

    # Close pedestrians only (d_min < CLOSE_PED)
    ped_palette = ['#E05C5C', '#E07B2A', '#8B5CF6', '#0EA5E9', '#10B981']
    close_peds = [(pid, locs, dmin) for pid, (locs, dmin) in ped_data.items() if dmin <= CLOSE_PED]
    close_peds.sort(key=lambda x: x[2])
    for i, (pid, ploc, pdmin) in enumerate(close_peds[:4]):
        c = ped_palette[i % len(ped_palette)]
        ax.scatter(ploc[:, 0], ploc[:, 1], c=c, s=18, alpha=0.70,
                   zorder=4, marker='^', label=f'Ped {pid} ({pdmin:.1f}m)')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(sp_min, sp_max))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label('Speed (m/s)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Speed profile inset
    ins = ax.inset_axes([0.03, 0.67, 0.36, 0.28])
    ins.plot(v_sp, color=accent, linewidth=1.5)
    ins.fill_between(range(len(v_sp)), v_sp, alpha=0.2, color=accent)
    ins.set_facecolor('white')
    ins.set_title('speed (m/s)', fontsize=6, pad=2)
    ins.tick_params(labelsize=5)
    ins.set_xlabel('frame', fontsize=5, labelpad=1)
    sns_grid_color = '#e0e0e0'
    for sp in ins.spines.values():
        sp.set_color(sns_grid_color)

    ax.set_title(f'{label_type.upper()}  —  {vid} | Track {tid} | {roi}\nd_min = {dm:.1f} m',
                 color=accent, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('X (m, world space)', fontsize=9)
    ax.set_ylabel('Y (m, world space)', fontsize=9)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle=':', alpha=0.5, color='#cccccc')
    ax.legend(fontsize=7, loc='lower right', framealpha=0.8)


def main():
    import seaborn as sns
    sns.set_theme(style='whitegrid')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    cam = build_camera_model()
    roi_top_world = project_roi(cam, POLY_TOP_PX)
    roi_bot_world = project_roi(cam, POLY_BOT_PX)
    print(f"TOP world: x=[{roi_top_world[:,0].min():.1f}, {roi_top_world[:,0].max():.1f}]  "
          f"y=[{roi_top_world[:,1].min():.1f}, {roi_top_world[:,1].max():.1f}]")
    print(f"BOT world: x=[{roi_bot_world[:,0].min():.1f}, {roi_bot_world[:,0].max():.1f}]  "
          f"y=[{roi_bot_world[:,1].min():.1f}, {roi_bot_world[:,1].max():.1f}]")

    with open(LABELS_PATH, 'rb') as f:
        label_strings, _ = pickle.load(f)

    parsed = [parse_label(s) for s in label_strings]
    parsed = [l for l in parsed if l and l[4] % 2 == 1 and l[4] <= MAX_VID_NUM]

    violations  = [l for l in parsed if l[3] == 0]
    compliances = [l for l in parsed if l[3] == 1]

    video_ids = sorted({l[0] for l in parsed})
    parquets  = {}
    for vid in video_ids:
        p = PARQUET_DIR / f'{vid}_interactions.parquet'
        if p.exists():
            parquets[vid] = pd.read_parquet(p)

    print(f"Loaded {len(parquets)} parquets, scoring {len(violations)} violations, "
          f"{len(compliances)} compliances...")

    v_scored = collect_scored(violations, score_violation, parquets)
    c_scored = collect_scored(compliances, score_compliance, parquets)

    if not v_scored or not c_scored:
        print("ERROR: no scored examples found.")
        return

    best_v, best_c = v_scored[0], c_scored[0]
    print(f"Best violation:  {best_v[1]} track={best_v[2]} roi={best_v[3]}  "
          f"mean_speed={np.mean(best_v[6]):.2f}  d_min={best_v[8]:.2f}m")
    print(f"Best compliance: {best_c[1]} track={best_c[2]} roi={best_c[3]}  "
          f"mean_speed={np.mean(best_c[6]):.2f}  d_min={best_c[8]:.2f}m")

    # Export JSON
    examples = []
    for lbl, result in [('violation', best_v), ('compliance', best_c)]:
        _, vid, tid, roi, ann, v_loc, v_sp, ped_data, dm = result
        examples.append({
            'label': lbl, 'video_id': vid,
            'tracking_id': int(tid), 'roi': roi, 'min_dmin_m': dm,
            'vehicle_loc_world': v_loc.tolist(),
            'vehicle_speed_ms':  v_sp.tolist(),
            'pedestrians': {
                str(pid): {'locs': locs.tolist(), 'dmin': dmin}
                for pid, (locs, dmin) in ped_data.items()
            },
        })
    with open(JSON_PATH, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"Saved trajectory examples to {JSON_PATH}")

    # Render
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(
        "World-Space BEV: Violation vs Compliance\n"
        "Trajectory colored by speed (red = fast, green = slow)",
        fontsize=13, fontweight='bold',
    )

    for ax, label_type, result in zip(axes, ['Violation', 'Compliance'], [best_v, best_c]):
        roi_str = result[3]
        roi_world = roi_bot_world if roi_str == 'BOT' else roi_top_world
        plot_event(ax, label_type, result, roi_world)

    # Shared legend items
    legend_items = [
        mpatches.Patch(facecolor='#FFD700', alpha=0.5, edgecolor='#B8860B',
                       linestyle='--', label='Crosswalk ROI (camera-projected)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue',
                   markersize=9, linestyle='None', label='Vehicle start'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='white',
                   markeredgecolor='gray', markersize=9, linestyle='None', label='Vehicle end'),
        mpatches.Patch(facecolor='red',   alpha=0.7, label='High speed'),
        mpatches.Patch(facecolor='green', alpha=0.7, label='Low speed'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=5,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = OUTPUT_DIR / 'bev_trajectories.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved BEV visualization to {out}")


if __name__ == '__main__':
    main()
