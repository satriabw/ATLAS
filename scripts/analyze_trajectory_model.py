"""
Post-hoc analysis of the trajectory-only CrossAttentionModel.

Produces:
  results/analysis/package.json      — per-event features, scores, categories
  results/analysis/model_analysis.txt — narrative report

Usage (run from project root):
  python scripts/analyze_trajectory_model.py
"""
import sys, json, pickle, logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))
from models import CrossAttentionModel
from evaluation.inference import _collect_events, _build_ped_stack
from evaluation.ap_calculator import compute_ap, compute_map, compute_pr_curve
from dataset.trajectory import DEFAULT_TOP_K, _to_frames, _to_loc

logging.basicConfig(level=logging.WARNING)

DATA_ROOT   = Path('/home/satria/Project/ATLAS')
PARQUET_DIR = DATA_ROOT / 'data/processed/interactions'
TEST_PKL    = DATA_ROOT / 'data/raw/labels/test_labels.pkl'
CKPT        = DATA_ROOT / 'training/checkpoints/best_model.pth'
OUTPUT_DIR  = DATA_ROOT / 'results/analysis'
TEST_VIDS   = [f"video_{n:03d}" for n in range(2, 121, 2)]
NUM_FRAMES  = 32
TOP_K       = DEFAULT_TOP_K


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(device):
    model = CrossAttentionModel(num_classes=2, top_k=TOP_K, num_frames=NUM_FRAMES).to(device)
    ckpt  = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint  epoch={ckpt['epoch']}  val_acc={ckpt.get('val_acc', float('nan')):.1f}%")
    return model


def run_inference_batch(model, events, device, v_mod=None, p_mod=None):
    """
    Run batched inference; optionally apply feature modifier callables:
      v_mod(v_tensor) → modified vehicle features
      p_mod(p_tensor) → modified pedestrian features
    Returns list of P(violation) scores.
    """
    v_trajs = np.stack([e['_v_traj'] for e in events])
    p_trajs = np.stack([e['_p_traj'] for e in events])
    v_masks = np.stack([e['_v_mask'] for e in events])
    p_masks = np.stack([e['_p_mask'] for e in events])

    scores = []
    for start in range(0, len(events), 64):
        sl = slice(start, start + 64)
        v = torch.from_numpy(v_trajs[sl]).to(device)
        p = torch.from_numpy(p_trajs[sl]).to(device)
        vm = torch.from_numpy(v_masks[sl]).to(device)
        pm = torch.from_numpy(p_masks[sl]).to(device)
        if v_mod: v = v_mod(v)
        if p_mod: p = p_mod(p)
        with torch.no_grad():
            logits = model(v, p, vm, pm)
            scores.extend(F.softmax(logits, dim=1)[:, 0].cpu().tolist())
    return scores


def extract_attention(model, events, device, indices):
    """Return attention weights (T_query × top_k) for selected event indices."""
    attn_cache = {}
    def hook(module, inp, out):
        attn_cache['w'] = out[1].detach().cpu().numpy()   # (B, T, K)

    handle = model.cross_attn.register_forward_hook(hook)
    results = {}
    for idx in indices:
        ev = events[idx]
        v = torch.from_numpy(ev['_v_traj'][None]).to(device)
        p = torch.from_numpy(ev['_p_traj'][None]).to(device)
        vm = torch.from_numpy(ev['_v_mask'][None]).to(device)
        pm = torch.from_numpy(ev['_p_mask'][None]).to(device)
        with torch.no_grad():
            model(v, p, vm, pm)
        if 'w' in attn_cache:
            results[idx] = attn_cache['w'][0]   # (T, K)
            attn_cache.clear()
    handle.remove()
    return results


def load_traj_features(events):
    """Augment events with raw trajectory statistics from parquet."""
    parquets = {}
    for ev in events:
        vid = ev['video_id']
        if vid not in parquets:
            p = PARQUET_DIR / f'{vid}_interactions.parquet'
            parquets[vid] = pd.read_parquet(p) if p.exists() else None
        df = parquets[vid]
        if df is None:
            ev['traj'] = None
            continue

        g = df[(df['v_track_id'] == ev['v_track_id']) & (df['roi'] == ev['roi'])]
        if g.empty:
            ev['traj'] = None
            continue

        speeds, locs, frames = [], [], []
        for _, row in g.iterrows():
            f  = np.asarray(row['frames'], dtype=np.int64).ravel()
            sp = np.asarray(row['v_speed'], dtype=np.float32).ravel()
            lc = _to_loc(row['v_loc_planar'])
            n  = min(len(f), len(sp), len(lc))
            frames.append(f[:n]); speeds.append(sp[:n]); locs.append(lc[:n])

        all_f   = np.concatenate(frames)
        all_sp  = np.concatenate(speeds)
        all_lc  = np.vstack(locs)
        order   = np.argsort(all_f)
        _, keep = np.unique(all_f[order], return_index=True)
        idx     = order[keep]
        sp      = all_sp[idx]
        lc      = all_lc[idx]

        diffs = np.diff(lc, axis=0)
        arc   = float(np.sqrt((diffs ** 2).sum(axis=1)).sum())

        # Speed profile: split into thirds
        n = len(sp)
        sp_first = sp[:max(n // 3, 1)]
        sp_last  = sp[max(n * 2 // 3, 1):]

        ev['traj'] = {
            'n_frames':       int(n),
            'mean_speed':     float(np.mean(sp)),
            'max_speed':      float(np.max(sp)),
            'min_speed':      float(np.min(sp)),
            'approach_speed': float(np.mean(sp_first)),
            'exit_speed':     float(np.mean(sp_last)),
            'decel':          float(np.mean(sp_first) - np.mean(sp_last)),
            'arc_length_m':   arc,
            'n_peds':         int(g['p_track_id'].nunique()),
            'min_dmin':       float(g['d_min'].apply(
                lambda x: float(np.asarray(x).ravel().min())).min()),
        }
    return events


# ── failure categorization ────────────────────────────────────────────────────

def categorize_failure(ev):
    t = ev.get('traj')
    if t is None:
        return 'missing_data'
    gt = ev['gt_label']   # 0 = violation, 1 = compliance
    score = ev['score']

    if gt == 0 and score < 0.5:     # FN — missed violation
        if t['n_frames'] < 80:
            return 'short_traj_fn'
        if t['min_speed'] < 0.3 and t['decel'] > 1.0:
            return 'brief_stop_fn'
        if t['mean_speed'] < 1.8:
            return 'slow_creep_fn'
        return 'ambiguous_fn'
    else:                            # FP — false alarm
        if t['approach_speed'] > 3.0 and t['exit_speed'] < 1.5:
            return 'high_approach_stops_fp'
        if t['min_dmin'] < 2.0:
            return 'close_ped_fp'
        return 'ambiguous_fp'


# ── group stats helpers ───────────────────────────────────────────────────────

def compute_group_stats(evs):
    """Median trajectory stats for a group of events."""
    keys = ['n_frames', 'mean_speed', 'approach_speed', 'exit_speed', 'decel', 'min_dmin']
    out = {'count': len(evs)}
    for k in keys:
        vals = [e['traj'][k] for e in evs if e.get('traj') and e['traj'].get(k) is not None]
        out[k] = float(np.median(vals)) if vals else float('nan')
    return out


def _stat_table(a_stats, a_label, b_stats, b_label, keys):
    """Two-column median comparison table."""
    rows = [f"  {'Feature':<20} {a_label:>14}  {b_label:>14}  {'Diff':>8}"]
    rows.append("  " + "─" * 60)
    for k in keys:
        a, b = a_stats.get(k, float('nan')), b_stats.get(k, float('nan'))
        diff = a - b if not (np.isnan(a) or np.isnan(b)) else float('nan')
        unit = ' m/s' if ('speed' in k or k == 'decel') else (' m' if 'dmin' in k else '    ')
        rows.append(f"  {k:<20} {a:>11.2f}{unit}  {b:>11.2f}{unit}  {diff:>+8.2f}")
    return '\n'.join(rows)


def compute_speed_band(all_events, lo, hi):
    """Violation rate for events where mean_speed is in [lo, hi]."""
    band = [e for e in all_events
            if e.get('traj') and lo <= e['traj'].get('mean_speed', -1) <= hi]
    if not band:
        return 0, 0, float('nan')
    n_viol = sum(1 for e in band if e['gt_label'] == 0)
    return len(band), n_viol, n_viol / len(band)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model  = load_model(device)
    events = _collect_events(PARQUET_DIR, TEST_PKL, TEST_VIDS, NUM_FRAMES, TOP_K)
    print(f"Test events: {len(events)}")

    # ── Baseline inference ──
    print("Running baseline inference...")
    scores = run_inference_batch(model, events, device)
    for ev, sc in zip(events, scores):
        ev['score']   = sc
        ev['score_n'] = 1.0 - sc

    # ── Load trajectory features ──
    print("Loading trajectory features from parquets...")
    events = load_traj_features(events)

    # ── Baseline AP ──
    baseline_map = compute_map(events)
    print(f"\nBaseline  APv={baseline_map['APv']:.3f}  APn={baseline_map['APn']:.3f}  mAP={baseline_map['mAP']:.3f}")

    # ── Feature ablation ──
    print("Running feature ablation...")
    ablations = {}

    # Speed zeroed (column 2 of vehicle, column 2 of ped)
    def zero_v_speed(v): v = v.clone(); v[:, :, 2] = 0; return v
    def zero_p_speed(p): p = p.clone(); p[:, :, 2] = 0; return p
    sc_sp = run_inference_batch(model, events, device, v_mod=zero_v_speed, p_mod=zero_p_speed)
    evs_sp = [dict(e, score=s, score_n=1-s) for e, s in zip(events, sc_sp)]
    ablations['speed_zeroed'] = compute_map(evs_sp)

    # Position zeroed (columns 0-1 of vehicle, columns 0-1 of ped)
    def zero_v_pos(v): v = v.clone(); v[:, :, :2] = 0; return v
    def zero_p_pos(p): p = p.clone(); p[:, :, :2] = 0; return p
    sc_pos = run_inference_batch(model, events, device, v_mod=zero_v_pos, p_mod=zero_p_pos)
    evs_pos = [dict(e, score=s, score_n=1-s) for e, s in zip(events, sc_pos)]
    ablations['position_zeroed'] = compute_map(evs_pos)

    # Pedestrian zeroed (all ped features = 0)
    def zero_ped(p): return torch.zeros_like(p)
    sc_ped = run_inference_batch(model, events, device, p_mod=zero_ped)
    evs_ped = [dict(e, score=s, score_n=1-s) for e, s in zip(events, sc_ped)]
    ablations['ped_zeroed'] = compute_map(evs_ped)

    print(f"  speed_zeroed   APv={ablations['speed_zeroed']['APv']:.3f}  mAP={ablations['speed_zeroed']['mAP']:.3f}")
    print(f"  position_zeroed APv={ablations['position_zeroed']['APv']:.3f}  mAP={ablations['position_zeroed']['mAP']:.3f}")
    print(f"  ped_zeroed      APv={ablations['ped_zeroed']['APv']:.3f}  mAP={ablations['ped_zeroed']['mAP']:.3f}")

    # ── Classify TP/TN/FP/FN ──
    for ev in events:
        pred = 0 if ev['score'] >= 0.5 else 1
        if ev['gt_label'] == pred:
            ev['outcome'] = 'TP' if ev['gt_label'] == 0 else 'TN'
        else:
            ev['outcome'] = 'FN' if ev['gt_label'] == 0 else 'FP'

    tp_evs = [e for e in events if e['outcome'] == 'TP']
    tn_evs = [e for e in events if e['outcome'] == 'TN']
    fp_evs = [e for e in events if e['outcome'] == 'FP']
    fn_evs = [e for e in events if e['outcome'] == 'FN']
    print(f"\nConfusion: TP={len(tp_evs)}  TN={len(tn_evs)}  FP={len(fp_evs)}  FN={len(fn_evs)}")

    n_viol  = sum(1 for e in events if e['gt_label'] == 0)
    n_comp  = sum(1 for e in events if e['gt_label'] == 1)
    sens    = len(tp_evs) / max(n_viol, 1)   # recall on violations
    spec    = len(tn_evs) / max(n_comp, 1)   # recall on compliances
    print(f"Sensitivity (violation recall): {sens:.2f}")
    print(f"Specificity (compliance recall): {spec:.2f}")

    # ── Categorize failures ──
    for ev in events:
        if ev['outcome'] in ('FP', 'FN'):
            ev['failure_category'] = categorize_failure(ev)

    from collections import Counter
    fn_cats = Counter(e['failure_category'] for e in fn_evs if 'failure_category' in e)
    fp_cats = Counter(e['failure_category'] for e in fp_evs if 'failure_category' in e)
    print(f"\nFN categories: {dict(fn_cats)}")
    print(f"FP categories: {dict(fp_cats)}")

    # ── Select top failure/success cases ──
    # Failures: most confident wrong predictions
    failures = sorted(
        [e for e in fn_evs] + [e for e in fp_evs],
        key=lambda e: abs(e['score'] - 0.5), reverse=True
    )[:10]

    # Successes: most confident correct predictions, balanced by class
    successes_v = sorted(tp_evs, key=lambda e: e['score'],  reverse=True)[:5]
    successes_n = sorted(tn_evs, key=lambda e: e['score_n'], reverse=True)[:5]
    successes   = successes_v + successes_n

    # ── Attention weights for failures ──
    failure_indices = [events.index(e) for e in failures[:6]]
    print("\nExtracting attention weights for top failures...")
    attn_weights = extract_attention(model, events, device, failure_indices)

    # Summarize: for each case, which ped slot gets highest average attention
    attn_summary = {}
    for idx, weights in attn_weights.items():
        ev = events[idx]
        # weights: (T, K) — average over valid (unpadded) timesteps
        v_mask = ev['_v_mask'] if '_v_mask' in ev else np.zeros(NUM_FRAMES, dtype=bool)
        valid  = ~v_mask if hasattr(v_mask, '__len__') else np.ones(NUM_FRAMES, dtype=bool)
        avg_w  = weights[valid].mean(axis=0) if valid.any() else weights.mean(axis=0)
        attn_summary[idx] = {
            'top_ped_slot':    int(np.argmax(avg_w)),
            'ped_attn_dist':   avg_w.tolist(),
            'attn_entropy':    float(-np.sum(avg_w * np.log(avg_w + 1e-9))),
        }

    # ── Build analysis package ──
    def serialise(ev):
        out = {k: v for k, v in ev.items() if not k.startswith('_')}
        # drop large arrays
        return out

    package = {
        'baseline_ap':      baseline_map,
        'ablations':        ablations,
        'confusion':        {'TP': len(tp_evs), 'TN': len(tn_evs),
                             'FP': len(fp_evs), 'FN': len(fn_evs)},
        'sensitivity':      round(sens, 4),
        'specificity':      round(spec, 4),
        'fn_categories':    dict(fn_cats),
        'fp_categories':    dict(fp_cats),
        'failure_cases':    [serialise(e) for e in failures],
        'success_cases':    [serialise(e) for e in successes],
        'attn_summary':     {str(k): v for k, v in attn_summary.items()},
    }

    pkg_path = OUTPUT_DIR / 'package.json'
    with open(pkg_path, 'w') as f:
        json.dump(package, f, indent=2, default=float)
    print(f"\nSaved analysis package → {pkg_path}")

    # Save all events (score, outcome, traj features) for visualization
    all_events_out = []
    for ev in events:
        t = ev.get('traj') or {}
        all_events_out.append({
            'score':            ev['score'],
            'gt_label':         ev['gt_label'],
            'outcome':          ev['outcome'],
            'failure_category': ev.get('failure_category', ''),
            'mean_speed':       t.get('mean_speed'),
            'n_frames':         t.get('n_frames'),
            'min_dmin':         t.get('min_dmin'),
            'arc_length_m':     t.get('arc_length_m'),
            'decel':            t.get('decel'),
        })
    all_events_path = OUTPUT_DIR / 'all_events.json'
    with open(all_events_path, 'w') as f:
        json.dump(all_events_out, f, default=float)
    print(f"Saved all events     → {all_events_path}")

    # ── Narrative ──
    write_narrative(package, failures, successes, attn_summary, events)


def write_narrative(pkg, failures, successes, attn_summary, all_events):
    bl   = pkg['baseline_ap']
    abl  = pkg['ablations']
    conf = pkg['confusion']

    speed_drop = bl['APv'] - abl['speed_zeroed']['APv']
    pos_drop   = bl['APv'] - abl['position_zeroed']['APv']
    ped_drop   = bl['APv'] - abl['ped_zeroed']['APv']

    fn_cats = pkg['fn_categories']
    fp_cats = pkg['fp_categories']

    # ── Group stats ──
    tp_evs = [e for e in all_events if e['outcome'] == 'TP']
    tn_evs = [e for e in all_events if e['outcome'] == 'TN']
    fp_evs = [e for e in all_events if e['outcome'] == 'FP']
    fn_evs = [e for e in all_events if e['outcome'] == 'FN']

    tp_stats = compute_group_stats(tp_evs)
    fn_stats = compute_group_stats(fn_evs)
    tn_stats = compute_group_stats(tn_evs)
    fp_stats = compute_group_stats(fp_evs)

    cat_stats = {}
    for cat in set(list(fn_cats.keys()) + list(fp_cats.keys())):
        cat_evs = [e for e in all_events if e.get('failure_category') == cat]
        cat_stats[cat] = compute_group_stats(cat_evs)

    stat_keys = ['mean_speed', 'approach_speed', 'exit_speed', 'n_frames', 'min_dmin', 'decel']
    viol_table = _stat_table(tp_stats, f'TP (n={tp_stats["count"]})',
                             fn_stats, f'FN (n={fn_stats["count"]})', stat_keys)
    comp_table = _stat_table(tn_stats, f'TN (n={tn_stats["count"]})',
                             fp_stats, f'FP (n={fp_stats["count"]})', stat_keys)

    # Ambiguity band
    band_n, band_viol, band_viol_rate = compute_speed_band(all_events, 1.0, 2.0)

    def cat_stat_line(cat):
        s = cat_stats.get(cat, {})
        if not s or s.get('count', 0) == 0:
            return '    No events.'
        return (
            f"    Computed medians (n={s['count']}): "
            f"mean_speed={s.get('mean_speed', float('nan')):.2f} m/s  "
            f"approach={s.get('approach_speed', float('nan')):.2f} m/s  "
            f"exit={s.get('exit_speed', float('nan')):.2f} m/s  "
            f"decel={s.get('decel', float('nan')):.2f} m/s  "
            f"n_frames={s.get('n_frames', float('nan')):.0f}  "
            f"min_dmin={s.get('min_dmin', float('nan')):.1f} m"
        )

    # Describe top failure cases
    failure_descs = []
    for ev in failures[:8]:
        t = ev.get('traj', {}) or {}
        cat = ev.get('failure_category', 'unknown')
        desc = (
            f"  [{ev['outcome']}/{cat}] {ev['video_id']} track={ev['v_track_id']} {ev['roi']} "
            f"score={ev['score']:.2f} "
            f"mean_sp={t.get('mean_speed', float('nan')):.2f} "
            f"n_frames={t.get('n_frames', '?')} "
            f"d_min={t.get('min_dmin', float('nan')):.1f}m"
        )
        failure_descs.append(desc)

    # Describe top success cases
    success_descs = []
    for ev in successes[:8]:
        t = ev.get('traj', {}) or {}
        desc = (
            f"  [{ev['outcome']}] {ev['video_id']} track={ev['v_track_id']} "
            f"score={ev['score']:.2f} "
            f"mean_sp={t.get('mean_speed', float('nan')):.2f} "
            f"n_frames={t.get('n_frames', '?')} "
            f"d_min={t.get('min_dmin', float('nan')):.1f}m"
        )
        success_descs.append(desc)

    # Attn entropy summary
    entropies = [v['attn_entropy'] for v in attn_summary.values()]
    avg_entropy = np.mean(entropies) if entropies else float('nan')
    max_entropy = np.log(DEFAULT_TOP_K)

    text = f"""Trajectory-Only Model Analysis: CrossAttentionModel (BiGRU + Cross-Attention)
================================================================================

1. OVERALL PERFORMANCE
----------------------
Test set: even-numbered videos (2–120), {sum(conf.values())} labeled events
  Violations:  {conf['TP'] + conf['FN']} GT  |  Compliances: {conf['TN'] + conf['FP']} GT

APv  : {bl['APv']:.3f}   (violation detection precision-recall area)
APn  : {bl['APn']:.3f}   (compliance detection precision-recall area)
mAP  : {bl['mAP']:.3f}

At threshold 0.5:
  TP={conf['TP']}  TN={conf['TN']}  FP={conf['FP']}  FN={conf['FN']}
  Sensitivity (violation recall): {pkg['sensitivity']:.2f}
  Specificity (compliance recall): {pkg['specificity']:.2f}

{'Bias toward predicting compliance (misses violations).' if pkg['sensitivity'] < pkg['specificity'] else 'Bias toward predicting violation (over-triggers).'}


2. WHERE THE MODEL SUCCEEDS
----------------------------
Top success cases (most confident correct predictions):

{chr(10).join(success_descs)}

TP vs FN — median feature comparison (violations only):
{viol_table}

TN vs FP — median feature comparison (compliances only):
{comp_table}


3. WHERE THE MODEL FAILS
-------------------------
Failure category breakdown:
  False Negatives (missed violations):  {sum(fn_cats.values())} total
    {chr(10).join(f'    {k}: {v}' for k, v in sorted(fn_cats.items(), key=lambda x: -x[1]))}

  False Positives (false alarms):  {sum(fp_cats.values())} total
    {chr(10).join(f'    {k}: {v}' for k, v in sorted(fp_cats.items(), key=lambda x: -x[1]))}

Top failure cases:

{chr(10).join(failure_descs)}

Failure type analysis:
  (Category thresholds defined in categorize_failure(); stats computed from actual events.)

  slow_creep_fn ({fn_cats.get('slow_creep_fn', 0)} cases):
    Threshold: mean_speed < 1.8 m/s, n_frames >= 80, min_speed >= 0.3 or decel <= 1.0
{cat_stat_line('slow_creep_fn')}

  brief_stop_fn ({fn_cats.get('brief_stop_fn', 0)} cases):
    Threshold: min_speed < 0.3 m/s and decel > 1.0 m/s, n_frames >= 80
{cat_stat_line('brief_stop_fn')}

  short_traj_fn ({fn_cats.get('short_traj_fn', 0)} cases):
    Threshold: n_frames < 80
{cat_stat_line('short_traj_fn')}

  ambiguous_fn ({fn_cats.get('ambiguous_fn', 0)} cases):
    Missed violations not captured by any above threshold.
{cat_stat_line('ambiguous_fn')}

  high_approach_stops_fp ({fp_cats.get('high_approach_stops_fp', 0)} cases):
    Threshold: approach_speed > 3.0 m/s and exit_speed < 1.5 m/s
{cat_stat_line('high_approach_stops_fp')}

  close_ped_fp ({fp_cats.get('close_ped_fp', 0)} cases):
    Threshold: min_dmin < 2.0 m
{cat_stat_line('close_ped_fp')}

  ambiguous_fp ({fp_cats.get('ambiguous_fp', 0)} cases):
    False alarms not captured by any above threshold.
{cat_stat_line('ambiguous_fp')}


4. FEATURE IMPORTANCE (ABLATION)
----------------------------------
APv when each feature group is zeroed out:

  Baseline (full features): {bl['APv']:.3f}
  Speed zeroed (v_speed + p_speed = 0): {abl['speed_zeroed']['APv']:.3f}  (Δ = {speed_drop:+.3f})
  Position zeroed (v_xy + p_rel_xy = 0): {abl['position_zeroed']['APv']:.3f}  (Δ = {pos_drop:+.3f})
  Pedestrian zeroed (all ped features = 0): {abl['ped_zeroed']['APv']:.3f}  (Δ = {ped_drop:+.3f})

{'Speed' if speed_drop >= pos_drop and speed_drop >= ped_drop else 'Position' if pos_drop >= ped_drop else 'Pedestrian context'} is the most important feature group (largest APv drop when removed).

  - Speed signal (Δ={speed_drop:+.3f}): {'Strong discriminator — velocity profile provides the primary violation signal.' if abs(speed_drop) > 0.02 else 'Weak. Model relies more on position than speed, suggesting positional heuristics rather than behavioral dynamics.'}

  - Position signal (Δ={pos_drop:+.3f}): {'Important for spatial context — where in the ROI the vehicle is relative to the crosswalk.' if abs(pos_drop) > 0.02 else 'Adds little. Model does not strongly use absolute world coordinates.'}

  - Pedestrian context (Δ={ped_drop:+.3f}): {'Cross-attention to pedestrians provides meaningful lift — validates the architecture choice.' if abs(ped_drop) > 0.02 else 'Minimal lift. Model classifies mostly on vehicle trajectory alone, ignoring pedestrian features despite the cross-attention design.'}


5. ATTENTION WEIGHT ANALYSIS
------------------------------
Average attention entropy over failure cases: {avg_entropy:.3f}
Maximum possible entropy (uniform over {DEFAULT_TOP_K} peds): {max_entropy:.3f}
Normalized entropy: {avg_entropy/max_entropy:.2f}

{'High normalized entropy (> 0.7): attention is near-uniform across pedestrian slots — no meaningful pedestrian selection learned.' if avg_entropy/max_entropy > 0.7 else 'Moderate normalized entropy: some pedestrian selectivity exists, but may not track geometric proximity.'}

Per-failure attention summary:
  (Slot 0 = closest pedestrian by d_min; slots 1–{DEFAULT_TOP_K-1} = progressively farther)

  {chr(10)  .join(f"  Event idx={k}: top_slot={v['top_ped_slot']} entropy={v['attn_entropy']:.2f}" for k, v in pkg['attn_summary'].items())}


6. SPEED AMBIGUITY BAND AND FUSION MOTIVATION
-----------------------------------------------
Speed band [1.0–2.0 m/s]: {band_n} events, {band_viol} violations ({band_viol_rate*100:.1f}% violation rate)
{'This band is effectively a coin-flip for the trajectory-only model.' if 0.3 <= band_viol_rate <= 0.7 else 'The model has some signal in this speed range, but failure modes above suggest visual context would still help.'}

Failure modes and the signal they lack:
  Failure mode                 Count   What trajectory cannot determine
  ─────────────────────────────────────────────────────────────────────
  slow_creep_fn                {fn_cats.get('slow_creep_fn', 0):>5}   Whether path crosses ped zone vs. genuine slow yield
  brief_stop_fn                {fn_cats.get('brief_stop_fn', 0):>5}   Whether ped cleared before vehicle re-accelerated
  short_traj_fn                {fn_cats.get('short_traj_fn', 0):>5}   Insufficient context; even a single frame helps
  high_approach_stops_fp       {fp_cats.get('high_approach_stops_fp', 0):>5}   Whether vehicle stopped at the boundary
  ambiguous_fn + ambiguous_fp  {fn_cats.get('ambiguous_fn', 0) + fp_cats.get('ambiguous_fp', 0):>5}   No single trajectory feature separates classes
"""

    out = OUTPUT_DIR / 'model_analysis.txt'
    with open(out, 'w') as f:
        f.write(text)
    print(f"Saved narrative  → {out}")
    print("\n" + "=" * 60)
    print(text)


if __name__ == '__main__':
    main()
