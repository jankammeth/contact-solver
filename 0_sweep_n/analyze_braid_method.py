#!/usr/bin/env python3
"""Analyze whether braid word method correctly identifies homotopy differences."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from contact_solver import ContactSolver, LiftedSCP, generate_random_positions, load_config
from contact_solver.cli._sweep_utils import build_scaled_config, compute_braid_word, normalize_braid_word, compare_braid_words


def analyze_case(N, seed, K=200):
    """Run both solvers and analyze braid word agreement vs trajectory difference."""
    print(f"\n{'='*70}")
    print(f"Analyzing N={N}, seed={seed}, K={K}")
    print('='*70)
    
    # Build config
    base_config = load_config("sweep_n")
    config = build_scaled_config(base_config, N, seed, K=K)
    
    # Generate scenario
    np.random.seed(config.random_seed)
    initial, final = generate_random_positions(config)
    
    # Run LiftedSCP
    print("\nRunning LiftedSCP...", flush=True)
    scp = LiftedSCP(config, verbose=False)
    scp.set_initial_states(initial)
    scp.set_final_states(final)
    scp_result = scp.generate_trajectories()
    scp_pos = scp_result["trajectories"]["positions"]
    print(f"  SCP: {scp_result['metrics']['convergence_reason']}, "
          f"time={scp_result['metrics']['timing']['total_time']:.2f}s")
    
    # Run ContactSolver
    print("Running ContactSolver...", flush=True)
    contact = ContactSolver(config, verbose=False)
    contact.set_initial_states(initial)
    contact.set_final_states(final)
    contact_result = contact.generate_trajectories()
    
    # Sample at SCP grid
    h = config.problem.timestep
    scp_times = np.arange(1, K + 1) * h
    _, contact_pos, _, _ = contact.sample(times=scp_times)
    print(f"  SCI: {contact_result['metrics']['convergence_reason']}, "
          f"time={contact_result['metrics']['timing']['total_time']:.2f}s, "
          f"contacts={contact_result['metrics']['num_contacts']}")
    
    # Compute trajectory RMSE
    rmse_vals = []
    for i in range(N):
        diff = scp_pos[i] - contact_pos[i]
        rmse_vals.append(np.sqrt(np.mean(diff**2)))
    overall_rmse = np.sqrt(np.mean([np.mean((scp_pos[i] - contact_pos[i])**2) for i in range(N)]))
    max_dev = max(np.max(np.linalg.norm(scp_pos[i] - contact_pos[i], axis=1)) for i in range(N))
    
    print(f"\nTrajectory differences:")
    print(f"  Overall RMSE: {overall_rmse:.6f}")
    print(f"  Max deviation: {max_dev:.6f}")
    print(f"  Per-robot RMSE: {[f'{r:.4f}' for r in rmse_vals]}")
    
    # Compute braid words (x-axis projection)
    print(f"\nBraid word analysis (x-axis projection):")
    braid_scp_x = compute_braid_word(scp_pos, axis=0, min_sep=0.0)
    braid_sci_x = compute_braid_word(contact_pos, axis=0, min_sep=0.0)
    norm_scp_x = normalize_braid_word(braid_scp_x)
    norm_sci_x = normalize_braid_word(braid_sci_x)
    
    print(f"  SCP crossings (x): {len(braid_scp_x)}")
    print(f"  SCI crossings (x): {len(braid_sci_x)}")
    print(f"  Normalized SCP (x): {len(norm_scp_x)}")
    print(f"  Normalized SCI (x): {len(norm_sci_x)}")
    
    # Detailed comparison
    from collections import Counter
    counter_scp_x = Counter(braid_scp_x)
    counter_sci_x = Counter(braid_sci_x)
    
    # Check for differences
    all_keys = set(counter_scp_x.keys()) | set(counter_sci_x.keys())
    diffs_x = []
    for key in sorted(all_keys):
        count_scp = counter_scp_x.get(key, 0)
        count_sci = counter_sci_x.get(key, 0)
        if count_scp != count_sci:
            diffs_x.append((key, count_scp, count_sci))
    
    if diffs_x:
        print(f"\n  ⚠️  X-axis crossing differences found:")
        for (i, j, sign), count_scp, count_sci in diffs_x[:20]:
            print(f"    Pair ({i},{j}) sign={sign:+d}: SCP={count_scp}, SCI={count_sci}")
        if len(diffs_x) > 20:
            print(f"    ... and {len(diffs_x) - 20} more")
    else:
        print(f"  ✓ X-axis braid words match perfectly")
    
    # Compute braid words (y-axis projection) 
    print(f"\nBraid word analysis (y-axis projection):")
    braid_scp_y = compute_braid_word(scp_pos, axis=1, min_sep=0.0)
    braid_sci_y = compute_braid_word(contact_pos, axis=1, min_sep=0.0)
    
    print(f"  SCP crossings (y): {len(braid_scp_y)}")
    print(f"  SCI crossings (y): {len(braid_sci_y)}")
    
    counter_scp_y = Counter(braid_scp_y)
    counter_sci_y = Counter(braid_sci_y)
    
    all_keys_y = set(counter_scp_y.keys()) | set(counter_sci_y.keys())
    diffs_y = []
    for key in sorted(all_keys_y):
        count_scp = counter_scp_y.get(key, 0)
        count_sci = counter_sci_y.get(key, 0)
        if count_scp != count_sci:
            diffs_y.append((key, count_scp, count_sci))
    
    if diffs_y:
        print(f"\n  ⚠️  Y-axis crossing differences found:")
        for (i, j, sign), count_scp, count_sci in diffs_y[:20]:
            print(f"    Pair ({i},{j}) sign={sign:+d}: SCP={count_scp}, SCI={count_sci}")
        if len(diffs_y) > 20:
            print(f"    ... and {len(diffs_y) - 20} more")
    else:
        print(f"  ✓ Y-axis braid words match perfectly")
    
    # Official comparison result
    comparison = compare_braid_words(braid_scp_x, braid_sci_x)
    print(f"\nOfficial braid word comparison (x-axis only):")
    print(f"  same_homotopy_class: {comparison['same_homotopy_class']}")
    print(f"  n_crossings_scp: {comparison['n_crossings_scp']}")
    print(f"  n_crossings_contact: {comparison['n_crossings_contact']}")
    
    # Conclusion
    print(f"\n{'─'*70}")
    if comparison['same_homotopy_class'] and overall_rmse > 0.01:
        print("⚠️  POTENTIAL BRAID METHOD LIMITATION:")
        print(f"   Braid words agree but trajectories differ significantly (RMSE={overall_rmse:.6f})")
        print("   This suggests either:")
        print("   1. Homotopy difference not captured by 1D x-projection")
        print("   2. Same homotopy class but different trajectory parameterization")
        print("   3. Numerical/discretization effects")
    elif not comparison['same_homotopy_class']:
        print("✓ Braid words disagree - correctly identified different homotopy classes")
    else:
        print("✓ Braid words agree and trajectories match well - consistent result")
    
    return {
        'N': N,
        'seed': seed,
        'rmse': overall_rmse,
        'max_dev': max_dev,
        'same_braid_x': comparison['same_homotopy_class'],
        'diffs_x': len(diffs_x),
        'diffs_y': len(diffs_y),
        'n_crossings_scp_x': len(braid_scp_x),
        'n_crossings_sci_x': len(braid_sci_x),
        'n_crossings_scp_y': len(braid_scp_y),
        'n_crossings_sci_y': len(braid_sci_y),
    }


def main():
    # Analyze the extreme outlier
    print("\n" + "="*70)
    print("INVESTIGATING BRAID WORD METHOD RELIABILITY")
    print("="*70)
    
    cases = [
        (18, 17),  # Extreme outlier: RMSE=0.108
        (18, 4),   # Second outlier: RMSE=0.043
        (12, 11),  # Third outlier: RMSE=0.041
        (18, 35),  # Weak outlier: RMSE=0.012
    ]
    
    results = []
    for N, seed in cases:
        result = analyze_case(N, seed, K=200)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'N':>3} {'seed':>4} {'RMSE':>10} {'max_dev':>10} {'braid_x':>8} {'diff_x':>7} {'diff_y':>7}")
    print("─"*70)
    for r in results:
        print(f"{r['N']:3d} {r['seed']:4d} {r['rmse']:10.6f} {r['max_dev']:10.6f} "
              f"{'✓' if r['same_braid_x'] else '✗':>8} {r['diffs_x']:7d} {r['diffs_y']:7d}")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    extreme_case = results[0]
    if extreme_case['same_braid_x'] and extreme_case['rmse'] > 0.01:
        print("⚠️  The braid word method (x-projection) marks N=18, seed=17 as")
        print("   'same homotopy class' despite RMSE=0.108, suggesting:")
        print()
        print("   1. The 1D braid projection may miss some homotopy differences")
        print("   2. OR the trajectories are homotopically equivalent but take")
        print("      very different spatial paths through the same topology")
        print()
        print("   Recommendation: Use both x and y projections, or check")
        print("   crossing number differences more carefully.")
    else:
        print("✓ Braid word method appears reliable for these cases")


if __name__ == "__main__":
    main()
