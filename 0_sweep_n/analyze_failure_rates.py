#!/usr/bin/env python3
"""Analyze solver failure rates from sweep results."""

import pandas as pd

def analyze_failure_rates(csv_path):
    df = pd.read_csv(csv_path)
    
    total = len(df)
    scp_failures = (~df['scp_success']).sum()
    sci_failures = (~df['contact_success']).sum()
    scp_fail_rate = 100 * scp_failures / total
    sci_fail_rate = 100 * sci_failures / total
    
    print('=' * 70)
    print('SOLVER FAILURE RATES (K=200, N=2-20, 50 seeds/N)')
    print('=' * 70)
    print(f'Total cases: {total}')
    print()
    print(f'LiftedSCP (discrete QP-based):')
    print(f'  Failures: {scp_failures:4d} / {total}  ({scp_fail_rate:.2f}%)')
    print(f'  Success:  {total - scp_failures:4d} / {total}  ({100 - scp_fail_rate:.2f}%)')
    print()
    print(f'ContactSolver (continuous analytical):')
    print(f'  Failures: {sci_failures:4d} / {total}  ({sci_fail_rate:.2f}%)')
    print(f'  Success:  {total - sci_failures:4d} / {total}  ({100 - sci_fail_rate:.2f}%)')
    print()
    
    # Breakdown by failure reason
    print('-' * 70)
    print('Failure reasons breakdown:')
    print('-' * 70)
    print('\nLiftedSCP failures:')
    if scp_failures > 0:
        scp_reasons = df[~df['scp_success']]['scp_reason'].value_counts()
        for reason, count in scp_reasons.items():
            print(f'  {reason:20s}: {count:4d} ({100*count/scp_failures:.1f}% of failures)')
    else:
        print('  None!')
        
    print('\nContactSolver failures:')
    if sci_failures > 0:
        sci_reasons = df[~df['contact_success']]['contact_reason'].value_counts()
        for reason, count in sci_reasons.items():
            print(f'  {reason:20s}: {count:4d} ({100*count/sci_failures:.1f}% of failures)')
    else:
        print('  None!')
    
    # Failure rate by N
    print()
    print('=' * 80)
    print('FAILURE RATE BY NUMBER OF ROBOTS')
    print('=' * 80)
    print(f'{"N":>3s} {"Cases":>6s} {"SCP Fail":>10s} {"SCP Rate":>10s} {"SCI Fail":>10s} {"SCI Rate":>10s}')
    print('-' * 80)
    
    for n in sorted(df['N'].unique()):
        n_df = df[df['N'] == n]
        n_total = len(n_df)
        n_scp_fail = (~n_df['scp_success']).sum()
        n_sci_fail = (~n_df['contact_success']).sum()
        
        n_scp_rate = 100 * n_scp_fail / n_total if n_total > 0 else 0
        n_sci_rate = 100 * n_sci_fail / n_total if n_total > 0 else 0
        
        print(f'{n:3d} {n_total:6d} {n_scp_fail:10d} {n_scp_rate:9.1f}% {n_sci_fail:10d} {n_sci_rate:9.1f}%')
    
    print('=' * 80)
    
    # Show which specific cases failed
    if scp_failures > 0 or sci_failures > 0:
        print()
        print('=' * 70)
        print('SPECIFIC FAILURE CASES')
        print('=' * 70)
        
        if scp_failures > 0:
            print('\nLiftedSCP failures:')
            scp_fail_df = df[~df['scp_success']][['N', 'seed', 'scp_reason']]
            for _, row in scp_fail_df.iterrows():
                print(f'  N={row["N"]:2d}, seed={row["seed"]:2d}: {row["scp_reason"]}')
        
        if sci_failures > 0:
            print('\nContactSolver failures:')
            sci_fail_df = df[~df['contact_success']][['N', 'seed', 'contact_reason']]
            for _, row in sci_fail_df.iterrows():
                print(f'  N={row["N"]:2d}, seed={row["seed"]:2d}: {row["contact_reason"]}')
        
        print('=' * 70)


if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else '0_sweep_n/run_K200/sweep_n_results.csv'
    analyze_failure_rates(csv_path)
