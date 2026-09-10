#!/usr/bin/env python3
"""Analyze cost ratios (ContactSolver / LiftedSCP) from sweep results."""

import pandas as pd
import numpy as np

def analyze_cost_ratios(csv_path):
    df = pd.read_csv(csv_path)
    
    # Filter to successful cases only
    success_df = df[(df['scp_success']) & (df['contact_success'])].copy()
    
    # Compute cost ratio
    success_df['cost_ratio'] = success_df['contact_cost'] / success_df['scp_cost']
    
    print('=' * 80)
    print('MEDIAN COST RATIOS BY NUMBER OF ROBOTS (ContactSolver / LiftedSCP)')
    print('=' * 80)
    print(f'{"N":>3s} {"Cases":>6s} {"Median Ratio":>18s} {"Mean Ratio":>18s} {"Std Dev":>15s}')
    print('-' * 80)
    
    for n in sorted(success_df['N'].unique()):
        n_df = success_df[success_df['N'] == n]
        median_ratio = n_df['cost_ratio'].median()
        mean_ratio = n_df['cost_ratio'].mean()
        std_ratio = n_df['cost_ratio'].std()
        cases = len(n_df)
        
        print(f'{n:3d} {cases:6d} {median_ratio:18.12f} {mean_ratio:18.12f} {std_ratio:15.10f}')
    
    # Overall statistics
    overall_median = success_df['cost_ratio'].median()
    overall_mean = success_df['cost_ratio'].mean()
    overall_std = success_df['cost_ratio'].std()
    total_cases = len(success_df)
    
    print('-' * 80)
    print(f'ALL {total_cases:6d} {overall_median:18.12f} {overall_mean:18.12f} {overall_std:15.10f}')
    print('=' * 80)
    
    # Additional statistics
    print()
    print('Additional statistics:')
    print(f'  Min ratio:  {success_df["cost_ratio"].min():.12f}')
    print(f'  Max ratio:  {success_df["cost_ratio"].max():.12f}')
    print(f'  25th pctile: {success_df["cost_ratio"].quantile(0.25):.12f}')
    print(f'  75th pctile: {success_df["cost_ratio"].quantile(0.75):.12f}')
    print()


if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else '0_sweep_n/run_K200/sweep_n_results.csv'
    analyze_cost_ratios(csv_path)
