"""
statistical_analysis.py — Welch t-test + plots for benchmark results.

Reads the summary.csv produced by benchmark_inference.py and performs
pairwise statistical comparisons between model configurations.

WHAT TO IMPLEMENT HERE
----------------------
1. Load summary.csv — columns: trial, seed, benchmark, test_auc, ...
2. Group by (trial, benchmark) to get the AUC distribution per config.
3. Welch t-test: scipy.stats.ttest_ind(a, b, equal_var=False)
4. Effect size: Cohen's d
5. Plots:
   - Box plot: per-config AUC distribution per benchmark
   - Bar chart: mean ± std per config per benchmark
   - (Optional) Scatter: val_auc_search vs mean test_auc to see if
     param-search rank correlates with cross-dataset rank

Usage (once implemented):
    python evaluation/statistical_analysis.py \\
        --inference_dir evaluation/results/benchmark_inference/FILL_IN \\
        --out_dir        evaluation/results/statistical_analysis/FILL_IN
"""

# TODO: implement once benchmark_inference.py has been run.
#
# Rough sketch:
#
# import pandas as pd
# import numpy as np
# from scipy import stats
# import matplotlib.pyplot as plt
#
# def welch_t_test(a, b):
#     t, p = stats.ttest_ind(a, b, equal_var=False)
#     pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
#     d = (np.mean(a) - np.mean(b)) / pooled_std
#     return {'t': t, 'p': p, 'cohens_d': d}
#
# df = pd.read_csv(inference_dir / 'summary.csv')
#
# # Per benchmark, compare each pair of trials
# for benchmark in df['benchmark'].unique():
#     sub = df[df['benchmark'] == benchmark]
#     for trial_a, trial_b in combinations(sub['trial'].unique(), 2):
#         aucs_a = sub[sub['trial'] == trial_a]['test_auc'].values
#         aucs_b = sub[sub['trial'] == trial_b]['test_auc'].values
#         result = welch_t_test(aucs_a, aucs_b)
#         print(f"{benchmark}: trial {trial_a} vs {trial_b} — "
#               f"p={result['p']:.4f}  d={result['cohens_d']:.3f}")
