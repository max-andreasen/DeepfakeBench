"""
pilot_study.py — estimate the number of seeds (N) needed for the Welch t-test.

WHAT THIS IS
------------
Before running the full retrain loop (Step 3), run a small pilot with a handful
of seeds (e.g. 5) for the top-1 model from each param search. From those pilot
runs you can estimate the variance of the test AUC distribution and compute the
minimum N that gives the Welch t-test sufficient statistical power.

STATISTICAL BACKGROUND
----------------------
The Welch t-test compares two independent groups (e.g. Transformer vs BiGRU)
with unequal variances. To determine the required sample size N per group:

    1. Pilot: run n_pilot seeds for each model, record test AUCs.
    2. Effect size: Cohen's d = (mean_A - mean_B) / pooled_std
       Where pooled_std = sqrt((std_A² + std_B²) / 2)
    3. Power analysis: given d, alpha=0.05, power=0.80, solve for N using
       statsmodels TTestIndPower (uses Welch approximation).

Typical result: if std ≈ 0.01 AUC and the expected difference is ~0.02,
d ≈ 2.0 → N ≈ 7. If the effect is smaller (d ≈ 0.5), N can be 50+.

USAGE (once you know what models to compare)
--------------------------------------------
    # TODO: fill in after retrain_top_k pilot runs are done
    python evaluation/pilot_study.py \\
        --pilot_dir_a  evaluation/results/retrain_top_k/transformer_search3 \\
        --pilot_dir_b  evaluation/results/retrain_top_k/bigru_search2_1 \\
        --alpha 0.05 \\
        --power 0.80

OUTPUT
------
    Prints recommended N and a summary table.
    Writes pilot_power_analysis.json to --out_dir (if specified).
"""

# TODO: implement once pilot runs are available.
#
# Rough implementation sketch:
#
# import numpy as np
# from scipy import stats
# from statsmodels.stats.power import TTestIndPower
#
# def load_pilot_aucs(retrain_dir, trial_num, benchmark_name):
#     """Read per-seed test AUCs from retrain_top_k output."""
#     trial_dir = Path(retrain_dir) / f'trial_{trial_num:04d}'
#     aucs = []
#     for seed_dir in sorted(trial_dir.glob('seed_*')):
#         result_file = seed_dir / 'benchmark_results' / benchmark_name / 'results.json'
#         if result_file.exists():
#             with open(result_file) as f:
#                 aucs.append(json.load(f)['per_video']['auc'])
#     return np.array(aucs)
#
# def cohens_d(a, b):
#     pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
#     return (np.mean(a) - np.mean(b)) / pooled_std
#
# def required_n(d, alpha=0.05, power=0.80):
#     analysis = TTestIndPower()
#     return analysis.solve_power(effect_size=abs(d), alpha=alpha, power=power,
#                                 alternative='two-sided')
#
# aucs_a = load_pilot_aucs(args.pilot_dir_a, trial_a, args.benchmark)
# aucs_b = load_pilot_aucs(args.pilot_dir_b, trial_b, args.benchmark)
# d = cohens_d(aucs_a, aucs_b)
# n = required_n(d)
# print(f"Cohen's d = {d:.3f} → recommended N = {int(np.ceil(n))} per group")
