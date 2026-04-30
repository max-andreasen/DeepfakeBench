"""
Pilot power analysis and pairwise Welch t-tests on CDFv2-clean test AUCs.

N = 5 seeds per model (pilot). Uses pingouin.ttest which applies the
non-central t-distribution for power (Colas et al. 2018 methodology).

Usage:
    python statistical_analysis.py
"""

import pandas as pd
import pingouin as pg

# ── Data ──────────────────────────────────────────────────────────────────────

RESULTS = {
    "Transformer": "evaluation/results/retrain_top_k/transformer_search3_10/results.csv",
    "BiGRU":       "evaluation/results/retrain_top_k/bigru_search2/results.csv",
    "Linear":      "evaluation/results/retrain_top_k/linear_search3_1/results.csv",
}

SEED_COLS = [f"seed_{i}_auc" for i in range(5)]


def load_aucs(path: str) -> list[float]:
    row = pd.read_csv(path).iloc[0]
    return [row[c] for c in SEED_COLS]


aucs = {name: load_aucs(path) for name, path in RESULTS.items()}

# ── Summary ───────────────────────────────────────────────────────────────────

print("=== Pilot AUC samples (N=5, CDFv2-clean test) ===")
for name, vals in aucs.items():
    mean = sum(vals) / len(vals)
    print(f"  {name:12s}  {[round(v, 4) for v in vals]}  mean={mean:.4f}")
print()

# ── Pairwise Welch t-tests ────────────────────────────────────────────────────

PAIRS = [
    ("Transformer", "BiGRU"),
    ("Transformer", "Linear"),
    ("BiGRU",       "Linear"),
]

print("=== Pairwise Welch t-tests (two-tailed, correction=True) ===")
print(f"{'Pair':<26}  {'T':>7}  {'dof':>5}  {'p-val':>8}  {'cohen-d':>8}  {'power':>7}")
print("-" * 70)

for a, b in PAIRS:
    res = pg.ttest(aucs[a], aucs[b], correction=True)
    T      = res["T"].values[0]
    dof    = res["dof"].values[0]
    p      = res["p-val"].values[0]
    d      = res["cohen-d"].values[0]
    power  = res["power"].values[0]
    label  = f"{a} vs {b}"
    print(f"{label:<26}  {T:>7.3f}  {dof:>5.2f}  {p:>8.4f}  {d:>8.4f}  {power:>7.4f}")

print()
print("Note: power is estimated from the pilot N=5. Use these values to derive")
print("the target N for the final repeated-seed study.")
