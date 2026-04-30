"""
Power analysis and pairwise Welch t-tests for frozen-backbone retrains.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
from scipy import stats


def load_aucs(results):
    aucs = {}
    for model, path in results.items():
        row = pd.read_csv(path).iloc[0]
        seed_cols = [c for c in row.index if c.startswith("seed_") and c.endswith("_auc")]
        aucs[model] = row[seed_cols].astype(float).to_numpy()
    return aucs


def welch_power(epsilon, std_a, std_b, n, alpha):
    var_a = std_a**2 / n
    var_b = std_b**2 / n
    df = (var_a + var_b) ** 2 / ((var_a**2 / (n - 1)) + (var_b**2 / (n - 1)))
    se = np.sqrt(var_a + var_b)
    critical = stats.t.ppf(1 - alpha / 2, df)
    noncentrality = epsilon / se
    return stats.nct.cdf(-critical, df, noncentrality) + 1 - stats.nct.cdf(
        critical, df, noncentrality
    )


def power_analysis(aucs, pairs, epsilon, alpha, target_power):
    rows = []
    for a, b in pairs:
        std_a = np.std(aucs[a], ddof=1)
        std_b = np.std(aucs[b], ddof=1)
        n = 2
        power = welch_power(epsilon, std_a, std_b, n, alpha)
        while power < target_power:
            n += 1
            power = welch_power(epsilon, std_a, std_b, n, alpha)
        # creates a nice dataframe to use as output
        rows.append(
            {
                "pair": f"{a} vs {b}",
                "epsilon": epsilon,
                "alpha": alpha,
                "target_power": target_power,
                "std_a": std_a,
                "std_b": std_b,
                "required_n": n,
                "achieved_power": power,
            }
        )
    return pd.DataFrame(rows)


def run_welch_ttest(aucs, pairs, alpha):
    rows = []
    for a, b in pairs:
        res = pg.ttest(aucs[a], aucs[b], correction=True).iloc[0]
        p_raw = float(res["p-val"])
        rows.append(
            {
                "pair": f"{a} vs {b}",
                "mean_a": np.mean(aucs[a]),
                "mean_b": np.mean(aucs[b]),
                "diff_a_minus_b": np.mean(aucs[a]) - np.mean(aucs[b]),
                "t": float(res["T"]),
                "dof": float(res["dof"]),
                "p_raw": p_raw,
                "p_bonferroni": min(p_raw * len(pairs), 1.0),
                "alpha_bonferroni": alpha / len(pairs),
                "ci95": res["CI95%"],
                "power": float(res["power"]),
            }
        )
    return pd.DataFrame(rows)


def plot_results(aucs, results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(results)
    data = [aucs[m] for m in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, labels=labels, showmeans=True)
    for i, vals in enumerate(data, start=1):
        ax.scatter(np.full(len(vals), i), vals, color="black", s=24, zorder=3)
    ax.set_ylabel("CDFv2-clean test AUC")
    ax.set_title("Frozen-backbone repeated-seed AUCs")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "frozen_auc_boxplot.png", dpi=200)
    plt.close(fig)


def main():

    results = {
        "Transformer": "evaluation/results/retrain_welch/transformer/results.csv",
        "BiGRU": "evaluation/results/retrain_welch/bigru/results.csv",
        "Linear": "evaluation/results/retrain_welch/linear/results.csv",
    }

    pairs = [
        ("Transformer", "BiGRU"),
        ("Transformer", "Linear"),
        ("BiGRU", "Linear"),
    ]

    out_dir = Path("evaluation/results/statistical_analysis")
    epsilon = 0.01
    alpha = 0.05
    target_power = 0.80

    aucs = load_aucs(results)

    print("=== Final repeated-seed samples ===")
    for model, vals in aucs.items():
        print(f"{model:12s} N={len(vals)} mean={np.mean(vals):.4f} std={np.std(vals, ddof=1):.4f}")
    print()

    power_df = power_analysis(aucs, pairs, epsilon, alpha, target_power)
    welch_df = run_welch_ttest(aucs, pairs, alpha)

    out_dir.mkdir(parents=True, exist_ok=True)
    power_df.to_csv(out_dir / "power_analysis.csv", index=False)
    welch_df.to_csv(out_dir / "welch_tests.csv", index=False)
    plot_results(aucs, results, out_dir)

    print("=== Power analysis ===")
    print(power_df.to_string(index=False))
    print()
    print("=== Welch tests (Bonferroni across frozen pairs) ===")
    print(welch_df.to_string(index=False))


if __name__ == "__main__":
    main()
