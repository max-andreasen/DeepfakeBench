#!/usr/bin/env python3
import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPOCH_RE = re.compile(
    r"epoch\s+(\d+)/(\d+)\s+"
    r"loss=([0-9.eE+-]+)\s+"
    r"val_auc=([0-9.eE+-]+)"
    r"(?:\s+val_acc=([0-9.eE+-]+))?"
    r"(?:\s+val_acc@best=([0-9.eE+-]+))?"
    r"(?:\s+thr=([0-9.eE+-]+))?"
    r"(?:\s+lr=([0-9.eE+-]+))?"
)


def resolve_log_path(path):
    path = Path(path)
    if path.is_dir():
        path = path / "training.log"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def parse_log(path):
    rows = []
    for line in path.read_text().splitlines():
        match = EPOCH_RE.search(line)
        if not match:
            continue
        epoch, total, loss, val_auc, val_acc, val_acc_best, thr, lr = match.groups()
        rows.append(
            {
                "epoch": int(epoch),
                "total_epochs": int(total),
                "loss": float(loss),
                "val_auc": float(val_auc),
                "val_acc": float(val_acc) if val_acc else None,
                "val_acc_best": float(val_acc_best) if val_acc_best else None,
                "threshold": float(thr) if thr else None,
                "lr": float(lr) if lr else None,
            }
        )
    if not rows:
        raise ValueError(f"No epoch summary lines found in {path}")
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, out_path, title):
    epochs = [r["epoch"] for r in rows]
    losses = [r["loss"] for r in rows]
    aucs = [r["val_auc"] for r in rows]
    best_idx = max(range(len(rows)), key=lambda i: rows[i]["val_auc"])

    fig, ax_loss = plt.subplots(figsize=(9, 5))
    ax_auc = ax_loss.twinx()

    loss_line = ax_loss.plot(
        epochs, losses, marker="o", linewidth=2, color="#1f77b4", label="train loss"
    )
    auc_line = ax_auc.plot(
        epochs, aucs, marker="s", linewidth=2, color="#d62728", label="val AUC"
    )

    best_epoch = rows[best_idx]["epoch"]
    best_auc = rows[best_idx]["val_auc"]
    ax_auc.scatter([best_epoch], [best_auc], color="#d62728", s=70, zorder=5)
    ax_auc.annotate(
        f"best val AUC={best_auc:.4f}\nepoch {best_epoch}",
        xy=(best_epoch, best_auc),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=9,
        color="#d62728",
    )

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train loss", color="#1f77b4")
    ax_auc.set_ylabel("Validation AUC", color="#d62728")
    ax_loss.tick_params(axis="y", labelcolor="#1f77b4")
    ax_auc.tick_params(axis="y", labelcolor="#d62728")
    ax_loss.grid(True, axis="both", alpha=0.25)

    lines = loss_line + auc_line
    labels = [line.get_label() for line in lines]
    ax_loss.legend(lines, labels, loc="upper center", ncol=2, frameon=False)
    ax_loss.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot PEFT train loss and validation AUC from training.log."
    )
    parser.add_argument("log_or_run_dir", help="Path to training.log or a PEFT run directory.")
    parser.add_argument("--out", default=None, help="Output PNG path.")
    parser.add_argument("--csv", default=None, help="Optional CSV path for parsed epoch metrics.")
    args = parser.parse_args()

    log_path = resolve_log_path(args.log_or_run_dir)
    rows = parse_log(log_path)

    out_path = Path(args.out) if args.out else log_path.with_name("training_curve.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot(rows, out_path, title=log_path.parent.name)

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, csv_path)

    best = max(rows, key=lambda r: r["val_auc"])
    print(f"wrote {out_path}")
    print(f"best val_auc={best['val_auc']:.4f} at epoch {best['epoch']}")


if __name__ == "__main__":
    main()
