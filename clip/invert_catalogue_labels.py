"""Flip the `label` column in catalogue CSVs (0 <-> 1).

Use when a catalogue was written with the opposite binary convention from the
training YAML's `label_dict`. For our FF++ catalogues, 'FF-real' is currently
label=1 (treated as the positive class), while the YAML convention is real=0 /
fake=1. Run this to align.

The `label_name` column is NOT touched — the string already names the class
correctly; only the int mapping is inverted.

Example:
    # flip and write alongside as <name>.inverted.csv (non-destructive default):
    python clip/invert_catalogue_labels.py \
        clip/embeddings/mtcnn/ViT-L-14-336-quickgelu_dim768/catalogue.csv

    # or flip all three FF++ catalogues in place (overwrites originals):
    python clip/invert_catalogue_labels.py \
        clip/embeddings/{dlib,mtcnn,raw}/ViT-L-14-336-quickgelu_dim768/catalogue.csv \
        --in-place

    # preview without writing:
    python clip/invert_catalogue_labels.py path/to/catalogue.csv --dry-run

WARNING: flipping is involutive — running twice restores the original state.
There's no "detect current convention" mode. Know which way you want before
running.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def summarize(df):
    """Crosstab of (label_cat or label_name) x label. Returns a formatted string."""
    group_col = 'label_cat' if 'label_cat' in df.columns else 'label_name'
    counts = df.groupby([group_col, 'label']).size().reset_index(name='count')
    return counts.to_string(index=False)


def invert_file(path: Path, in_place: bool, dry_run: bool) -> None:
    df = pd.read_csv(path)

    if 'label' not in df.columns:
        raise ValueError(f"{path}: no 'label' column")
    bad = set(df['label'].unique()) - {0, 1}
    if bad:
        raise ValueError(f"{path}: non-binary label values present: {sorted(bad)}")

    print(f"\n=== {path} ===")
    print("Before:")
    print(summarize(df))

    df['label'] = 1 - df['label']

    print("\nAfter:")
    print(summarize(df))

    if dry_run:
        print("(dry-run, not written)")
        return

    out_path = path if in_place else path.with_suffix('.inverted.csv')
    df.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Flip the `label` column (0 <-> 1) in one or more catalogue CSVs.")
    ap.add_argument('paths', nargs='+', type=Path,
                    help='Catalogue CSV path(s).')
    ap.add_argument('--in-place', action='store_true',
                    help='Overwrite the source file instead of writing <name>.inverted.csv.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print before/after summaries but do not write anything.')
    args = ap.parse_args()

    missing = [p for p in args.paths if not p.is_file()]
    if missing:
        print(f"ERROR: missing file(s): {missing}", file=sys.stderr)
        sys.exit(1)

    for p in args.paths:
        invert_file(p, in_place=args.in_place, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
