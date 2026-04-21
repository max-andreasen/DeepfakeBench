import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Shared model registry (repo root). So eval rebuilds the exact same class
# that train.py instantiated for a given model_type.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import MODELS  # type: ignore[import-not-found]

from data_loader import DeepfakeTestDataset
from tester import Tester
from logger import create_logger  # type: ignore[import-not-found]

"""
This script does following;
1) Loads the YAML config and parses CLI.
2) Builds the output directory (which tester.py also writes to).
3) Loads a trained model weights and rebuilds based on run_config.json (from training).
4) Loads the test data with a DataLoader (from CLIP embeddings that is).
5) Runs inference using tester.py, passing necessary arguments.
6) Logs some of the results with the logger during the process.
"""

# Used to confirm the yaml config is correct.
REQUIRED_FIELDS = [
    'output_dir',
    'trained_model_dir',
    'split_file',
    'catalogue_file',
    'test_dataset',
    'batch_size',
    'num_frames',
    'device',
    'seed',
]

# CLI args.
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained temporal-classifier checkpoint.')
    parser.add_argument('--config', type=str, required=True,
                        help='path to evaluation YAML config')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='output subfolder name under evaluation/results/<trained_dir>/. '
                             'Defaults to the config filename stem.')
    parser.add_argument('--shuffle_frames', dest='shuffle_frames', action='store_true', default=None,
                        help='CLI override: enable temporal-shuffling ablation. '
                             'If unset, falls back to yaml field `shuffle_frames`.')
    args = parser.parse_args()

    if args.run_tag is None:
        args.run_tag = Path(args.config).stem
    return args


# The yaml config.
def load_config(config_path, cli_shuffle_frames):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    missing = [k for k in REQUIRED_FIELDS if k not in config]
    if missing:
        raise ValueError(f"Eval config {config_path} missing required fields: {missing}")

    # CLI wins over yaml for shuffle_frames; yaml default is False if unset.
    if cli_shuffle_frames is not None:
        config['shuffle_frames'] = cli_shuffle_frames
    else:
        config.setdefault('shuffle_frames', False)

    config.setdefault('window_aggregation', 'mean')
    config.setdefault('workers', 4)
    config.setdefault('root_dir', '')
    return config


def load_trained_model(trained_model_dir, root_dir=''):
    """Loads the model from the run_config.json (so that we know the correct parameters)."""

    trained_dir = Path(root_dir) / trained_model_dir if root_dir else Path(trained_model_dir)
    run_config_path = trained_dir / 'run_config.json'
    model_path = trained_dir / 'model.pth'

    if not run_config_path.is_file():
        raise FileNotFoundError(f"Missing run_config.json at {run_config_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model.pth at {model_path}")

    with open(run_config_path, 'r') as f:
        run_config = json.load(f)

    model_type = run_config['model_type']
    model_kwargs = run_config.get('model_kwargs', {})
    if model_type not in MODELS:
        raise ValueError(f"Unknown model_type '{model_type}'; not in MODELS: {list(MODELS)}")

    model = MODELS[model_type](**model_kwargs)
    state = torch.load(str(model_path), map_location='cpu')
    model.load_state_dict(state)
    return model, run_config


def build_test_loader(config, input_transform='none'):
    """Builds the data loader for the test set, preparing the data for inference.
    input_transform must match what the model was trained with (read from
    run_config.json); otherwise a diff-trained model gets raw embeddings at
    test time and silently collapses."""

    root = config.get('root_dir', '')
    dataset = DeepfakeTestDataset(
        split_file=os.path.join(root, config['split_file']),
        catalogue_file=os.path.join(root, config['catalogue_file']),
        num_frames=config['num_frames'],
        input_transform=input_transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=int(config['workers']),
    )
    return loader


def setup_output(config, run_tag):
    """Build the eval output dir and a file logger inside it.
    Layout: <output_dir>/<basename(trained_model_dir)>/<run_tag>/test.log
    """
    out_dir = Path(config['output_dir']) / Path(config['trained_model_dir']).name / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(str(out_dir / 'test.log'))
    return out_dir, logger


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config, args.shuffle_frames)

    out_dir, logger = setup_output(config, args.run_tag)
    logger.info(f"CLI args: {vars(args)}")
    logger.info(f"Resolved config: {config}")
    logger.info(f"Output dir: {out_dir}")

    model, run_config = load_trained_model(config['trained_model_dir'], config.get('root_dir', ''))
    logger.info(f"Loaded model_type={run_config['model_type']} "
                f"kwargs={run_config.get('model_kwargs', {})}")
    logger.info(f"Trained on: {run_config.get('data', {}).get('train_dataset')}")
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # Pull input_transform from the trained-model metadata so the model sees
    # the same kind of input it was trained on (raw vs diff).
    trained_transform = run_config.get('input_transform', 'none')
    logger.info(f"input_transform (from run_config): {trained_transform}")
    test_loader = build_test_loader(config, input_transform=trained_transform)
    test_dataset = test_loader.dataset
    assert isinstance(test_dataset, DeepfakeTestDataset)
    logger.info(f"Total batches: {len(test_loader)}  (test videos: {len(test_dataset)})")

    tester = Tester(config, model, logger)

    standard = tester.evaluate(test_loader, shuffle_frames=False)
    torch.manual_seed(config['seed'])  # reproducible shuffled pass
    shuffled = tester.evaluate(test_loader, shuffle_frames=True)

    tester.save_results(out_dir, standard, shuffled, eval_config_dict=config)
