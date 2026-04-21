# Training script for temporal deepfake detection on CLIP embeddings.
# Adapted from DeepfakeBench (Zhiyuan Yan).

import os
import argparse
import random
import datetime
import yaml
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, StepLR

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_loader import DeepfakeDataset
from trainer import Trainer
from models import MODELS
from logger import create_logger



def init_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# split = 'train' or 'val'
def prepare_data(config, split):
    root = config.get('root_dir', '')
    dataset = DeepfakeDataset(
        split_file=os.path.join(root, config['split_file']),
        catalogue_file=os.path.join(root, config['catalogue_file']),
        split=split,
        num_frames=config['num_frames'][split],
        input_transform=config.get('input_transform', 'none'),
    )
    return DataLoader(
        dataset,
        batch_size=config['batchSize'][split],
        shuffle=(split == 'train'),    # shuffles only for train split
        num_workers=int(config['workers']),
    )


def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    opt_cfg = config['optimizer'][opt_name]
    if opt_name == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=opt_cfg['lr'],
            momentum=opt_cfg['momentum'],
            weight_decay=opt_cfg['weight_decay'],
        )
    elif opt_name == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=opt_cfg['lr'],
            weight_decay=opt_cfg['weight_decay'],
            betas=(opt_cfg['beta1'], opt_cfg['beta2']),
            eps=opt_cfg['eps'],
            amsgrad=opt_cfg.get('amsgrad', False),
        )
    elif opt_name == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=opt_cfg['lr'],
            weight_decay=opt_cfg['weight_decay'],
            betas=(opt_cfg['beta1'], opt_cfg['beta2']),
            eps=opt_cfg['eps'],
        )
    else:
        raise NotImplementedError(f'Optimizer {opt_name} is not implemented')



def choose_scheduler(config, optimizer):
    sched = config['lr_scheduler']
    if sched is None or sched == 'constant':
        return None
    elif sched == 'step':
        return StepLR(optimizer, step_size=config['step_size'], gamma=config['step_gamma'])
    elif sched == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=0)
    elif sched == 'cosine_warmup':
        warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=config['warmup_epochs'])
        cosine = CosineAnnealingLR(optimizer, T_max=config['num_epochs'] - config['warmup_epochs'])
        return SequentialLR(optimizer, [warmup, cosine], milestones=[config['warmup_epochs']])
    else:
        raise NotImplementedError(f'Scheduler {sched} is not implemented')


# NOTE: Save this, modify so can output multiple metrics
def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def build_log_path(config, task_target=""):
    """Timestamped log dir under config['log_dir'].
    Kept as a helper so both main() and Optuna callers can reuse the convention,
    or bypass it by passing an explicit log_path to train_from_config.
    """
    timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    task_str = f"_{task_target}" if task_target else ""
    return os.path.join(config['log_dir'], config['model_type'] + task_str + '_' + timenow)


def train_from_config(config, trial=None, log_path=None):
    """Run a full training loop from a merged config dict.

    Args:
        config:   merged config dict (not a YAML path).
        trial:    optional optuna.Trial. If provided, per-epoch val AUROC is
                  reported via trial.report() and optuna.TrialPruned is raised
                  when trial.should_prune() is True.
        log_path: explicit output directory. If None, a timestamped one is
                  built under config['log_dir'] (the legacy behavior).

    Returns dict with keys:
        'model':            trained nn.Module (moved to CPU so GPU memory can
                            be freed between trials),
        'log_path':         output directory (str),
        'best_val_auroc':   float, best per-epoch val AUROC seen,
        'final_val_auroc':  float, val AUROC from last completed epoch,
        'epochs_completed': int, epochs that ran to completion.

    Raises:
        optuna.TrialPruned when the pruner decides the trial should stop.
    """
    init_seed(config['seed'])

    if log_path is None:
        log_path = build_log_path(config)
    os.makedirs(log_path, exist_ok=True)
    logger = create_logger(os.path.join(log_path, 'training.log'))
    model_type = config['model_type']
    embed_dim = config['model'][model_type].get('clip_embed_dim', '?')
    logger.info(f"CLIP embeddings: {config['catalogue_file']}  (dim={embed_dim})")
    logger.info(f"Data split: {config['split_file']}")
    logger.info(f"  train: {config['num_frames']['train']} frames/video, batch_size={config['batchSize']['train']}")
    logger.info(f"  val:   {config['num_frames']['val']} frames/video, batch_size={config['batchSize']['val']}")
    logger.info(f"  input_transform: {config.get('input_transform', 'none')}")
    logger.info(f"Config: {config}")

    train_loader = prepare_data(config, 'train')
    val_loader = prepare_data(config, 'val')

    model_class = MODELS[config['model_type']]
    model = model_class(**config['model'][config['model_type']])

    optimizer = choose_optimizer(model, config)
    scheduler = choose_scheduler(config, optimizer)
    _ = choose_metric(config)  # validates metric_scoring name

    trainer = Trainer(config, model, optimizer, scheduler, logger)

    best_val_auroc = 0.0
    final_val_auroc = 0.0
    epochs_completed = 0
    pruned = False

    try:
        for epoch in range(config['num_epochs']):
            auc = trainer.train_epoch(
                epoch=epoch,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
            )
            if scheduler is not None:
                scheduler.step()

            final_val_auroc = float(auc) if auc is not None else 0.0
            best_val_auroc = max(best_val_auroc, final_val_auroc)
            epochs_completed = epoch + 1

            if trial is not None:
                import optuna  # deferred so train.py doesn't hard-depend on optuna
                trial.report(final_val_auroc, epoch)
                if trial.should_prune():
                    pruned = True
                    raise optuna.TrialPruned()
    finally:
        # run_config.json captures what was attempted, pruned or not.
        trainer.save_run_config(os.path.join(log_path, 'run_config.json'))
        if not pruned and config.get('save_ckpt', True):
            trainer.save_ckpt(os.path.join(log_path, 'model.pth'))
        model.cpu()
        logger.info(
            "Training pruned at epoch {}.".format(epochs_completed) if pruned
            else "Training complete."
        )

    return {
        'model': model,
        'log_path': log_path,
        'best_val_auroc': best_val_auroc,
        'final_val_auroc': final_val_auroc,
        'epochs_completed': epochs_completed,
    }


def main():
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--config', type=str, default='',
                        help='path to training YAML config file')
    parser.add_argument("--train_dataset", nargs="+")
    parser.add_argument("--val_dataset", nargs="+")
    parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
    # To store feature embeddings. Good for PEFT tuning on CLIP. Save, but probably won't use too much.
    parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=False)
    parser.add_argument('--task_target', type=str, default="",
                        help='specify the target of current training task')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.val_dataset:
        config['val_dataset'] = args.val_dataset
    config['save_ckpt'] = args.save_ckpt

    log_path = build_log_path(config, task_target=args.task_target)
    train_from_config(config, log_path=log_path)


if __name__ == '__main__':
    main()
