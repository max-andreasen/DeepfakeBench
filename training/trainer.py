import json
from datetime import datetime, timezone

import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from copy import deepcopy

"""
Will replace the train_linearcls.py and train_transformer.py and
create a unified trainer class that can be applicable for all
models.
"""

# TODO: Implement the trainer class
# TODO: Add support for biGRU model.

class Trainer:
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        logger,
        ):
        if config is None or model is None or optimizer is None or logger is None:
            raise NotImplementedError("config, model, optimizer, and logger must be provided")

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    # Does not really need checkpointing rigt now.
    def load_ckpt(self, model_path):
        """
        Load the model checkpoint from the given path.
        """
        pass

    def save_ckpt(self, path):
        """
        Save the model state_dict to the given path.
        """
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Saved model weights to {path}")

    def save_run_config(self, path, extra=None):
        """
        Write a curated run_config.json capturing what's needed to reproduce /
        interpret this training run (model kwargs, optimizer, scheduler, data paths, etc.).
        The full YAML is also in training.log; this is the machine-readable summary.
        Pass extra={...} to merge additional fields (e.g. epochs_completed, best_val_auroc).
        """
        cfg = self.config
        model_type = cfg.get('model_type')
        run_config = {
            'saved_utc': datetime.now(timezone.utc).isoformat(),
            'model_type': model_type,
            'model_kwargs': cfg.get('model', {}).get(model_type, {}),
            'optimizer': {
                'type': cfg.get('optimizer', {}).get('type'),
                'params': cfg.get('optimizer', {}).get(cfg.get('optimizer', {}).get('type'), {}),
            },
            'lr_scheduler': cfg.get('lr_scheduler'),
            'num_epochs': cfg.get('num_epochs'),
            'warmup_epochs': cfg.get('warmup_epochs'),
            'batch_size': cfg.get('batchSize'),
            'num_frames': cfg.get('num_frames'),
            'seed': cfg.get('seed'),
            'metric_scoring': cfg.get('metric_scoring'),
            'use_data_augmentation': cfg.get('use_data_augmentation', False),
            'data_aug': cfg.get('data_aug') if cfg.get('use_data_augmentation') else None,
            'input_transform': cfg.get('input_transform', 'none'),
            'data': {
                'root_dir': cfg.get('root_dir', ''),
                'split_file': cfg.get('split_file'),
                'catalogue_file': cfg.get('catalogue_file'),
                'train_dataset': cfg.get('train_dataset'),
                'val_dataset': cfg.get('val_dataset'),
                'compression': cfg.get('compression'),
            },
            'device': str(self.device),
        }
        if extra:
            run_config.update(extra)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(run_config, f, indent=2)
        self.logger.info(f"Saved run config to {path}")


    def train_step(self, x, labels):
        """
        Perform a single training step.
        Loads the optimizer and and computes the loss.
        x = the input batch
        labels = the ground truth labels
        """
        predictions = self.model(x)
        loss = F.cross_entropy(predictions, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, predictions


    def train_epoch(
        self,
        epoch,
        train_dataloader,
        val_dataloader
        ):
        self.setTrain()

        """
        Train the model for one epoch.
        """

        # Loops through every batch in training data
        total_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", unit="batch", leave=False)
        for batch_idx, (x, labels) in enumerate(pbar):
            x = x.to(self.device)
            labels = labels.to(self.device)

            loss, predictions = self.train_step(x, labels)
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        self.logger.info(f"Epoch [{epoch}]: Loss: {avg_loss}")

        # Evaluate on validation set if provided
        if val_dataloader is not None:
            return self.eval_epoch(epoch, val_dataloader)


    @torch.no_grad()
    def eval_epoch(self, epoch, val_dataloader):
        """
        Evaluate the model on the validation set for one epoch.
        """
        self.setEval()
        all_preds = []
        all_labels = []

        for x, labels in val_dataloader:
            x = x.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(x)
            all_preds.append(logits)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Compute metrics
        self.logger.info(f"--- Validation ---")
        acc = (all_preds.argmax(dim=1) == all_labels).float().mean().item()
        self.logger.info(f"Epoch [{epoch}]: Accuracy: {acc}")

        probs = torch.softmax(all_preds, dim=1)[:, 1].cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        auroc = roc_auc_score(labels_np, probs)
        self.logger.info(f"Epoch [{epoch}]: AUROC: {auroc}")

        # Accuracy at the threshold that maximizes Youden's J (tpr - fpr).
        # Reported alongside the fixed-0.5 acc above, which can be misleading under class imbalance.
        fpr, tpr, thresholds = roc_curve(labels_np, probs)
        best_thresh = thresholds[(tpr - fpr).argmax()]
        acc_at_best = ((probs >= best_thresh) == labels_np).mean()
        self.logger.info(f"Epoch [{epoch}]: Accuracy@best_thresh={best_thresh:.3f}: {acc_at_best:.4f}")

        return auroc

    def save_metrics(self, ):
        pass

    def inference(self, data_dict):
        pass
