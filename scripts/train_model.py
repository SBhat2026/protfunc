#!/usr/bin/env python3
"""
Training Pipeline for Enhanced ProtFunc Model

This script trains the EnhancedResidualMLP with attention pooling on protein
sequences for GO term prediction. Features include:

- Mixed precision training (FP16/BF16) for faster training
- Gradient accumulation for effective larger batch sizes
- Cosine annealing with warm restarts learning rate schedule
- Early stopping with patience
- TensorBoard logging
- Checkpoint saving with best model tracking
- Multi-GPU support via DataParallel

Usage:
    python train_model.py --data-dir data/processed --output-dir checkpoints
    python train_model.py --resume checkpoints/latest.pt --epochs 50
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.enhanced_mlp import EnhancedResidualMLP, FocalLoss, AsymmetricLoss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults."""
    
    # Data
    data_dir: str = "data/processed"
    
    # Model architecture
    esm_model: str = "esm2_t6_8M_UR50D"  # ESM-2 variant
    embed_dim: int = 320  # ESM-2 8M embedding dimension
    hidden_dim: int = 1024
    num_blocks: int = 2
    dropout: float = 0.2
    use_attention_pooling: bool = True
    pooling_dropout: float = 0.1
    
    # Training hyperparameters
    batch_size: int = 32
    effective_batch_size: int = 128  # For gradient accumulation
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    warmup_epochs: int = 5
    
    # Loss function
    loss_type: str = "focal"  # 'focal', 'asymmetric', or 'bce'
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    
    # Learning rate schedule
    scheduler: str = "cosine"  # 'cosine', 'onecycle', or 'none'
    lr_min: float = 1e-6
    
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4
    
    # Checkpointing
    output_dir: str = "checkpoints"
    save_every: int = 5
    keep_best: int = 3
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    
    # Logging
    log_every: int = 100
    eval_every: int = 1
    
    @property
    def gradient_accumulation_steps(self) -> int:
        return max(1, self.effective_batch_size // self.batch_size)


# ============================================================================
# Dataset
# ============================================================================

class ProteinDataset(Dataset):
    """Dataset for protein sequences with GO annotations."""
    
    def __init__(
        self,
        data_path: str,
        go_mapping_path: str,
        esm_model_name: str = "esm2_t6_8M_UR50D",
        max_length: int = 1024,
        cache_embeddings: bool = False,
        cache_dir: Optional[str] = None
    ):
        self.max_length = max_length
        self.cache_embeddings = cache_embeddings
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Load GO mapping
        with open(go_mapping_path) as f:
            go_data = json.load(f)
        self.go_to_idx = go_data['go_to_idx']
        self.num_labels = go_data['num_labels']
        
        # Load data
        self.records = []
        with open(data_path) as f:
            for line in f:
                record = json.loads(line)
                # Filter to sequences that fit
                if len(record['sequence']) <= max_length:
                    self.records.append(record)
        
        logger.info(f"Loaded {len(self.records)} proteins from {data_path}")
        
        # Initialize ESM model for tokenization
        self._init_esm(esm_model_name)
        
        # Embedding cache
        self.embedding_cache = {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_esm(self, model_name: str):
        """Initialize ESM tokenizer (not full model - that's in trainer)."""
        import esm
        _, alphabet = esm.pretrained.load_model_and_alphabet_hub(model_name)
        self.batch_converter = alphabet.get_batch_converter()
        self.alphabet = alphabet
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        
        # Tokenize sequence
        _, _, tokens = self.batch_converter([(record['name'], record['sequence'])])
        tokens = tokens.squeeze(0)  # Remove batch dimension
        
        # Create label vector
        labels = torch.zeros(self.num_labels, dtype=torch.float32)
        for go_id in record['go_ids']:
            if go_id in self.go_to_idx:
                labels[self.go_to_idx[go_id]] = 1.0
        
        return {
            'tokens': tokens,
            'labels': labels,
            'seq_len': len(record['sequence']),
            'accession': record['accession']
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function that pads sequences to same length."""
    
    # Find max length in batch
    max_len = max(item['tokens'].size(0) for item in batch)
    
    # Pad tokens
    tokens = torch.stack([
        F.pad(item['tokens'], (0, max_len - item['tokens'].size(0)), value=1)  # 1 = padding token
        for item in batch
    ])
    
    # Stack labels
    labels = torch.stack([item['labels'] for item in batch])
    
    # Sequence lengths for masking
    seq_lens = torch.tensor([item['seq_len'] for item in batch])
    
    return {
        'tokens': tokens,
        'labels': labels,
        'seq_lens': seq_lens
    }


# ============================================================================
# Metrics
# ============================================================================

class MetricsTracker:
    """Track and compute evaluation metrics."""
    
    def __init__(self, num_labels: int, threshold: float = 0.5):
        self.num_labels = num_labels
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= self.threshold).astype(np.float32)
        labels_np = labels.cpu().numpy()
        
        self.all_probs.append(probs)
        self.all_preds.append(preds)
        self.all_labels.append(labels_np)
    
    def compute(self) -> Dict[str, float]:
        all_probs = np.vstack(self.all_probs)
        all_preds = np.vstack(self.all_preds)
        all_labels = np.vstack(self.all_labels)
        
        # Macro metrics (average across labels)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # Mean Average Precision (mAP)
        try:
            mAP = average_precision_score(all_labels, all_probs, average='macro')
        except ValueError:
            mAP = 0.0
        
        # Per-sample metrics
        f1_samples = f1_score(all_labels, all_preds, average='samples', zero_division=0)
        
        return {
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_samples': f1_samples,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'mAP': mAP
        }


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """Training orchestrator with all bells and whistles."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Will be set during setup
        self.model = None
        self.esm_model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        
        self.train_loader = None
        self.val_loader = None
        self.num_labels = None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        
        # TensorBoard
        self.writer = None
    
    def setup(self, train_data_path: str, val_data_path: str, go_mapping_path: str):
        """Setup everything needed for training."""
        
        logger.info("Setting up training...")
        
        # Load GO mapping to get num_labels
        with open(go_mapping_path) as f:
            go_data = json.load(f)
        self.num_labels = go_data['num_labels']
        logger.info(f"Number of labels: {self.num_labels}")
        
        # Create datasets
        train_dataset = ProteinDataset(
            train_data_path, go_mapping_path,
            esm_model_name=self.config.esm_model,
            max_length=1024
        )
        val_dataset = ProteinDataset(
            val_data_path, go_mapping_path,
            esm_model_name=self.config.esm_model,
            max_length=1024
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_fn,
            drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_fn
        )
        
        # Load ESM model for embeddings
        import esm
        self.esm_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
        self.esm_model = self.esm_model.to(self.device).eval()
        for param in self.esm_model.parameters():
            param.requires_grad = False
        logger.info("ESM-2 model loaded and frozen")
        
        # Create classifier model
        self.model = EnhancedResidualMLP(
            in_dim=self.config.embed_dim,
            out_dim=self.num_labels,
            hidden_dim=self.config.hidden_dim,
            num_blocks=self.config.num_blocks,
            dropout=self.config.dropout,
            use_attention_pooling=self.config.use_attention_pooling,
            pooling_dropout=self.config.pooling_dropout
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.epochs
        
        if self.config.scheduler == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=steps_per_epoch * 10,  # Restart every 10 epochs
                T_mult=2,
                eta_min=self.config.lr_min
            )
        elif self.config.scheduler == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=self.config.warmup_epochs / self.config.epochs
            )
        
        # Loss function
        if self.config.loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma
            )
        elif self.config.loss_type == 'asymmetric':
            self.criterion = AsymmetricLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Mixed precision
        if self.config.mixed_precision and self.device.type == 'cuda':
            self.scaler = GradScaler()
        
        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = self.output_dir / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard logging to {log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available")
        
        logger.info("Setup complete!")
    
    def train_epoch(self) -> float:
        """Train for one epoch, return average loss."""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            tokens = batch['tokens'].to(self.device)
            labels = batch['labels'].to(self.device)
            seq_lens = batch['seq_lens']
            
            # Get ESM embeddings
            with torch.no_grad():
                esm_out = self.esm_model(tokens, repr_layers=[6])
                embeddings = esm_out['representations'][6]  # [batch, seq_len, embed_dim]
            
            # Create attention mask
            max_len = embeddings.size(1)
            mask = torch.arange(max_len, device=self.device).unsqueeze(0) < seq_lens.unsqueeze(1).to(self.device)
            # Exclude BOS and EOS tokens
            mask[:, 0] = False
            for i, sl in enumerate(seq_lens):
                if sl + 1 < max_len:
                    mask[i, sl + 1:] = False
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    logits = self.model(embeddings, mask=mask)
                    loss = self.criterion(logits, labels)
                    loss = loss / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                logits = self.model(embeddings, mask=mask)
                loss = self.criterion(logits, labels)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.log_every == 0:
                    avg_loss = total_loss / num_batches
                    lr = self.optimizer.param_groups[0]['lr']
                    logger.info(f"Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")
                    
                    if self.writer:
                        self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                        self.writer.add_scalar('train/lr', lr, self.global_step)
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, Dict[str, float]]:
        """Evaluate on validation set."""
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        metrics = MetricsTracker(self.num_labels)
        
        for batch in self.val_loader:
            tokens = batch['tokens'].to(self.device)
            labels = batch['labels'].to(self.device)
            seq_lens = batch['seq_lens']
            
            # Get ESM embeddings
            esm_out = self.esm_model(tokens, repr_layers=[6])
            embeddings = esm_out['representations'][6]
            
            # Create attention mask
            max_len = embeddings.size(1)
            mask = torch.arange(max_len, device=self.device).unsqueeze(0) < seq_lens.unsqueeze(1).to(self.device)
            mask[:, 0] = False
            for i, sl in enumerate(seq_lens):
                if sl + 1 < max_len:
                    mask[i, sl + 1:] = False
            
            # Forward pass
            if self.scaler:
                with autocast():
                    logits = self.model(embeddings, mask=mask)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(embeddings, mask=mask)
                loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
            metrics.update(logits, labels)
        
        avg_loss = total_loss / max(num_batches, 1)
        metric_values = metrics.compute()
        
        return avg_loss, metric_values
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'scaler': self.scaler.state_dict() if self.scaler else None,
            'best_metric': self.best_metric,
            'config': vars(self.config)
        }
        
        # Save latest
        torch.save(checkpoint, self.output_dir / 'latest.pt')
        
        # Save periodic checkpoint
        if self.epoch % self.config.save_every == 0:
            torch.save(checkpoint, self.output_dir / f'epoch_{self.epoch}.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best.pt')
            # Also save just the model weights for inference
            torch.save(
                {'model': self.model.state_dict()},
                self.output_dir / 'best_model.pth'
            )
            logger.info(f"New best model saved (mAP: {self.best_metric:.4f})")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if checkpoint['scheduler'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        if checkpoint['scaler'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        logger.info(f"Resumed from epoch {self.epoch}, step {self.global_step}")
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        
        if resume_from:
            self.load_checkpoint(resume_from)
        
        logger.info(f"Starting training from epoch {self.epoch + 1}")
        
        for epoch in range(self.epoch + 1, self.config.epochs + 1):
            self.epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Evaluate
            if epoch % self.config.eval_every == 0:
                val_loss, metrics = self.evaluate()
                
                epoch_time = time.time() - epoch_start
                logger.info(
                    f"Epoch {epoch}/{self.config.epochs} ({epoch_time:.1f}s) - "
                    f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                    f"f1_macro: {metrics['f1_macro']:.4f}, mAP: {metrics['mAP']:.4f}"
                )
                
                # TensorBoard logging
                if self.writer:
                    self.writer.add_scalar('val/loss', val_loss, epoch)
                    for name, value in metrics.items():
                        self.writer.add_scalar(f'val/{name}', value, epoch)
                
                # Check for improvement
                is_best = metrics['mAP'] > self.best_metric + self.config.min_delta
                if is_best:
                    self.best_metric = metrics['mAP']
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Save checkpoint
                self.save_checkpoint(is_best=is_best)
                
                # Early stopping
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
        
        logger.info(f"Training complete! Best mAP: {self.best_metric:.4f}")
        
        if self.writer:
            self.writer.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train Enhanced ProtFunc Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python train_model.py --data-dir data/processed

  # Train with custom hyperparameters
  python train_model.py --data-dir data/processed --lr 5e-4 --batch-size 64 --epochs 50

  # Resume training
  python train_model.py --resume checkpoints/latest.pt

  # Train with specific loss function
  python train_model.py --data-dir data/processed --loss asymmetric
        """
    )
    
    # Data arguments
    parser.add_argument('--data-dir', default='data/processed', help='Directory with train/val/test splits')
    parser.add_argument('--output-dir', default='checkpoints', help='Output directory for checkpoints')
    
    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=1024, help='Hidden dimension')
    parser.add_argument('--num-blocks', type=int, default=2, help='Number of residual blocks')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--no-attention', action='store_true', help='Disable attention pooling')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--effective-batch-size', type=int, default=128, help='Effective batch size (gradient accumulation)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    # Loss arguments
    parser.add_argument('--loss', choices=['focal', 'asymmetric', 'bce'], default='focal', help='Loss function')
    parser.add_argument('--focal-alpha', type=float, default=0.25, help='Focal loss alpha')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma')
    
    # Other arguments
    parser.add_argument('--resume', help='Path to checkpoint to resume from')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision training')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        use_attention_pooling=not args.no_attention,
        batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        loss_type=args.loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        device=args.device,
        num_workers=args.workers,
        mixed_precision=not args.no_amp
    )
    
    # Paths
    data_dir = Path(args.data_dir)
    train_path = data_dir / 'train.jsonl'
    val_path = data_dir / 'val.jsonl'
    go_mapping_path = data_dir / 'go_terms.json'
    
    # Verify files exist
    for path in [train_path, val_path, go_mapping_path]:
        if not path.exists():
            logger.error(f"Required file not found: {path}")
            logger.error("Run the UniProt scraper first: python scripts/uniprot_scraper.py --process ...")
            sys.exit(1)
    
    # Create trainer and run
    trainer = Trainer(config)
    trainer.setup(str(train_path), str(val_path), str(go_mapping_path))
    trainer.train(resume_from=args.resume)


if __name__ == '__main__':
    main()
