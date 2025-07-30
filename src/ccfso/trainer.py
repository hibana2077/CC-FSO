"""
Training Pipeline for Ultra-Fine-Grained Visual Categorization

This module implements:
- Complete training loop with curvature-aware optimization
- Theoretical convergence monitoring
- Comprehensive evaluation metrics
- Experiment tracking and visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import time
import random
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
try:
    import umap
except ImportError:
    umap = None
try:
    import wandb
except ImportError:
    wandb = None

from .models import create_model, get_model_info
from .losses import CurvatureAwareLoss, CombinedLoss
from .optimizers import create_optimizer, create_scheduler
from .geometry import RiemannianGeometry, compute_manifold_diameter
from .augmentations import create_augmentation_pipeline, CutMix


class MetricsTracker:
    """
    Track and compute various metrics for UFGVC evaluation
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.correct = 0
        self.total = 0
        self.class_correct = {}
        self.class_total = {}
        self.predictions = []
        self.targets = []
        self.features = []
        self.losses = []
        
    def update(self, outputs: torch.Tensor, targets: torch.Tensor, 
              features: Optional[torch.Tensor] = None, loss: Optional[float] = None):
        """
        Update metrics with batch results
        
        Args:
            outputs: Model predictions
            targets: Ground truth labels
            features: Feature representations (optional)
            loss: Batch loss (optional)
        """
        _, predicted = torch.max(outputs.data, 1)
        self.total += targets.size(0)
        self.correct += (predicted == targets).sum().item()
        
        # Store for detailed analysis
        self.predictions.extend(predicted.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
        if features is not None:
            self.features.extend(features.cpu().numpy())
        
        if loss is not None:
            self.losses.append(loss)
        
        # Per-class accuracy
        for target, pred in zip(targets.cpu().numpy(), predicted.cpu().numpy()):
            if target not in self.class_correct:
                self.class_correct[target] = 0
                self.class_total[target] = 0
            
            self.class_total[target] += 1
            if target == pred:
                self.class_correct[target] += 1
    
    def get_accuracy(self) -> float:
        """Get overall accuracy"""
        return 100.0 * self.correct / self.total if self.total > 0 else 0.0
    
    def get_class_accuracies(self) -> Dict[int, float]:
        """Get per-class accuracies"""
        class_acc = {}
        for class_id in self.class_total:
            class_acc[class_id] = 100.0 * self.class_correct[class_id] / self.class_total[class_id]
        return class_acc
    
    def get_average_loss(self) -> float:
        """Get average loss"""
        return np.mean(self.losses) if self.losses else 0.0
    
    def get_confusion_matrix(self, num_classes: int) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(self.targets, self.predictions, 
                              labels=list(range(num_classes)))
    
    def get_classification_report(self, class_names: Optional[List[str]] = None) -> str:
        """Get detailed classification report"""
        return classification_report(self.targets, self.predictions, 
                                   target_names=class_names, zero_division=0)


class CurvatureTracker:
    """
    Track curvature-related metrics during training
    """
    
    def __init__(self, feature_dim: int):
        self.geometry = RiemannianGeometry(feature_dim)
        self.curvature_history = []
        self.distance_ratios = []
        self.manifold_diameters = []
        
    def update(self, features: torch.Tensor, labels: torch.Tensor, 
              current_kappa: float):
        """
        Update curvature metrics
        
        Args:
            features: Feature tensor
            labels: Label tensor
            current_kappa: Current curvature parameter
        """
        with torch.no_grad():
            # Compute metric tensor
            metric_tensor = self.geometry.compute_metric_tensor(features, labels)
            
            # Compute curvature
            ricci_curvature = self.geometry.ricci_curvature(metric_tensor).item()
            self.curvature_history.append(ricci_curvature)
            
            # Compute inter/intra class distance ratio
            distance_ratio = self._compute_distance_ratio(features, labels, metric_tensor)
            self.distance_ratios.append(distance_ratio)
            
            # Compute manifold diameter
            diameter = compute_manifold_diameter(features, metric_tensor, self.geometry).item()
            self.manifold_diameters.append(diameter)
    
    def _compute_distance_ratio(self, features: torch.Tensor, labels: torch.Tensor,
                               metric_tensor: torch.Tensor) -> float:
        """Compute ratio of inter-class to intra-class distances"""
        batch_size = features.size(0)
        
        inter_distances = []
        intra_distances = []
        
        for i in range(min(batch_size, 50)):  # Sample for efficiency
            for j in range(i + 1, min(batch_size, 50)):
                dist = self.geometry.riemannian_distance(
                    features[i], features[j], metric_tensor).item()
                
                if labels[i] == labels[j]:
                    intra_distances.append(dist)
                else:
                    inter_distances.append(dist)
        
        if len(inter_distances) == 0 or len(intra_distances) == 0:
            return 1.0
        
        avg_inter = np.mean(inter_distances)
        avg_intra = np.mean(intra_distances)
        
        return avg_inter / (avg_intra + 1e-8)
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get latest curvature metrics"""
        return {
            'ricci_curvature': self.curvature_history[-1] if self.curvature_history else 0.0,
            'distance_ratio': self.distance_ratios[-1] if self.distance_ratios else 1.0,
            'manifold_diameter': self.manifold_diameters[-1] if self.manifold_diameters else 0.0
        }


class UFGVCTrainer:
    """
    Complete training pipeline for Ultra-Fine-Grained Visual Categorization
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[Any] = None,
                 device: torch.device = torch.device('cuda'),
                 save_dir: str = './checkpoints',
                 experiment_name: str = 'ufgvc_experiment',
                 use_wandb: bool = False,
                 wandb_project: str = 'ufgvc'):
        """
        Initialize UFGVC trainer
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device for training
            save_dir: Directory to save checkpoints
            experiment_name: Name of the experiment
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Setup directories
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Setup logging
        self.use_wandb = use_wandb and wandb is not None
        if self.use_wandb:
            wandb.init(project=wandb_project, name=experiment_name)
            wandb.watch(model)
        
        # Initialize trackers
        feature_dim = getattr(model, 'feature_dim', 512)
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        self.curvature_tracker = CurvatureTracker(feature_dim)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'curvature': [],
            'distance_ratio': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # Setup CutMix if needed
        self.cutmix = CutMix(alpha=1.0, prob=0.5)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Apply CutMix with some probability
            if random.random() < 0.5:
                data, targets_a, targets_b, lam = self.cutmix(data, targets)
            else:
                targets_a, targets_b, lam = targets, targets, 1.0
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'forward') and 'return_features' in self.model.forward.__code__.co_varnames:
                outputs, features = self.model(data, return_features=True)
            else:
                outputs = self.model(data)
                features = None
            
            # Compute loss
            if isinstance(self.criterion, CombinedLoss):
                loss, loss_components = self.criterion(features, outputs, targets)
            elif isinstance(self.criterion, CurvatureAwareLoss):
                if features is not None:
                    loss = self.criterion(features, targets)
                else:
                    # Fallback to cross-entropy if no features available
                    loss = F.cross_entropy(outputs, targets)
            else:
                if lam != 1.0:  # CutMix case
                    loss = lam * self.criterion(outputs, targets_a) + \
                           (1 - lam) * self.criterion(outputs, targets_b)
                else:
                    loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Get current curvature for optimizer
            current_kappa = 0.0
            if hasattr(self.criterion, 'kappa'):
                current_kappa = self.criterion.kappa.item()
            
            # Optimizer step with curvature info
            if hasattr(self.optimizer, 'step') and 'curvature' in self.optimizer.step.__code__.co_varnames:
                self.optimizer.step(curvature=current_kappa)
            else:
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            self.train_metrics.update(outputs, targets, features, loss.item())
            
            # Update curvature tracker
            if features is not None:
                self.curvature_tracker.update(features, targets, current_kappa)
            
            # Log batch progress
            if batch_idx % 50 == 0:
                self.logger.info(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                               f'Loss: {loss.item():.4f}')
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        accuracy = self.train_metrics.get_accuracy()
        
        # Update learning rate
        if self.scheduler is not None:
            if hasattr(self.scheduler, 'step'):
                if 'metrics' in self.scheduler.step.__code__.co_varnames:
                    self.scheduler.step({'loss': avg_loss, 'accuracy': accuracy})
                else:
                    self.scheduler.step()
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'curvature_metrics': self.curvature_tracker.get_latest_metrics()
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward') and 'return_features' in self.model.forward.__code__.co_varnames:
                    outputs, features = self.model(data, return_features=True)
                else:
                    outputs = self.model(data)
                    features = None
                
                # Compute loss
                if isinstance(self.criterion, (CombinedLoss, CurvatureAwareLoss)):
                    if features is not None:
                        if isinstance(self.criterion, CombinedLoss):
                            loss, _ = self.criterion(features, outputs, targets)
                        else:
                            loss = self.criterion(features, targets)
                    else:
                        loss = F.cross_entropy(outputs, targets)
                else:
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                self.val_metrics.update(outputs, targets, features, loss.item())
        
        avg_loss = total_loss / num_batches
        accuracy = self.val_metrics.get_accuracy()
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], 
                       is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'model_info': get_model_info(self.model)
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'{self.experiment_name}_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / f'{self.experiment_name}_best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f'New best model saved with accuracy: {metrics["val_acc"]:.2f}%')
    
    def train(self, num_epochs: int, save_every: int = 10, 
             validate_every: int = 1) -> Dict[str, List[float]]:
        """
        Complete training loop
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            validate_every: Validate every N epochs
            
        Returns:
            Training history
        """
        self.logger.info(f'Starting training for {num_epochs} epochs')
        self.logger.info(f'Model info: {get_model_info(self.model)}')
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = {}
            if epoch % validate_every == 0:
                val_metrics = self.validate_epoch(epoch)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['curvature'].append(train_metrics['curvature_metrics']['ricci_curvature'])
            self.history['distance_ratio'].append(train_metrics['curvature_metrics']['distance_ratio'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Check if best model
            is_best = False
            if val_metrics and val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch
                is_best = True
            
            # Logging
            epoch_time = time.time() - epoch_start
            log_msg = (f'Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s) - '
                      f'Train Loss: {train_metrics["loss"]:.4f}, '
                      f'Train Acc: {train_metrics["accuracy"]:.2f}%')
            
            if val_metrics:
                log_msg += (f', Val Loss: {val_metrics["loss"]:.4f}, '
                           f'Val Acc: {val_metrics["accuracy"]:.2f}%')
            
            log_msg += (f', Curvature: {train_metrics["curvature_metrics"]["ricci_curvature"]:.4f}, '
                       f'Distance Ratio: {train_metrics["curvature_metrics"]["distance_ratio"]:.4f}, '
                       f'LR: {current_lr:.6f}')
            
            self.logger.info(log_msg)
            
            # W&B logging
            if self.use_wandb:
                wandb_metrics = {
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'train/accuracy': train_metrics['accuracy'],
                    'train/curvature': train_metrics['curvature_metrics']['ricci_curvature'],
                    'train/distance_ratio': train_metrics['curvature_metrics']['distance_ratio'],
                    'learning_rate': current_lr
                }
                
                if val_metrics:
                    wandb_metrics.update({
                        'val/loss': val_metrics['loss'],
                        'val/accuracy': val_metrics['accuracy']
                    })
                
                wandb.log(wandb_metrics)
            
            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                combined_metrics = {**train_metrics, **val_metrics}
                combined_metrics['val_acc'] = val_metrics.get('accuracy', 0.0)
                self.save_checkpoint(epoch, combined_metrics, is_best)
        
        total_time = time.time() - start_time
        self.logger.info(f'Training completed in {total_time/3600:.2f} hours')
        self.logger.info(f'Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}')
        
        return self.history
    
    def evaluate(self, test_loader: DataLoader, 
                class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation on test set
        
        Args:
            test_loader: Test data loader
            class_names: List of class names
            
        Returns:
            Evaluation results
        """
        self.logger.info('Starting comprehensive evaluation...')
        
        self.model.eval()
        test_metrics = MetricsTracker()
        
        all_features = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                if hasattr(self.model, 'forward') and 'return_features' in self.model.forward.__code__.co_varnames:
                    outputs, features = self.model(data, return_features=True)
                    all_features.extend(features.cpu().numpy())
                else:
                    outputs = self.model(data)
                
                test_metrics.update(outputs, targets)
        
        # Compute metrics
        accuracy = test_metrics.get_accuracy()
        class_accuracies = test_metrics.get_class_accuracies()
        
        num_classes = len(class_accuracies)
        confusion_mat = test_metrics.get_confusion_matrix(num_classes)
        classification_rep = test_metrics.get_classification_report(class_names)
        
        results = {
            'accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'confusion_matrix': confusion_mat,
            'classification_report': classification_rep,
            'predictions': test_metrics.predictions,
            'targets': test_metrics.targets
        }
        
        # Feature visualization if available
        if all_features:
            results['feature_visualization'] = self._visualize_features(
                np.array(all_features), test_metrics.targets, class_names)
        
        self.logger.info(f'Test accuracy: {accuracy:.2f}%')
        self.logger.info(f'Classification report:\n{classification_rep}')
        
        return results
    
    def _visualize_features(self, features: np.ndarray, labels: np.ndarray,
                           class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create feature space visualizations
        
        Args:
            features: Feature array
            labels: Label array
            class_names: Class names
            
        Returns:
            Visualization results
        """
        # Sample for visualization if too many points
        if len(features) > 1000:
            indices = np.random.choice(len(features), 1000, replace=False)
            features = features[indices]
            labels = np.array(labels)[indices]
        
        # t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(features)
        
        # UMAP visualization
        features_umap = None
        if umap is not None:
            reducer = umap.UMAP(random_state=42)
            features_umap = reducer.fit_transform(features)
        
        return {
            'tsne_features': features_tsne,
            'umap_features': features_umap,
            'labels': labels,
            'class_names': class_names
        }
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        if self.history['val_loss']:
            axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        if self.history['val_acc']:
            axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        
        # Curvature plot
        axes[1, 0].plot(self.history['curvature'])
        axes[1, 0].set_title('Ricci Curvature')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Curvature')
        
        # Distance ratio plot
        axes[1, 1].plot(self.history['distance_ratio'])
        axes[1, 1].set_title('Inter/Intra Class Distance Ratio')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Distance Ratio')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if self.use_wandb:
            wandb.log({"training_history": wandb.Image(fig)})
        
        plt.show()


def create_trainer(model_name: str,
                  num_classes: int,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  device: torch.device,
                  lr: float = 0.01,
                  use_cal: bool = True,
                  **kwargs) -> UFGVCTrainer:
    """
    Factory function to create UFGVC trainer
    
    Args:
        model_name: Name of the model
        num_classes: Number of classes
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Training device
        lr: Learning rate
        use_cal: Whether to use Curvature-Aware Loss
        **kwargs: Additional arguments
        
    Returns:
        Trainer instance
    """
    # Create model
    feature_dim = kwargs.get('feature_dim', 512)
    model = create_model(model_name, num_classes, feature_dim)
    
    # Create loss function
    if use_cal:
        criterion = CombinedLoss(feature_dim)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    optimizer_name = kwargs.get('optimizer', 'riemannian_sgd' if use_cal else 'adam')
    optimizer = create_optimizer(model, optimizer_name, lr)
    
    # Create scheduler
    scheduler_name = kwargs.get('scheduler', 'cosine')
    scheduler = create_scheduler(optimizer, scheduler_name, **kwargs)
    
    return UFGVCTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        **kwargs
    )
