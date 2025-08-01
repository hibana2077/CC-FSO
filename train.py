"""
Main training script for CC-FSO: Ultra-Fine-Grained Visual Categorization
黎曼流形上的超細粒度視覺分類主訓練腳本
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import sys
from pathlib import Path
import logging
import json
import random
import numpy as np

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from dataset.ufgvc import UFGVCDataset
from ccfso import (
    create_model, create_trainer, create_augmentation_pipeline,
    CurvatureAwareLoss, CombinedLoss, RiemannianSGD, RiemannianAdam
)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CC-FSO model for UFGVC')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cotton80',
                      choices=['cotton80', 'soybean', 'soy_ageing_r1', 'soy_ageing_r3', 
                              'soy_ageing_r4', 'soy_ageing_r5', 'soy_ageing_r6'],
                      help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='./data',
                      help='Root directory for datasets')
    parser.add_argument('--image_size', type=int, default=224,
                      help='Input image size. Must be compatible with chosen model (e.g., 224 for models with _224, 384 for models with _384)')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='swin_base_patch4_window7_224',
                      help='Model architecture. Note: model name should match --image_size (e.g., model with 224 in name requires --image_size 224)')
    parser.add_argument('--feature_dim', type=int, default=512,
                      help='Feature dimension for Riemannian manifold')
    parser.add_argument('--pretrained', action='store_true', default=True,
                      help='Use pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='Weight decay')
    
    # Loss function arguments
    parser.add_argument('--loss', type=str, default='combined',
                      choices=['cal', 'combined', 'ce'],
                      help='Loss function to use')
    parser.add_argument('--curvature_constraint', type=float, default=-0.1,
                      help='Curvature constraint parameter κ')
    parser.add_argument('--lambda_curvature', type=float, default=0.1,
                      help='Weight for curvature regularization')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='riemannian_sgd',
                      choices=['riemannian_sgd', 'riemannian_adam', 'adam', 'sgd', 'adamw'],
                      help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine',
                      choices=['cosine', 'step', 'curvature'],
                      help='Learning rate scheduler')
    
    # Augmentation arguments
    parser.add_argument('--use_ufgvc_aug', action='store_true', default=True,
                      help='Use UFGVC-specific augmentations')
    parser.add_argument('--contrastive_aug', action='store_true',
                      help='Use contrastive augmentation')
    
    # Training settings
    parser.add_argument('--num_workers', type=int, default=8,
                      help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--experiment_name', type=str, default='ufgvc_experiment',
                      help='Name of the experiment')
    parser.add_argument('--save_every', type=int, default=10,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--validate_every', type=int, default=1,
                      help='Validate every N epochs')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                      help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='ufgvc',
                      help='W&B project name')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    return parser.parse_args()


def validate_model_image_size_compatibility(model_name: str, image_size: int) -> None:
    """
    Validate that the model and image_size are compatible
    
    Args:
        model_name: Name of the model architecture
        image_size: Target image size
        
    Raises:
        ValueError: If model and image_size are incompatible
    """
    # Extract expected size from model name if it contains size info
    if '224' in model_name and image_size != 224:
        if 'swin' in model_name.lower() or 'vit' in model_name.lower():
            raise ValueError(
                f"Model '{model_name}' expects input size 224x224, but --image_size {image_size} was specified.\n"
                f"Solutions:\n"
                f"  1. Use --image_size 224\n"
                f"  2. Use a different model that supports {image_size}x{image_size} inputs:\n"
                f"     - For 384x384: vit_small_patch16_384, swin_base_patch4_window7_384 (if available)\n"
                f"     - For flexible sizes: resnet50, resnet101, efficientnet models"
            )
    
    elif '384' in model_name and image_size != 384:
        if 'swin' in model_name.lower() or 'vit' in model_name.lower():
            raise ValueError(
                f"Model '{model_name}' expects input size 384x384, but --image_size {image_size} was specified.\n"
                f"Solutions:\n"
                f"  1. Use --image_size 384\n"
                f"  2. Use a different model that supports {image_size}x{image_size} inputs"
            )
    
    # Provide helpful suggestions for common cases
    if image_size == 384 and '224' in model_name:
        suggested_models = []
        if 'vit' in model_name.lower():
            suggested_models.append('vit_small_patch16_384')
            suggested_models.append('vit_base_patch16_384')
        elif 'swin' in model_name.lower():
            suggested_models.append('swin_base_patch4_window12_384')
            suggested_models.append('swin_large_patch4_window12_384')
        
        if suggested_models:
            print(f"💡 Tip: For --image_size 384, consider using these models: {', '.join(suggested_models)}")


def get_device(device_arg: str) -> torch.device:
    """Get appropriate device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    return device


def create_dataloaders(args):
    """Create training and validation dataloaders"""
    print(f"Creating dataloaders for dataset: {args.dataset}")
    
    # Create augmentation pipelines
    train_transform = create_augmentation_pipeline(
        image_size=args.image_size,
        training=True,
        contrastive=args.contrastive_aug,
        ufgvc_specific=args.use_ufgvc_aug
    )
    
    val_transform = create_augmentation_pipeline(
        image_size=args.image_size,
        training=False,
        ufgvc_specific=False
    )
    
    # Create datasets
    train_dataset = UFGVCDataset(
        dataset_name=args.dataset,
        root=args.data_root,
        split='train',
        transform=train_transform,
        download=True
    )
    
    val_dataset = UFGVCDataset(
        dataset_name=args.dataset,
        root=args.data_root,
        split='val',
        transform=val_transform,
        download=False
    )
    
    # If no validation set, try test set
    if len(val_dataset) == 0:
        try:
            val_dataset = UFGVCDataset(
                dataset_name=args.dataset,
                root=args.data_root,
                split='test',
                transform=val_transform,
                download=False
            )
        except:
            # Split training set for validation
            print("No validation set found, splitting training set...")
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {train_dataset.dataset.num_classes if hasattr(train_dataset, 'dataset') else len(train_dataset.classes)}")
    
    return train_loader, val_loader


def main():
    """Main training function"""
    args = parse_args()
    
    # Validate model and image_size compatibility
    try:
        validate_model_image_size_compatibility(args.model, args.image_size)
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        sys.exit(1)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Get device
    device = get_device(args.device)
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.save_dir, f'{args.experiment_name}_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)
    
    # Get number of classes
    if hasattr(train_loader.dataset, 'dataset'):
        num_classes = len(train_loader.dataset.dataset.classes)
    else:
        num_classes = len(train_loader.dataset.classes)
    
    print(f"Number of classes: {num_classes}")
    
    # Create model
    print(f"Creating model: {args.model}")
    model = create_model(
        model_name=args.model,
        num_classes=num_classes,
        feature_dim=args.feature_dim,
        pretrained=args.pretrained
    )
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    if args.loss == 'cal':
        criterion = CurvatureAwareLoss(
            feature_dim=args.feature_dim,
            curvature_constraint=args.curvature_constraint,
            lambda_curvature=args.lambda_curvature
        )
    elif args.loss == 'combined':
        criterion = CombinedLoss(feature_dim=args.feature_dim)
    else:  # cross-entropy
        criterion = nn.CrossEntropyLoss()
    
    criterion = criterion.to(device)
    
    # Create optimizer
    if args.optimizer == 'riemannian_sgd':
        optimizer = RiemannianSGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'riemannian_adam':
        optimizer = RiemannianAdam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:  # SGD
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=0.9
        )
    
    # Create scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None
    
    # Create trainer
    from ccfso.trainer import UFGVCTrainer
    
    trainer = UFGVCTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        experiment_name=args.experiment_name,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project
    )
    
    # Start training
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Experiment: {args.experiment_name}")
    print(f"Save directory: {args.save_dir}")
    
    history = trainer.train(
        num_epochs=args.epochs,
        save_every=args.save_every,
        validate_every=args.validate_every
    )
    
    # Save training history
    history_path = os.path.join(args.save_dir, f'{args.experiment_name}_history.json')
    with open(history_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_history = {}
        for key, values in history.items():
            json_history[key] = [float(v) if isinstance(v, (np.float32, np.float64)) else v for v in values]
        json.dump(json_history, f, indent=2)
    
    # Plot training history
    trainer.plot_training_history(
        save_path=os.path.join(args.save_dir, f'{args.experiment_name}_history.png')
    )
    
    print("Training completed!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}% at epoch {trainer.best_epoch}")
    
    # Test evaluation if test set exists
    try:
        test_dataset = UFGVCDataset(
            dataset_name=args.dataset,
            root=args.data_root,
            split='test',
            transform=create_augmentation_pipeline(args.image_size, training=False),
            download=False
        )
        
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            
            print("Evaluating on test set...")
            test_results = trainer.evaluate(test_loader, test_dataset.classes)
            print(f"Test accuracy: {test_results['accuracy']:.2f}%")
            
            # Save test results
            test_results_path = os.path.join(args.save_dir, f'{args.experiment_name}_test_results.json')
            
            # Helper function to convert numpy types to Python types
            def convert_numpy_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {str(k): convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            # Convert all numpy types to JSON-serializable types
            json_results = convert_numpy_types(test_results)
            
            with open(test_results_path, 'w') as f:
                json.dump(json_results, f, indent=2)
    
    except Exception as e:
        print(f"No test set available or error in evaluation: {e}")


if __name__ == '__main__':
    main()
