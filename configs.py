"""
Quick start configuration for CC-FSO experiments
快速開始配置文件
"""

# Default experimental configurations
CONFIGS = {
    # Quick test configuration
    'quick_test': {
        'dataset': 'cotton80',
        'model': 'resnet50',
        'epochs': 5,
        'batch_size': 8,
        'lr': 0.001,
        'loss': 'combined',
        'optimizer': 'riemannian_sgd',
        'pretrained': False,
        'save_every': 2,
        'validate_every': 1
    },
    
    # Full Cotton80 experiment
    'cotton80_full': {
        'dataset': 'cotton80',
        'model': 'swin_base_patch4_window7_224',
        'epochs': 100,
        'batch_size': 32,
        'lr': 0.001,
        'loss': 'combined',
        'optimizer': 'riemannian_sgd',
        'curvature_constraint': -0.1,
        'use_ufgvc_aug': True,
        'pretrained': True
    },
    
    # Soybean experiment with higher learning rate
    'soybean_experiment': {
        'dataset': 'soybean',
        'model': 'swin_base_patch4_window7_224',
        'epochs': 120,
        'batch_size': 24,
        'lr': 0.002,
        'loss': 'combined',
        'optimizer': 'riemannian_adam',
        'curvature_constraint': -0.05,
        'use_ufgvc_aug': True,
        'contrastive_aug': True
    },
    
    # Baseline comparison with ResNet
    'resnet_baseline': {
        'dataset': 'cotton80',
        'model': 'resnet50',
        'epochs': 100,
        'batch_size': 32,
        'lr': 0.01,
        'loss': 'ce',
        'optimizer': 'sgd',
        'use_ufgvc_aug': False,
        'contrastive_aug': False
    },
    
    # ViT baseline
    'vit_baseline': {
        'dataset': 'cotton80',
        'model': 'vit_base_patch16_224',
        'epochs': 100,
        'batch_size': 16,
        'lr': 0.0001,
        'loss': 'ce',
        'optimizer': 'adam',
        'use_ufgvc_aug': False,
        'contrastive_aug': False
    },
    
    # Curvature ablation study
    'curvature_ablation': {
        'base_config': {
            'dataset': 'cotton80',
            'model': 'swin_small_patch4_window7_224',
            'epochs': 50,
            'batch_size': 16,
            'lr': 0.001,
            'loss': 'cal',
            'optimizer': 'riemannian_sgd'
        },
        'curvature_values': [-1.0, -0.5, -0.1, -0.01, 0.0, 0.1]
    }
}

# Dataset specific settings
DATASET_CONFIGS = {
    'cotton80': {
        'expected_classes': 80,
        'recommended_batch_size': 32,
        'recommended_epochs': 100,
        'difficulty': 'medium'
    },
    'soybean': {
        'expected_classes': 200,
        'recommended_batch_size': 24,
        'recommended_epochs': 120,
        'difficulty': 'high'
    },
    'soy_ageing_r1': {
        'expected_classes': 198,
        'recommended_batch_size': 16,
        'recommended_epochs': 150,
        'difficulty': 'very_high'
    }
}

# Model specific settings
MODEL_CONFIGS = {
    'swin_base_patch4_window7_224': {
        'feature_dim': 1024,
        'recommended_lr': 0.001,
        'memory_usage': 'high'
    },
    'swin_small_patch4_window7_224': {
        'feature_dim': 768,
        'recommended_lr': 0.002,
        'memory_usage': 'medium'
    },
    'vit_base_patch16_224': {
        'feature_dim': 768,
        'recommended_lr': 0.0001,
        'memory_usage': 'high'
    },
    'resnet50': {
        'feature_dim': 2048,
        'recommended_lr': 0.01,
        'memory_usage': 'low'
    }
}


def get_config(config_name: str) -> dict:
    """Get configuration by name"""
    if config_name in CONFIGS:
        return CONFIGS[config_name].copy()
    else:
        raise ValueError(f"Configuration '{config_name}' not found. Available: {list(CONFIGS.keys())}")


def get_recommended_config(dataset: str, model: str) -> dict:
    """Get recommended configuration for dataset and model combination"""
    base_config = {
        'dataset': dataset,
        'model': model,
        'batch_size': 32,
        'lr': 0.001,
        'epochs': 100,
        'loss': 'combined',
        'optimizer': 'riemannian_sgd'
    }
    
    # Apply dataset specific settings
    if dataset in DATASET_CONFIGS:
        dataset_config = DATASET_CONFIGS[dataset]
        base_config['batch_size'] = dataset_config['recommended_batch_size']
        base_config['epochs'] = dataset_config['recommended_epochs']
    
    # Apply model specific settings
    if model in MODEL_CONFIGS:
        model_config = MODEL_CONFIGS[model]
        base_config['lr'] = model_config['recommended_lr']
        base_config['feature_dim'] = model_config['feature_dim']
    
    return base_config


def print_available_configs():
    """Print all available configurations"""
    print("Available Configurations:")
    print("=" * 50)
    
    for name, config in CONFIGS.items():
        print(f"\n{name}:")
        for key, value in config.items():
            if key != 'curvature_values':
                print(f"  {key}: {value}")
    
    print(f"\nDataset Options: {list(DATASET_CONFIGS.keys())}")
    print(f"Model Options: {list(MODEL_CONFIGS.keys())}")


if __name__ == '__main__':
    print_available_configs()
