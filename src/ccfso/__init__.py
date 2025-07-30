"""
CC-FSO: Curvature-Constrained Feature Space Optimization
黎曼流形上的超細粒度視覺分類：曲率約束的特徵空間優化理論

This package implements the Curvature-Aware Loss (CAL) for ultra-fine-grained visual categorization
on Riemannian manifolds with theoretical guarantees.
"""

# Import main components
from .models import (
    SwinTransformerBackbone, 
    ViTWithRiemannianHead, 
    ResNetWithRiemannianHead,
    RiemannianProjectionHead,
    create_model,
    get_model_info
)

from .losses import (
    CurvatureAwareLoss,
    InfoNCEWithCurvature,
    TripletLossWithCurvature,
    CombinedLoss
)

from .optimizers import (
    RiemannianSGD,
    RiemannianAdam,
    CurvatureScheduler,
    create_optimizer,
    create_scheduler
)

from .geometry import (
    RiemannianGeometry,
    AdaptiveCurvatureEstimator,
    compute_manifold_diameter,
    compute_covering_number
)

from .augmentations import (
    RiemannianAugmentation,
    ContrastiveAugmentation,
    UltraFineGrainedAugmentation,
    AugmentationPipeline,
    CutMix,
    create_augmentation_pipeline
)

from .trainer import (
    UFGVCTrainer,
    MetricsTracker,
    CurvatureTracker,
    create_trainer
)

__version__ = "1.0.0"
__author__ = "hibana2077"

__all__ = [
    # Models
    'SwinTransformerBackbone', 'ViTWithRiemannianHead', 'ResNetWithRiemannianHead',
    'RiemannianProjectionHead', 'create_model', 'get_model_info',
    
    # Losses
    'CurvatureAwareLoss', 'InfoNCEWithCurvature', 'TripletLossWithCurvature', 'CombinedLoss',
    
    # Optimizers
    'RiemannianSGD', 'RiemannianAdam', 'CurvatureScheduler', 'create_optimizer', 'create_scheduler',
    
    # Geometry
    'RiemannianGeometry', 'AdaptiveCurvatureEstimator', 'compute_manifold_diameter', 'compute_covering_number',
    
    # Augmentations
    'RiemannianAugmentation', 'ContrastiveAugmentation', 'UltraFineGrainedAugmentation',
    'AugmentationPipeline', 'CutMix', 'create_augmentation_pipeline',
    
    # Trainer
    'UFGVCTrainer', 'MetricsTracker', 'CurvatureTracker', 'create_trainer'
]
