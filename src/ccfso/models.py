"""
Model Architectures for Ultra-Fine-Grained Visual Categorization

This module implements:
- Swin Transformer with Riemannian feature space
- Vision Transformer with curvature-aware heads
- Feature projection layers for manifold embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Tuple, Optional, Dict, Any
import math

from .geometry import RiemannianGeometry


class RiemannianProjectionHead(nn.Module):
    """
    Projection head for mapping features to Riemannian manifold
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int = 512,
                 hidden_dim: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # L2 normalization for manifold embedding
        self.normalize = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features to Riemannian manifold
        
        Args:
            x: Input features of shape (batch_size, input_dim)
            
        Returns:
            Manifold features of shape (batch_size, output_dim)
        """
        # Project to manifold space
        manifold_features = self.projection(x)
        
        # Normalize for manifold embedding
        manifold_features = self.normalize(manifold_features)
        
        return manifold_features


class SwinTransformerBackbone(nn.Module):
    """
    Swin Transformer backbone for feature extraction
    """
    
    def __init__(self, 
                 model_name: str = 'swin_base_patch4_window7_224',
                 pretrained: bool = True,
                 num_classes: int = 1000,
                 feature_dim: int = 512):
        super().__init__()
        
        # Load pretrained Swin Transformer
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_dim = self.backbone(dummy_input).shape[-1]
        
        self.backbone_dim = backbone_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Riemannian projection head
        self.riemannian_head = RiemannianProjectionHead(
            input_dim=backbone_dim,
            output_dim=feature_dim
        )
        
        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for new layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, 
                return_features: bool = False) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images of shape (batch_size, 3, H, W)
            return_features: Whether to return intermediate features
            
        Returns:
            Logits or (logits, features) if return_features=True
        """
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Project to Riemannian manifold
        manifold_features = self.riemannian_head(backbone_features)
        
        # Classification
        logits = self.classifier(manifold_features)
        
        if return_features:
            return logits, manifold_features
        
        return logits


class ViTWithRiemannianHead(nn.Module):
    """
    Vision Transformer with Riemannian feature space
    """
    
    def __init__(self,
                 model_name: str = 'vit_base_patch16_224',
                 pretrained: bool = True,
                 num_classes: int = 1000,
                 feature_dim: int = 512):
        super().__init__()
        
        # Load pretrained ViT
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='token'
        )

        # Get image size from model name
        if '224' in model_name:
            self.image_size = 224
        elif '384' in model_name:
            self.image_size = 384
        else:
            raise ValueError(f"Unsupported model: {model_name}. Expected image size 224 or 384.")
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.image_size, self.image_size)
            backbone_dim = self.backbone(dummy_input).shape[-1]
        
        self.backbone_dim = backbone_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Riemannian projection head
        self.riemannian_head = RiemannianProjectionHead(
            input_dim=backbone_dim,
            output_dim=feature_dim
        )
        
        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for new layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, 
                return_features: bool = False) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images of shape (batch_size, 3, H, W)
            return_features: Whether to return intermediate features
            
        Returns:
            Logits or (logits, features) if return_features=True
        """
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Project to Riemannian manifold
        manifold_features = self.riemannian_head(backbone_features)
        
        # Classification
        logits = self.classifier(manifold_features)
        
        if return_features:
            return logits, manifold_features
        
        return logits


class ResNetWithRiemannianHead(nn.Module):
    """
    ResNet backbone with Riemannian feature space
    """
    
    def __init__(self,
                 model_name: str = 'resnet50',
                 pretrained: bool = True,
                 num_classes: int = 1000,
                 feature_dim: int = 512):
        super().__init__()
        
        # Load pretrained ResNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_dim = self.backbone(dummy_input).shape[-1]
        
        self.backbone_dim = backbone_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Riemannian projection head
        self.riemannian_head = RiemannianProjectionHead(
            input_dim=backbone_dim,
            output_dim=feature_dim
        )
        
        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for new layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, 
                return_features: bool = False) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images of shape (batch_size, 3, H, W)
            return_features: Whether to return intermediate features
            
        Returns:
            Logits or (logits, features) if return_features=True
        """
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Project to Riemannian manifold
        manifold_features = self.riemannian_head(backbone_features)
        
        # Classification
        logits = self.classifier(manifold_features)
        
        if return_features:
            return logits, manifold_features
        
        return logits


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for enhanced feature representation
    """
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling
        
        Args:
            x: Features of shape (batch_size, seq_len, feature_dim)
            
        Returns:
            Pooled features of shape (batch_size, feature_dim)
        """
        # Compute attention weights
        attention_weights = self.attention(x)  # (batch_size, seq_len, 1)
        
        # Apply attention pooling
        pooled = torch.sum(x * attention_weights, dim=1)  # (batch_size, feature_dim)
        
        return pooled


class MultiScaleFeatureExtractor(nn.Module):
    """
    Multi-scale feature extraction for fine-grained details
    """
    
    def __init__(self, backbone_name: str = 'swin_base_patch4_window7_224'):
        super().__init__()
        
        # Load backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3, 4]  # Multiple scales
        )
        
        # Get feature dimensions for each scale
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dims = [f.shape[1] for f in features]
        
        # Projection layers for each scale
        self.projections = nn.ModuleList([
            nn.Conv2d(dim, 256, 1) for dim in self.feature_dims
        ])
        
        # Fusion layer
        self.fusion = nn.Conv2d(256 * len(self.feature_dims), 512, 1)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale features
        
        Args:
            x: Input images of shape (batch_size, 3, H, W)
            
        Returns:
            Fused features of shape (batch_size, 512)
        """
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Project all features to same dimension and resize
        projected_features = []
        target_size = features[-1].shape[-2:]  # Use smallest feature map size
        
        for feat, proj in zip(features, self.projections):
            projected = proj(feat)
            if projected.shape[-2:] != target_size:
                projected = F.interpolate(
                    projected, size=target_size, mode='bilinear', align_corners=False)
            projected_features.append(projected)
        
        # Concatenate and fuse
        concatenated = torch.cat(projected_features, dim=1)
        fused = self.fusion(concatenated)
        
        # Global pooling
        pooled = self.global_pool(fused).flatten(1)
        
        return pooled


def create_model(model_name: str,
                num_classes: int,
                feature_dim: int = 512,
                pretrained: bool = True) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        feature_dim: Dimension of Riemannian feature space
        pretrained: Whether to use pretrained weights
        
    Returns:
        Model instance
    """
    if 'swin' in model_name.lower():
        return SwinTransformerBackbone(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            feature_dim=feature_dim
        )
    elif 'vit' in model_name.lower():
        return ViTWithRiemannianHead(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            feature_dim=feature_dim
        )
    elif 'resnet' in model_name.lower():
        return ResNetWithRiemannianHead(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            feature_dim=feature_dim
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_type': type(model).__name__,
        'feature_dim': getattr(model, 'feature_dim', None),
        'num_classes': getattr(model, 'num_classes', None)
    }
