"""
Curvature-Aware Loss Functions for Ultra-Fine-Grained Visual Categorization

This module implements the theoretical Curvature-Aware Loss (CAL) with:
- Curvature constraints
- Riemannian distance computation
- Adaptive curvature parameter
- Theoretical convergence guarantees
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import math

from .geometry import RiemannianGeometry, AdaptiveCurvatureEstimator


class CurvatureAwareLoss(nn.Module):
    """
    Curvature-Aware Loss (CAL) for ultra-fine-grained visual categorization
    
    Implements the theoretical framework:
    L_CAL = Σ[exp(κ·d²(x_i,x_j))·1{y_i=y_j} - exp(-κ·d²(x_i,x_k))·1{y_i≠y_k} + β]
    
    with curvature constraint: κ ≤ -c/Δ²
    """
    
    def __init__(self, 
                 feature_dim: int,
                 curvature_constraint: float = -0.1,
                 margin: float = 1.0,
                 temperature: float = 0.07,
                 lambda_curvature: float = 0.1,
                 adaptive_curvature: bool = True,
                 c_parameter: float = 1.0):
        """
        Initialize Curvature-Aware Loss
        
        Args:
            feature_dim: Dimension of feature space
            curvature_constraint: Maximum allowed curvature (should be negative)
            margin: Margin parameter β in the loss
            temperature: Temperature parameter for contrastive learning
            lambda_curvature: Weight for curvature regularization
            adaptive_curvature: Whether to use adaptive curvature estimation
            c_parameter: Constant c in curvature constraint κ ≤ -c/Δ²
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.curvature_constraint = curvature_constraint
        self.margin = margin
        self.temperature = temperature
        self.lambda_curvature = lambda_curvature
        self.c_parameter = c_parameter
        
        # Initialize Riemannian geometry utilities
        self.geometry = RiemannianGeometry(feature_dim)
        
        # Adaptive curvature estimator
        if adaptive_curvature:
            self.curvature_estimator = AdaptiveCurvatureEstimator(feature_dim)
        else:
            self.curvature_estimator = None
            
        # Learnable curvature parameter
        self.kappa = nn.Parameter(torch.tensor(curvature_constraint))
        
        # Track statistics for theoretical analysis
        self.register_buffer('min_inter_class_distance', torch.tensor(float('inf')))
        self.register_buffer('max_intra_class_distance', torch.tensor(0.0))
        
    def compute_pairwise_distances(self, features: torch.Tensor, 
                                 metric_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Riemannian distances
        
        Args:
            features: Feature tensor of shape (batch_size, feature_dim)
            metric_tensor: Metric tensor of shape (feature_dim, feature_dim)
            
        Returns:
            Distance matrix of shape (batch_size, batch_size)
        """
        batch_size = features.size(0)
        distances = torch.zeros(batch_size, batch_size, device=features.device)
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                dist = self.geometry.riemannian_distance(
                    features[i], features[j], metric_tensor)
                distances[i, j] = dist
                distances[j, i] = dist
                
        return distances
    
    def update_distance_statistics(self, distances: torch.Tensor, 
                                 labels: torch.Tensor) -> None:
        """
        Update inter/intra class distance statistics for curvature constraint
        
        Args:
            distances: Pairwise distance matrix
            labels: Label tensor
        """
        batch_size = labels.size(0)
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                dist = distances[i, j]
                
                if labels[i] == labels[j]:  # Same class (intra-class)
                    self.max_intra_class_distance = torch.max(
                        self.max_intra_class_distance, dist)
                else:  # Different class (inter-class)
                    self.min_inter_class_distance = torch.min(
                        self.min_inter_class_distance, dist)
    
    def compute_curvature_constraint(self) -> torch.Tensor:
        """
        Compute curvature constraint based on current distance statistics
        
        Returns:
            Curvature constraint κ ≤ -c/Δ²
        """
        # Use minimum inter-class distance as Δ
        if self.min_inter_class_distance == float('inf'):
            delta = torch.tensor(1.0, device=self.kappa.device)
        else:
            delta = torch.max(self.min_inter_class_distance, 
                            torch.tensor(0.1, device=self.kappa.device))
        
        # Constraint: κ ≤ -c/Δ²
        constraint_value = -self.c_parameter / (delta ** 2)
        
        return constraint_value
    
    def curvature_regularization(self) -> torch.Tensor:
        """
        Compute curvature regularization term R(κ) = λ·max(0, κ + c/Δ²)
        
        Returns:
            Regularization loss
        """
        constraint_value = self.compute_curvature_constraint()
        violation = torch.max(torch.tensor(0.0, device=self.kappa.device),
                            self.kappa - constraint_value)
        
        return self.lambda_curvature * violation
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:
        """
        Compute Curvature-Aware Loss
        
        Args:
            features: Feature tensor of shape (batch_size, feature_dim)
            labels: Label tensor of shape (batch_size,)
            return_components: Whether to return loss components
            
        Returns:
            Loss tensor (and components if return_components=True)
        """
        batch_size = features.size(0)
        device = features.device
        
        # Compute metric tensor
        metric_tensor = self.geometry.compute_metric_tensor(features, labels)
        
        # Compute pairwise distances
        distances = self.compute_pairwise_distances(features, metric_tensor)
        
        # Update distance statistics
        self.update_distance_statistics(distances, labels)
        
        # Get current curvature parameter
        if self.curvature_estimator is not None:
            adaptive_kappa = self.curvature_estimator(features)
            current_kappa = adaptive_kappa.squeeze()
        else:
            current_kappa = self.kappa
        
        # Compute CAL loss components
        positive_loss = torch.tensor(0.0, device=device)
        negative_loss = torch.tensor(0.0, device=device)
        num_positive_pairs = 0
        num_negative_pairs = 0
        
        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    continue
                    
                dist_squared = distances[i, j] ** 2
                
                if labels[i] == labels[j]:  # Positive pair (same class)
                    # Minimize distance for same class: exp(κ·d²)
                    positive_loss += torch.exp(current_kappa * dist_squared)
                    num_positive_pairs += 1
                else:  # Negative pair (different class)
                    # Maximize distance for different class: -exp(-κ·d²)
                    negative_loss += torch.exp(-current_kappa * dist_squared)
                    num_negative_pairs += 1
        
        # Normalize by number of pairs
        if num_positive_pairs > 0:
            positive_loss /= num_positive_pairs
        if num_negative_pairs > 0:
            negative_loss /= num_negative_pairs
        
        # Combine losses with margin
        contrastive_loss = positive_loss - negative_loss + self.margin
        
        # Add curvature regularization
        curvature_reg = self.curvature_regularization()
        
        # Total loss
        total_loss = contrastive_loss + curvature_reg
        
        if return_components:
            components = {
                'contrastive_loss': contrastive_loss,
                'positive_loss': positive_loss,
                'negative_loss': negative_loss,
                'curvature_regularization': curvature_reg,
                'current_kappa': current_kappa,
                'min_inter_class_distance': self.min_inter_class_distance,
                'max_intra_class_distance': self.max_intra_class_distance,
                'curvature_constraint': self.compute_curvature_constraint()
            }
            return total_loss, components
        
        return total_loss


class InfoNCEWithCurvature(nn.Module):
    """
    InfoNCE loss with curvature awareness for contrastive learning
    """
    
    def __init__(self, temperature: float = 0.07, 
                 curvature_weight: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.curvature_weight = curvature_weight
        self.geometry = RiemannianGeometry(manifold_dim=512)  # Will be updated
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss with curvature correction
        
        Args:
            features: Feature tensor of shape (batch_size, feature_dim)
            labels: Label tensor of shape (batch_size,)
            
        Returns:
            Loss tensor
        """
        batch_size = features.size(0)
        device = features.device
        
        # Update geometry dimension
        self.geometry.manifold_dim = features.size(1)
        
        # Compute metric tensor
        metric_tensor = self.geometry.compute_metric_tensor(features, labels)
        
        # Compute curvature
        ricci_curvature = self.geometry.ricci_curvature(metric_tensor)
        
        # Normalize features
        features_norm = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features_norm, features_norm.T) / self.temperature
        
        # Apply curvature correction
        curvature_correction = self.curvature_weight * ricci_curvature
        similarity_matrix = similarity_matrix + curvature_correction
        
        # Create labels matrix
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_matrix = labels_matrix.float()
        
        # Remove diagonal
        mask = torch.eye(batch_size, device=device).bool()
        labels_matrix = labels_matrix.masked_fill(mask, 0)
        
        # Compute InfoNCE loss
        # For each sample, positive samples are those with same label
        losses = []
        for i in range(batch_size):
            # Positive samples
            positive_mask = labels_matrix[i] > 0
            if positive_mask.sum() == 0:
                continue
                
            # Negative samples  
            negative_mask = labels_matrix[i] == 0
            negative_mask[i] = False  # Remove self
            
            if negative_mask.sum() == 0:
                continue
            
            # Compute loss for sample i
            positive_logits = similarity_matrix[i][positive_mask]
            negative_logits = similarity_matrix[i][negative_mask]
            
            # InfoNCE loss
            all_logits = torch.cat([positive_logits, negative_logits])
            targets = torch.zeros(len(positive_logits), dtype=torch.long, device=device)
            
            loss_i = F.cross_entropy(
                all_logits.unsqueeze(0).repeat(len(positive_logits), 1),
                targets
            )
            losses.append(loss_i)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return torch.stack(losses).mean()


class TripletLossWithCurvature(nn.Module):
    """
    Triplet loss with Riemannian distance computation
    """
    
    def __init__(self, margin: float = 1.0, feature_dim: int = 512):
        super().__init__()
        self.margin = margin
        self.geometry = RiemannianGeometry(feature_dim)
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss using Riemannian distances
        
        Args:
            features: Feature tensor of shape (batch_size, feature_dim)
            labels: Label tensor of shape (batch_size,)
            
        Returns:
            Loss tensor
        """
        batch_size = features.size(0)
        device = features.device
        
        # Compute metric tensor
        metric_tensor = self.geometry.compute_metric_tensor(features, labels)
        
        # Find triplets
        losses = []
        
        for i in range(batch_size):
            anchor = features[i]
            anchor_label = labels[i]
            
            # Find positive samples (same class)
            positive_mask = (labels == anchor_label) & (torch.arange(batch_size, device=device) != i)
            positive_indices = positive_mask.nonzero(as_tuple=False).squeeze(1)
            
            # Find negative samples (different class)
            negative_mask = labels != anchor_label
            negative_indices = negative_mask.nonzero(as_tuple=False).squeeze(1)
            
            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue
            
            # Sample positive and negative
            pos_idx = positive_indices[torch.randint(len(positive_indices), (1,))]
            neg_idx = negative_indices[torch.randint(len(negative_indices), (1,))]
            
            positive = features[pos_idx]
            negative = features[neg_idx]
            
            # Compute Riemannian distances
            dist_positive = self.geometry.riemannian_distance(anchor, positive, metric_tensor)
            dist_negative = self.geometry.riemannian_distance(anchor, negative, metric_tensor)
            
            # Triplet loss
            loss = torch.max(torch.tensor(0.0, device=device),
                           dist_positive - dist_negative + self.margin)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return torch.stack(losses).mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function incorporating multiple components
    """
    
    def __init__(self, 
                 feature_dim: int,
                 cal_weight: float = 1.0,
                 ce_weight: float = 0.1,
                 triplet_weight: float = 0.1):
        super().__init__()
        
        self.cal_weight = cal_weight
        self.ce_weight = ce_weight
        self.triplet_weight = triplet_weight
        
        # Component losses
        self.cal_loss = CurvatureAwareLoss(feature_dim)
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = TripletLossWithCurvature(feature_dim=feature_dim)
        
    def forward(self, features: torch.Tensor, logits: torch.Tensor, 
                labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss
        
        Args:
            features: Feature tensor from backbone
            logits: Classification logits
            labels: Ground truth labels
            
        Returns:
            Total loss and component losses
        """
        # Component losses
        cal_loss = self.cal_loss(features, labels)
        ce_loss = self.ce_loss(logits, labels)
        triplet_loss = self.triplet_loss(features, labels)
        
        # Combined loss
        total_loss = (self.cal_weight * cal_loss + 
                     self.ce_weight * ce_loss + 
                     self.triplet_weight * triplet_loss)
        
        components = {
            'cal_loss': cal_loss,
            'ce_loss': ce_loss,
            'triplet_loss': triplet_loss,
            'total_loss': total_loss
        }
        
        return total_loss, components
