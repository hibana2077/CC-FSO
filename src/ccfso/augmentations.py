"""
Advanced Data Augmentation for Ultra-Fine-Grained Visual Categorization

This module implements:
- Instance-level augmentation for contrastive learning
- Riemannian-aware augmentation strategies
- Ultra-fine-grained specific augmentations
- Theoretical positive/negative pair generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import random
import cv2
from typing import Tuple, List, Optional, Union, Callable
import math


class RiemannianAugmentation:
    """
    Riemannian manifold-aware augmentation strategies
    """
    
    def __init__(self, 
                 image_size: int = 224,
                 mask_ratio_range: Tuple[float, float] = (0.15, 0.45),
                 num_patches: int = 4):
        """
        Initialize Riemannian augmentation
        
        Args:
            image_size: Target image size
            mask_ratio_range: Range for random masking ratio α ∈ [0.15, 0.45]
            num_patches: Number of patches for patch shuffling (n=4)
        """
        self.image_size = image_size
        self.mask_ratio_range = mask_ratio_range
        self.num_patches = num_patches
        
    def random_masking(self, image: torch.Tensor, 
                      mask_ratio: Optional[float] = None) -> torch.Tensor:
        """
        Apply random masking to generate positive samples
        
        Args:
            image: Input image tensor of shape (C, H, W)
            mask_ratio: Masking ratio (if None, sample from range)
            
        Returns:
            Masked image tensor
        """
        if mask_ratio is None:
            mask_ratio = random.uniform(*self.mask_ratio_range)
        
        C, H, W = image.shape
        
        # Create random mask
        mask = torch.rand(H, W) > mask_ratio
        mask = mask.float().unsqueeze(0).expand(C, -1, -1)
        
        # Apply mask (set masked pixels to random values)
        noise = torch.randn_like(image) * 0.1
        masked_image = image * mask + noise * (1 - mask)
        
        return masked_image
    
    def patch_shuffling(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply patch shuffling to generate positive samples
        
        Args:
            image: Input image tensor of shape (C, H, W)
            
        Returns:
            Patch-shuffled image tensor
        """
        C, H, W = image.shape
        
        # Calculate patch size
        patch_h = H // int(math.sqrt(self.num_patches))
        patch_w = W // int(math.sqrt(self.num_patches))
        
        # Extract patches
        patches = []
        for i in range(0, H, patch_h):
            for j in range(0, W, patch_w):
                if i + patch_h <= H and j + patch_w <= W:
                    patch = image[:, i:i+patch_h, j:j+patch_w]
                    patches.append(patch)
        
        # Shuffle patches
        random.shuffle(patches)
        
        # Reconstruct image
        shuffled_image = torch.zeros_like(image)
        patch_idx = 0
        for i in range(0, H, patch_h):
            for j in range(0, W, patch_w):
                if i + patch_h <= H and j + patch_w <= W and patch_idx < len(patches):
                    shuffled_image[:, i:i+patch_h, j:j+patch_w] = patches[patch_idx]
                    patch_idx += 1
        
        return shuffled_image
    
    def manifold_noise(self, image: torch.Tensor, 
                      noise_std: float = 0.05) -> torch.Tensor:
        """
        Add manifold-consistent noise
        
        Args:
            image: Input image tensor
            noise_std: Standard deviation of noise
            
        Returns:
            Noisy image tensor
        """
        # Add correlated noise that preserves manifold structure
        noise = torch.randn_like(image) * noise_std
        
        # Apply spatial correlation to noise
        if len(image.shape) == 3:  # (C, H, W)
            kernel_size = 3
            kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
            kernel = kernel.expand(image.size(0), 1, -1, -1)
            
            # Apply convolution to create spatially correlated noise
            noise = F.conv2d(noise.unsqueeze(0), kernel, 
                           padding=kernel_size//2, groups=image.size(0))
            noise = noise.squeeze(0)
        
        return torch.clamp(image + noise, 0, 1)


class ContrastiveAugmentation:
    """
    Contrastive learning augmentation with positive/negative pair generation
    """
    
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.riemannian_aug = RiemannianAugmentation(image_size)
        
        # Standard augmentations
        self.weak_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.strong_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def generate_positive_pair(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate positive pair for contrastive learning
        
        Args:
            image: Input PIL image
            
        Returns:
            Tuple of two augmented image tensors (positive pair)
        """
        # First view: weak augmentation
        view1 = self.weak_transform(image)
        
        # Second view: strong augmentation + Riemannian augmentation
        view2_pil = self.strong_transform(image)
        
        # Convert back to tensor for Riemannian augmentation
        if isinstance(view2_pil, torch.Tensor):
            view2 = view2_pil
        else:
            view2 = transforms.ToTensor()(view2_pil)
            view2 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(view2)
        
        # Apply Riemannian-specific augmentations
        if random.random() > 0.5:
            view2 = self.riemannian_aug.random_masking(view2)
        else:
            view2 = self.riemannian_aug.patch_shuffling(view2)
        
        # Add manifold noise
        view2 = self.riemannian_aug.manifold_noise(view2)
        
        return view1, view2
    
    def __call__(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate positive pair
        
        Args:
            image: Input PIL image
            
        Returns:
            Positive pair of augmented images
        """
        return self.generate_positive_pair(image)


class UltraFineGrainedAugmentation:
    """
    Specialized augmentations for ultra-fine-grained features
    """
    
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
    
    def local_feature_emphasis(self, image: torch.Tensor, 
                              emphasis_ratio: float = 0.3) -> torch.Tensor:
        """
        Emphasize local features important for fine-grained classification
        
        Args:
            image: Input image tensor of shape (C, H, W)
            emphasis_ratio: Ratio of image to emphasize
            
        Returns:
            Image with emphasized local features
        """
        C, H, W = image.shape
        
        # Create attention mask based on local variance
        gray_image = image.mean(dim=0, keepdim=True)  # Convert to grayscale
        
        # Compute local variance using convolution
        kernel = torch.ones(1, 1, 5, 5) / 25.0
        local_mean = F.conv2d(gray_image.unsqueeze(0), kernel, padding=2).squeeze(0)
        local_variance = F.conv2d((gray_image - local_mean).pow(2).unsqueeze(0), 
                                 kernel, padding=2).squeeze(0)
        
        # Create emphasis mask
        threshold = torch.quantile(local_variance, 1 - emphasis_ratio)
        emphasis_mask = (local_variance > threshold).float()
        emphasis_mask = emphasis_mask.expand(C, -1, -1)
        
        # Apply emphasis
        emphasized_image = image * (1 + 0.5 * emphasis_mask)
        
        return torch.clamp(emphasized_image, 0, 1)
    
    def texture_enhancement(self, image: torch.Tensor) -> torch.Tensor:
        """
        Enhance texture details for ultra-fine-grained features
        
        Args:
            image: Input image tensor
            
        Returns:
            Texture-enhanced image
        """
        # High-pass filter for texture enhancement
        blur_kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16.0
        blur_kernel = blur_kernel.unsqueeze(0).unsqueeze(0).expand(image.size(0), 1, -1, -1)
        
        # Apply Gaussian blur
        blurred = F.conv2d(image.unsqueeze(0), blur_kernel, padding=1, groups=image.size(0))
        blurred = blurred.squeeze(0)
        
        # High-pass filtered image
        high_pass = image - blurred
        
        # Enhanced image
        enhanced = image + 0.3 * high_pass
        
        return torch.clamp(enhanced, 0, 1)
    
    def adaptive_contrast(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive contrast enhancement
        
        Args:
            image: Input image tensor
            
        Returns:
            Contrast-enhanced image
        """
        # Compute local statistics
        mean_val = image.mean()
        std_val = image.std()
        
        # Adaptive contrast adjustment
        target_std = 0.25  # Target standard deviation
        contrast_factor = target_std / (std_val + 1e-8)
        contrast_factor = torch.clamp(contrast_factor, 0.5, 2.0)
        
        # Apply contrast enhancement
        enhanced = (image - mean_val) * contrast_factor + mean_val
        
        return torch.clamp(enhanced, 0, 1)


class AugmentationPipeline:
    """
    Complete augmentation pipeline for UFGVC training
    """
    
    def __init__(self, 
                 image_size: int = 224,
                 training: bool = True,
                 contrastive: bool = True):
        """
        Initialize augmentation pipeline
        
        Args:
            image_size: Target image size
            training: Whether in training mode
            contrastive: Whether to use contrastive augmentation
        """
        self.image_size = image_size
        self.training = training
        self.contrastive = contrastive
        
        # Initialize augmentation modules
        self.contrastive_aug = ContrastiveAugmentation(image_size)
        self.ufgvc_aug = UltraFineGrainedAugmentation(image_size)
        
        # Standard training augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation/test augmentation
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image: Image.Image) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply augmentation pipeline
        
        Args:
            image: Input PIL image
            
        Returns:
            Augmented image(s)
        """
        if not self.training:
            # Validation/test mode
            return self.val_transform(image)
        
        if self.contrastive:
            # Contrastive learning mode: return positive pair
            view1, view2 = self.contrastive_aug(image)
            
            # Apply ultra-fine-grained augmentations
            if random.random() > 0.5:
                view1 = self.ufgvc_aug.local_feature_emphasis(view1)
            if random.random() > 0.5:
                view2 = self.ufgvc_aug.texture_enhancement(view2)
            if random.random() > 0.5:
                view2 = self.ufgvc_aug.adaptive_contrast(view2)
            
            return view1, view2
        else:
            # Standard training mode
            augmented = self.train_transform(image)
            
            # Apply ultra-fine-grained augmentations
            if random.random() > 0.3:
                augmented = self.ufgvc_aug.local_feature_emphasis(augmented)
            if random.random() > 0.3:
                augmented = self.ufgvc_aug.texture_enhancement(augmented)
            if random.random() > 0.3:
                augmented = self.ufgvc_aug.adaptive_contrast(augmented)
            
            return augmented


class CutMix:
    """
    CutMix augmentation adapted for ultra-fine-grained classification
    """
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, batch: torch.Tensor, 
                 labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to a batch
        
        Args:
            batch: Batch of images
            labels: Batch of labels
            
        Returns:
            Mixed batch, original labels, shuffled labels, mixing ratio
        """
        if random.random() > self.prob:
            return batch, labels, labels, 1.0
        
        batch_size = batch.size(0)
        indices = torch.randperm(batch_size)
        
        # Sample mixing ratio
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Sample bounding box
        H, W = batch.size(2), batch.size(3)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        batch[:, :, bby1:bby2, bbx1:bbx2] = batch[indices, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return batch, labels, labels[indices], lam


def create_augmentation_pipeline(image_size: int = 224,
                               training: bool = True,
                               contrastive: bool = False,
                               ufgvc_specific: bool = True) -> Callable:
    """
    Factory function to create augmentation pipeline
    
    Args:
        image_size: Target image size
        training: Whether in training mode
        contrastive: Whether to use contrastive augmentation
        ufgvc_specific: Whether to include UFGVC-specific augmentations
        
    Returns:
        Augmentation pipeline function
    """
    if ufgvc_specific:
        return AugmentationPipeline(
            image_size=image_size,
            training=training,
            contrastive=contrastive
        )
    else:
        if training:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
