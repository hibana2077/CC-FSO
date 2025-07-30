"""
Simple test script to verify CC-FSO implementation
簡單測試腳本驗證實現是否正確
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from ccfso import (
        create_model, CurvatureAwareLoss, RiemannianSGD, 
        RiemannianGeometry, create_augmentation_pipeline
    )
    from dataset.ufgvc import UFGVCDataset
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_model_creation():
    """Test model creation"""
    print("\n--- Testing Model Creation ---")
    try:
        model = create_model(
            model_name='resnet50',  # Use ResNet50 as it's more stable
            num_classes=10,
            feature_dim=256,
            pretrained=False  # Don't download pretrained weights for testing
        )
        print(f"✓ Model created: {type(model).__name__}")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits, features = model(x, return_features=True)
        
        print(f"✓ Forward pass successful: logits {logits.shape}, features {features.shape}")
        return model
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return None


def test_geometry():
    """Test Riemannian geometry operations"""
    print("\n--- Testing Riemannian Geometry ---")
    try:
        geometry = RiemannianGeometry(manifold_dim=256)
        
        # Test with dummy data
        features = torch.randn(4, 256)
        labels = torch.tensor([0, 0, 1, 1])
        
        # Test metric tensor computation
        metric_tensor = geometry.compute_metric_tensor(features, labels)
        print(f"✓ Metric tensor computed: {metric_tensor.shape}")
        
        # Test curvature computation
        ricci_curvature = geometry.ricci_curvature(metric_tensor)
        print(f"✓ Ricci curvature computed: {ricci_curvature.item():.4f}")
        
        # Test distance computation
        dist = geometry.riemannian_distance(features[0], features[1], metric_tensor)
        print(f"✓ Riemannian distance computed: {dist.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Geometry test failed: {e}")
        return False


def test_loss_function():
    """Test Curvature-Aware Loss"""
    print("\n--- Testing Curvature-Aware Loss ---")
    try:
        criterion = CurvatureAwareLoss(feature_dim=256)
        
        # Test with dummy data
        features = torch.randn(4, 256)
        labels = torch.tensor([0, 0, 1, 1])
        
        loss = criterion(features, labels)
        print(f"✓ CAL loss computed: {loss.item():.4f}")
        
        # Test backward pass
        loss.backward()
        print("✓ Backward pass successful")
        
        return True
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        return False


def test_optimizer():
    """Test Riemannian optimizer"""
    print("\n--- Testing Riemannian Optimizer ---")
    try:
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 256)
        )
        
        optimizer = RiemannianSGD(model.parameters(), lr=0.01)
        
        # Test optimization step
        x = torch.randn(4, 10)
        target = torch.randn(4, 256)
        
        output = model(x)
        loss = nn.MSELoss()(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(curvature=-0.1)
        
        print(f"✓ Optimizer step successful: loss {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Optimizer test failed: {e}")
        return False


def test_augmentation():
    """Test augmentation pipeline"""
    print("\n--- Testing Augmentation Pipeline ---")
    try:
        from PIL import Image
        import numpy as np
        
        # Create dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Test augmentation pipeline
        aug_pipeline = create_augmentation_pipeline(
            image_size=224,
            training=True,
            contrastive=False,
            ufgvc_specific=True
        )
        
        augmented = aug_pipeline(dummy_image)
        print(f"✓ Augmentation successful: {augmented.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Augmentation test failed: {e}")
        return False


def test_dataset():
    """Test dataset loading"""
    print("\n--- Testing Dataset ---")
    try:
        # Test dataset creation (this will download data)
        dataset = UFGVCDataset(
            dataset_name='cotton80',
            root='./test_data',
            split='train',
            download=True
        )
        
        print(f"✓ Dataset created: {len(dataset)} samples, {len(dataset.classes)} classes")
        
        # Test sample loading
        if len(dataset) > 0:
            sample_image, sample_label = dataset[0]
            print(f"✓ Sample loaded: image {sample_image.size}, label {sample_label}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False


def test_integration():
    """Test integration of all components"""
    print("\n--- Testing Integration ---")
    try:
        # Create all components
        model = create_model('resnet50', num_classes=80, feature_dim=256, pretrained=False)
        criterion = CurvatureAwareLoss(feature_dim=256)
        optimizer = RiemannianSGD(model.parameters(), lr=0.001)
        
        # Dummy data
        x = torch.randn(2, 3, 224, 224)
        labels = torch.tensor([0, 1])
        
        # Forward pass
        logits, features = model(x, return_features=True)
        
        # Compute loss
        loss = criterion(features, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(curvature=criterion.kappa.item())
        
        print(f"✓ Integration test successful: loss {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("CC-FSO IMPLEMENTATION TEST")
    print("="*60)
    
    tests = [
        test_model_creation,
        test_geometry,
        test_loss_function,
        test_optimizer,
        test_augmentation,
        test_dataset,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! CC-FSO implementation is ready.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
