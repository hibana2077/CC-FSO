"""
Comprehensive experiment script for CC-FSO research
根據實驗計劃執行完整的消融實驗和對比實驗
"""

import subprocess
import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import argparse


class ExperimentRunner:
    """
    Experiment runner for systematic evaluation
    """
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.results_dir = Path('experiment_results')
        self.results_dir.mkdir(exist_ok=True)
        
    def run_experiment(self, config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
        """
        Run a single experiment
        
        Args:
            config: Experiment configuration
            experiment_name: Name of the experiment
            
        Returns:
            Experiment results
        """
        print(f"\n{'='*60}")
        print(f"Running experiment: {experiment_name}")
        print(f"{'='*60}")
        
        # Build command
        cmd = ['python', 'train.py']
        
        for key, value in config.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f'--{key}')
            else:
                cmd.extend([f'--{key}', str(value)])
        
        # Set experiment name
        cmd.extend(['--experiment_name', experiment_name])
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Run experiment
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
            
            if result.returncode == 0:
                print(f"Experiment {experiment_name} completed successfully")
                
                # Try to load results
                history_file = Path('checkpoints') / f'{experiment_name}_history.json'
                if history_file.exists():
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                    
                    return {
                        'status': 'success',
                        'config': config,
                        'history': history,
                        'best_val_acc': max(history.get('val_acc', [0])) if history.get('val_acc') else 0,
                        'final_train_acc': history.get('train_acc', [0])[-1] if history.get('train_acc') else 0
                    }
                else:
                    return {
                        'status': 'success',
                        'config': config,
                        'message': 'No history file found'
                    }
            else:
                print(f"Experiment {experiment_name} failed")
                print(f"Error: {result.stderr}")
                return {
                    'status': 'failed',
                    'config': config,
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            print(f"Experiment {experiment_name} timed out")
            return {
                'status': 'timeout',
                'config': config
            }
        except Exception as e:
            print(f"Experiment {experiment_name} failed with exception: {e}")
            return {
                'status': 'error',
                'config': config,
                'error': str(e)
            }


def curvature_constraint_ablation(runner: ExperimentRunner) -> List[Dict[str, Any]]:
    """
    消融實驗：測試不同曲率約束強度的影響
    κ = {-∞, -0.5, -0.1, 0, 0.1, 0.5}
    """
    print("\n" + "="*80)
    print("ABLATION STUDY: Curvature Constraint")
    print("="*80)
    
    curvature_values = [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5]
    results = []
    
    for kappa in curvature_values:
        config = runner.base_config.copy()
        config.update({
            'curvature_constraint': kappa,
            'loss': 'cal',
            'optimizer': 'riemannian_sgd',
            'epochs': 50  # Shorter for ablation
        })
        
        experiment_name = f'ablation_curvature_k{kappa}'
        result = runner.run_experiment(config, experiment_name)
        results.append(result)
    
    return results


def optimizer_comparison(runner: ExperimentRunner) -> List[Dict[str, Any]]:
    """
    對比實驗：黎曼優化器 vs 傳統優化器
    """
    print("\n" + "="*80)
    print("COMPARISON: Riemannian vs Euclidean Optimizers")
    print("="*80)
    
    optimizers = [
        ('riemannian_sgd', 'cal'),
        ('riemannian_adam', 'cal'),
        ('sgd', 'ce'),
        ('adam', 'ce')
    ]
    
    results = []
    
    for optimizer, loss in optimizers:
        config = runner.base_config.copy()
        config.update({
            'optimizer': optimizer,
            'loss': loss,
            'epochs': 100
        })
        
        experiment_name = f'comparison_opt_{optimizer}_loss_{loss}'
        result = runner.run_experiment(config, experiment_name)
        results.append(result)
    
    return results


def model_architecture_comparison(runner: ExperimentRunner) -> List[Dict[str, Any]]:
    """
    對比實驗：不同模型架構
    """
    print("\n" + "="*80)
    print("COMPARISON: Model Architectures")
    print("="*80)
    
    models = [
        'swin_base_patch4_window7_224',
        'vit_base_patch16_224',
        'resnet50',
        'swin_small_patch4_window7_224'
    ]
    
    results = []
    
    for model in models:
        config = runner.base_config.copy()
        config.update({
            'model': model,
            'loss': 'combined',
            'optimizer': 'riemannian_sgd',
            'epochs': 100
        })
        
        experiment_name = f'comparison_model_{model.replace("_", "-")}'
        result = runner.run_experiment(config, experiment_name)
        results.append(result)
    
    return results


def dataset_comparison(runner: ExperimentRunner) -> List[Dict[str, Any]]:
    """
    對比實驗：在不同數據集上的性能
    """
    print("\n" + "="*80)
    print("COMPARISON: Datasets")
    print("="*80)
    
    datasets = ['cotton80', 'soybean', 'soy_ageing_r1', 'soy_ageing_r3']
    results = []
    
    for dataset in datasets:
        config = runner.base_config.copy()
        config.update({
            'dataset': dataset,
            'loss': 'combined',
            'optimizer': 'riemannian_sgd',
            'epochs': 100
        })
        
        experiment_name = f'comparison_dataset_{dataset}'
        result = runner.run_experiment(config, experiment_name)
        results.append(result)
    
    return results


def loss_function_ablation(runner: ExperimentRunner) -> List[Dict[str, Any]]:
    """
    消融實驗：不同損失函數組合
    """
    print("\n" + "="*80)
    print("ABLATION STUDY: Loss Functions")
    print("="*80)
    
    loss_configs = [
        'cal',
        'combined',
        'ce'
    ]
    
    results = []
    
    for loss in loss_configs:
        config = runner.base_config.copy()
        config.update({
            'loss': loss,
            'optimizer': 'riemannian_sgd' if loss in ['cal', 'combined'] else 'adam',
            'epochs': 80
        })
        
        experiment_name = f'ablation_loss_{loss}'
        result = runner.run_experiment(config, experiment_name)
        results.append(result)
    
    return results


def augmentation_ablation(runner: ExperimentRunner) -> List[Dict[str, Any]]:
    """
    消融實驗：數據增強策略
    """
    print("\n" + "="*80)
    print("ABLATION STUDY: Data Augmentation")
    print("="*80)
    
    aug_configs = [
        {'use_ufgvc_aug': True, 'contrastive_aug': True},
        {'use_ufgvc_aug': True, 'contrastive_aug': False},
        {'use_ufgvc_aug': False, 'contrastive_aug': True},
        {'use_ufgvc_aug': False, 'contrastive_aug': False}
    ]
    
    results = []
    
    for i, aug_config in enumerate(aug_configs):
        config = runner.base_config.copy()
        config.update(aug_config)
        config.update({
            'loss': 'combined',
            'optimizer': 'riemannian_sgd',
            'epochs': 60
        })
        
        ufgvc_str = 'ufgvc' if aug_config['use_ufgvc_aug'] else 'no-ufgvc'
        contrast_str = 'contrast' if aug_config['contrastive_aug'] else 'no-contrast'
        experiment_name = f'ablation_aug_{ufgvc_str}_{contrast_str}'
        
        result = runner.run_experiment(config, experiment_name)
        results.append(result)
    
    return results


def full_comparison_with_baselines(runner: ExperimentRunner) -> List[Dict[str, Any]]:
    """
    完整對比實驗：與現有方法的比較
    """
    print("\n" + "="*80)
    print("FULL COMPARISON: CC-FSO vs Baselines")
    print("="*80)
    
    baselines = [
        # Our method
        {
            'name': 'CC-FSO (Ours)',
            'config': {
                'model': 'swin_base_patch4_window7_224',
                'loss': 'combined',
                'optimizer': 'riemannian_sgd',
                'use_ufgvc_aug': True,
                'contrastive_aug': True,
                'epochs': 120
            }
        },
        # Traditional CNN
        {
            'name': 'ResNet-50',
            'config': {
                'model': 'resnet50',
                'loss': 'ce',
                'optimizer': 'sgd',
                'use_ufgvc_aug': False,
                'contrastive_aug': False,
                'epochs': 120
            }
        },
        # Vision Transformer
        {
            'name': 'ViT-Base',
            'config': {
                'model': 'vit_base_patch16_224',
                'loss': 'ce',
                'optimizer': 'adam',
                'use_ufgvc_aug': False,
                'contrastive_aug': False,
                'epochs': 120
            }
        },
        # Swin Transformer
        {
            'name': 'Swin-Base',
            'config': {
                'model': 'swin_base_patch4_window7_224',
                'loss': 'ce',
                'optimizer': 'adam',
                'use_ufgvc_aug': False,
                'contrastive_aug': False,
                'epochs': 120
            }
        }
    ]
    
    results = []
    
    for baseline in baselines:
        config = runner.base_config.copy()
        config.update(baseline['config'])
        
        experiment_name = f"baseline_{baseline['name'].replace(' ', '_').replace('(', '').replace(')', '').lower()}"
        result = runner.run_experiment(config, experiment_name)
        result['method_name'] = baseline['name']
        results.append(result)
    
    return results


def analyze_results(all_results: Dict[str, List[Dict[str, Any]]]):
    """
    分析實驗結果並生成報告
    """
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS ANALYSIS")
    print("="*80)
    
    analysis = {}
    
    for experiment_type, results in all_results.items():
        print(f"\n{experiment_type.upper()}:")
        print("-" * 40)
        
        successful_results = [r for r in results if r['status'] == 'success' and 'best_val_acc' in r]
        
        if not successful_results:
            print("No successful experiments")
            continue
        
        # Sort by best validation accuracy
        successful_results.sort(key=lambda x: x['best_val_acc'], reverse=True)
        
        analysis[experiment_type] = successful_results
        
        for i, result in enumerate(successful_results[:3]):  # Top 3
            config = result['config']
            acc = result['best_val_acc']
            
            print(f"{i+1}. Accuracy: {acc:.2f}%")
            
            # Print key config parameters
            key_params = ['model', 'loss', 'optimizer', 'curvature_constraint', 'dataset']
            for param in key_params:
                if param in config:
                    print(f"   {param}: {config[param]}")
            print()
    
    # Save analysis
    analysis_file = Path('experiment_results') / 'analysis.json'
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nDetailed analysis saved to: {analysis_file}")


def main():
    """主實驗函數"""
    parser = argparse.ArgumentParser(description='Run comprehensive experiments')
    parser.add_argument('--dataset', type=str, default='cotton80',
                      help='Primary dataset for experiments')
    parser.add_argument('--quick', action='store_true',
                      help='Run quick experiments (fewer epochs)')
    parser.add_argument('--experiments', nargs='+', 
                      choices=['curvature', 'optimizer', 'model', 'dataset', 'loss', 'augmentation', 'baseline'],
                      default=['curvature', 'optimizer', 'loss'],
                      help='Which experiments to run')
    
    args = parser.parse_args()
    
    # Base configuration
    base_config = {
        'dataset': args.dataset,
        'data_root': './data',
        'image_size': 224,
        'batch_size': 16,  # Smaller batch size for stability
        'lr': 0.001,
        'weight_decay': 1e-4,
        'num_workers': 4,
        'save_dir': './checkpoints',
        'device': 'auto',
        'seed': 42,
        'save_every': 20,
        'validate_every': 5
    }
    
    if args.quick:
        print("Running quick experiments (reduced epochs)")
        # Reduce epochs for quick testing
        base_config.update({
            'epochs': 20,
            'save_every': 5,
            'validate_every': 2
        })
    
    runner = ExperimentRunner(base_config)
    all_results = {}
    
    # Run selected experiments
    if 'curvature' in args.experiments:
        all_results['curvature_ablation'] = curvature_constraint_ablation(runner)
    
    if 'optimizer' in args.experiments:
        all_results['optimizer_comparison'] = optimizer_comparison(runner)
    
    if 'model' in args.experiments:
        all_results['model_comparison'] = model_architecture_comparison(runner)
    
    if 'dataset' in args.experiments:
        all_results['dataset_comparison'] = dataset_comparison(runner)
    
    if 'loss' in args.experiments:
        all_results['loss_ablation'] = loss_function_ablation(runner)
    
    if 'augmentation' in args.experiments:
        all_results['augmentation_ablation'] = augmentation_ablation(runner)
    
    if 'baseline' in args.experiments:
        all_results['baseline_comparison'] = full_comparison_with_baselines(runner)
    
    # Save all results
    results_file = runner.results_dir / 'all_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to: {results_file}")
    
    # Analyze results
    analyze_results(all_results)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)


if __name__ == '__main__':
    main()
