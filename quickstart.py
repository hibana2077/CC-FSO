"""
Quick start script for CC-FSO
快速開始腳本
"""

import subprocess
import sys
import argparse
from configs import get_config, get_recommended_config, print_available_configs


def run_command(cmd_args):
    """Run training command with given arguments"""
    cmd = ['python', 'train.py']
    
    for key, value in cmd_args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✓ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Quick start script for CC-FSO')
    parser.add_argument('--config', type=str, 
                       choices=['quick_test', 'cotton80_full', 'soybean_experiment', 
                               'resnet_baseline', 'vit_baseline'],
                       help='Use predefined configuration')
    parser.add_argument('--dataset', type=str, default='cotton80',
                       choices=['cotton80', 'soybean', 'soy_ageing_r1', 'soy_ageing_r3'],
                       help='Dataset to use')
    parser.add_argument('--model', type=str, default='swin_small_patch4_window7_224',
                       choices=['swin_base_patch4_window7_224', 'swin_small_patch4_window7_224',
                               'vit_base_patch16_224', 'resnet50'],
                       help='Model architecture')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test (5 epochs)')
    parser.add_argument('--list-configs', action='store_true',
                       help='List all available configurations')
    
    args = parser.parse_args()
    
    if args.list_configs:
        print_available_configs()
        return
    
    # Get configuration
    if args.config:
        config = get_config(args.config)
        experiment_name = f"quickstart_{args.config}"
    elif args.quick:
        config = get_config('quick_test')
        config['dataset'] = args.dataset
        config['model'] = args.model
        experiment_name = f"quickstart_test_{args.dataset}_{args.model.replace('_', '-')}"
    else:
        config = get_recommended_config(args.dataset, args.model)
        experiment_name = f"quickstart_{args.dataset}_{args.model.replace('_', '-')}"
    
    # Set experiment name
    config['experiment_name'] = experiment_name
    
    # Print configuration
    print("Configuration:")
    print("=" * 40)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 40)
    print()
    
    # Confirm before running
    response = input("Do you want to start training with this configuration? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Training cancelled.")
        return
    
    # Run training
    run_command(config)


if __name__ == '__main__':
    main()
