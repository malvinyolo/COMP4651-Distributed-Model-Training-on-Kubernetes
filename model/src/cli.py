"""
CLI: Single entrypoint for training and evaluation
"""
import argparse
import os
import yaml
from copy import deepcopy

from utils import seed_everything, get_device, log, Timer
from datamod import build_dataloaders
from models import LSTMClassifier, GRUClassifier
from train import fit, test
from artifacts import (
    make_run_dir,
    save_state_dict,
    save_json,
    save_yaml,
    save_confusion
)
from metrics import bin_metrics


# Default configuration
DEFAULT_CONFIG = {
    'data': {
        'npz_path': '/mnt/data/sp500_classification.npz',
        'valid_from_train': 0.1,
        'shuffle_train': False,
        'norm': 'zscore'
    },
    'model': {
        'kind': 'lstm',
        'input_dim': None,
        'hidden': 64,
        'layers': 1,
        'dropout': 0.1
    },
    'train': {
        'epochs': 25,
        'batch_size': 64,
        'lr': 1.0e-3,
        'weight_decay': 0.0,
        'early_stop_metric': 'auc',
        'early_stop_patience': 5,
        'seed': 42,
        'device': 'auto'
    },
    'eval': {
        'threshold': 0.5,
        'save_cm': True
    },
    'io': {
        'save_dir': './outputs',
        'run_name': None
    }
}


def load_config(config_path: str = None, overrides: dict = None) -> dict:
    """
    Load configuration from file and apply overrides.
    
    Args:
        config_path: Optional path to YAML config file
        overrides: Optional dictionary of overrides
    
    Returns:
        Resolved configuration dictionary
    """
    # Start with defaults
    cfg = deepcopy(DEFAULT_CONFIG)
    
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_cfg = yaml.safe_load(f)
        
        # Deep merge
        for section, values in file_cfg.items():
            if section in cfg and isinstance(values, dict):
                cfg[section].update(values)
            else:
                cfg[section] = values
    
    # Apply CLI overrides
    if overrides:
        for key, value in overrides.items():
            # Parse nested keys like "model.kind" or "train.epochs"
            if '.' in key:
                section, param = key.split('.', 1)
                if section in cfg:
                    cfg[section][param] = value
            else:
                cfg[key] = value
    
    return cfg


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train binary classifier on SP500 sequences'
    )
    
    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML config file'
    )
    
    # Common overrides
    parser.add_argument(
        '--npz_path',
        type=str,
        default=None,
        help='Path to NPZ data file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['lstm', 'gru'],
        default=None,
        help='Model type'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--hidden',
        type=int,
        default=None,
        help='Hidden layer size'
    )
    
    parser.add_argument(
        '--layers',
        type=int,
        default=None,
        help='Number of recurrent layers'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default=None,
        help='Device to use'
    )
    
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Custom run name'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    
    # Parse arguments
    args = parse_args()
    
    # Build overrides from CLI args
    overrides = {}
    if args.npz_path:
        overrides['data.npz_path'] = args.npz_path
    if args.model:
        overrides['model.kind'] = args.model
    if args.epochs:
        overrides['train.epochs'] = args.epochs
    if args.batch_size:
        overrides['train.batch_size'] = args.batch_size
    if args.lr:
        overrides['train.lr'] = args.lr
    if args.hidden:
        overrides['model.hidden'] = args.hidden
    if args.layers:
        overrides['model.layers'] = args.layers
    if args.seed:
        overrides['train.seed'] = args.seed
    if args.device:
        overrides['train.device'] = args.device
    if args.run_name:
        overrides['io.run_name'] = args.run_name
    
    # Load and resolve config
    cfg = load_config(args.config, overrides)
    
    # Set random seed
    seed_everything(cfg['train']['seed'])
    log(f"Set random seed: {cfg['train']['seed']}")
    
    # Get device
    device = get_device(cfg['train']['device'])
    log(f"Using device: {device}")
    
    # Create run directory
    run_dir = make_run_dir(cfg['io']['save_dir'], cfg['io']['run_name'])
    log(f"Run directory: {run_dir}")
    
    # Build dataloaders
    log("Loading data and building dataloaders...")
    with Timer("Data loading"):
        train_loader, val_loader, test_loader, norm_stats, input_dim = build_dataloaders(cfg)
    
    log(f"  Train batches: {len(train_loader)}")
    log(f"  Val batches: {len(val_loader)}")
    log(f"  Test batches: {len(test_loader)}")
    log(f"  Input dim: {input_dim}")
    
    # Set input_dim in config
    if cfg['model']['input_dim'] is None:
        cfg['model']['input_dim'] = input_dim
    else:
        assert cfg['model']['input_dim'] == input_dim, \
            f"Config input_dim {cfg['model']['input_dim']} != data input_dim {input_dim}"
    
    # Build model
    log(f"Building {cfg['model']['kind'].upper()} model...")
    if cfg['model']['kind'] == 'lstm':
        model = LSTMClassifier(
            input_dim=cfg['model']['input_dim'],
            hidden=cfg['model']['hidden'],
            layers=cfg['model']['layers'],
            dropout=cfg['model']['dropout']
        )
    elif cfg['model']['kind'] == 'gru':
        model = GRUClassifier(
            input_dim=cfg['model']['input_dim'],
            hidden=cfg['model']['hidden'],
            layers=cfg['model']['layers'],
            dropout=cfg['model']['dropout']
        )
    else:
        raise ValueError(f"Unknown model kind: {cfg['model']['kind']}")
    
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters: {n_params:,}")
    
    # Train
    log("\n" + "="*60)
    log("TRAINING")
    log("="*60)
    
    checkpoint_path = os.path.join(run_dir, 'best.ckpt')
    
    with Timer("Training"):
        _, history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            save_path=checkpoint_path
        )
    
    # Evaluate on validation set with best checkpoint
    log("\n" + "="*60)
    log("VALIDATION (BEST CHECKPOINT)")
    log("="*60)
    
    val_results = test(
        model=model,
        test_loader=val_loader,
        checkpoint_path=checkpoint_path,
        device=device,
        threshold=cfg['eval']['threshold']
    )
    
    # Test
    log("\n" + "="*60)
    log("TEST")
    log("="*60)
    
    test_results = test(
        model=model,
        test_loader=test_loader,
        checkpoint_path=checkpoint_path,
        device=device,
        threshold=cfg['eval']['threshold']
    )
    
    # Save artifacts
    log("\n" + "="*60)
    log("SAVING ARTIFACTS")
    log("="*60)
    
    # Save config
    config_path = os.path.join(run_dir, 'config.yaml')
    save_yaml(cfg, config_path)
    log(f"Saved config: {config_path}")
    
    # Save normalization stats
    if norm_stats:
        norm_path = os.path.join(run_dir, 'norm_stats.json')
        save_json(norm_stats, norm_path)
        log(f"Saved norm stats: {norm_path}")
    
    # Save validation metrics
    val_metrics_path = os.path.join(run_dir, 'metrics_valid.json')
    save_json(val_results['metrics'], val_metrics_path)
    log(f"Saved validation metrics: {val_metrics_path}")
    
    # Save test metrics
    test_metrics_path = os.path.join(run_dir, 'metrics_test.json')
    save_json(test_results['metrics'], test_metrics_path)
    log(f"Saved test metrics: {test_metrics_path}")
    
    # Save confusion matrices
    if cfg['eval']['save_cm']:
        val_cm_path = os.path.join(run_dir, 'confusion_matrix_valid.png')
        save_confusion(val_results['confusion_matrix'], val_cm_path)
        log(f"Saved validation confusion matrix: {val_cm_path}")
        
        test_cm_path = os.path.join(run_dir, 'confusion_matrix_test.png')
        save_confusion(test_results['confusion_matrix'], test_cm_path)
        log(f"Saved test confusion matrix: {test_cm_path}")
    
    # Print summary
    log("\n" + "="*60)
    log("SUMMARY")
    log("="*60)
    
    val_m = val_results['metrics']
    test_m = test_results['metrics']
    
    log(f"VAL:  acc={val_m['acc']:.2f} auc={val_m['auc']:.2f} "
        f"prec={val_m['prec']:.2f} rec={val_m['rec']:.2f}")
    log(f"TEST: acc={test_m['acc']:.2f} auc={test_m['auc']:.2f} "
        f"prec={test_m['prec']:.2f} rec={test_m['rec']:.2f}")
    log(f"Saved → {run_dir}")
    
    log("\n✓ Done!")


if __name__ == '__main__':
    main()
