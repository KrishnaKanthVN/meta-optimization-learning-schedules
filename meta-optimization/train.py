import torch
import torch.nn as nn
import argparse
import json
import random
import numpy as np
import os
from datetime import datetime

from models import get_model
from optimizers import get_optimizer, get_scheduler
from utils import get_dataloaders, train_epoch, validate, calculate_weight_norm, ExperimentTracker


def get_config():
    """Define the experiment configuration."""
    # Check if MPS is available (MacOS GPU)
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Apple GPU is available! Using Mac GPU acceleration.")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("CUDA not available. Using CPU.")
    else:
        device = 'cpu'
        print("MPS not available. Using CPU.")
    
    config = {
        #'experiment_name': f'meta_opt_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        #'experiment_name': f'{config["optimizer"]}_{config["scheduler"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'dataset': 'cifar10',  # 'cifar10' or 'fashionmnist'
        'model': 'resnet18',  # 'resnet18' or 'simple_cnn'
        'batch_size': 128,
        'epochs': 100,
        'optimizer': 'adamw',  # 'sgd', 'adam', 'rmsprop', 'adamw'
        'optimizer_params': {
            'lr': 0.001, #0.1 for sgd
            'momentum': 0.9,
            'weight_decay': 0.01, #5e-4 for sgd
            'betas':(0.9,0.999)
        },
        'scheduler': 'cosine',  # 'constant', 'step', 'cosine', 'onecycle', 'cyclical'
        'scheduler_params': {
            'T_max': 100,  # For cosine annealing
            'eta_min': 0,
            'step_size': 30,  # For step scheduler
            'gamma': 0.1,  # For step scheduler
        },
        'seed': 42,
        'device': device,
        'save_dir': './results',
        'save_frequency': 10,  # Save model every N epochs
    }

    config['experiment_name'] = f'{config["optimizer"]}_{config["scheduler"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    return config


def main():
    """Main function to run the training."""
    # Get configuration
    config = get_config()
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(config['experiment_name'], config['save_dir'])
    tracker.save_config(config)
    
    # Set device
    device = torch.device(config['device'])
    
    # Get dataloaders
    train_loader, test_loader = get_dataloaders(
        config['dataset'], 
        batch_size=config['batch_size']
    )
    
    # Determine input channels based on dataset
    input_channels = 3 if config['dataset'] == 'cifar10' else 1
    
    # Initialize model
    model = get_model(config['model'], num_classes=10, input_channels=input_channels)
    model = model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer
    optimizer = get_optimizer(
        config['optimizer'], 
        model.parameters(), 
        **config['optimizer_params']
    )
    
    # Update scheduler params with dataset-specific values if needed
    if config['scheduler'] == 'onecycle':
        config['scheduler_params']['steps_per_epoch'] = len(train_loader)
        config['scheduler_params']['epochs'] = config['epochs']
    
    # Initialize scheduler
    scheduler = get_scheduler(
        config['scheduler'], 
        optimizer, 
        **config['scheduler_params']
    )
    
    print(f"Training {config['model']} on {config['dataset']} with {config['optimizer']} optimizer and {config['scheduler']} scheduler")
    print(f"Using device: {device}")
    
    # Training loop
    for epoch in range(config['epochs']):
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train for one epoch
        train_loss, train_acc, grad_norm = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate the model
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # Calculate weight norm
        weight_norm = calculate_weight_norm(model)
        
        # Step the scheduler if it exists
        if scheduler:
            if config['scheduler'] != 'onecycle':  # OneCycleLR steps after each batch
                scheduler.step()
        
        # Update tracking metrics
        tracker.update(
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            grad_norm,
            weight_norm,
            current_lr
        )
        
        # Print metrics
        print(f"Epoch {epoch+1}/{config['epochs']}, LR: {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Grad Norm: {grad_norm:.4f}, Weight Norm: {weight_norm:.4f}")
        print("-" * 50)
        
        # Save model periodically
        if (epoch + 1) % config['save_frequency'] == 0:
            tracker.save_model(model, epoch + 1)
    
    # Save final model
    tracker.save_model(model, config['epochs'])
    
    print("Training complete!")


if __name__ == "__main__":
    main()