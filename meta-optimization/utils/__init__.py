import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from pathlib import Path

class ExperimentTracker:
    """Tracks experiment metrics and saves them to disk."""
    def __init__(self, exp_name, save_dir="./results"):
        self.exp_name = exp_name
        self.save_dir = Path(save_dir) / exp_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'grad_norms': [],
            'weight_norms': [],
            'lr_values': []
        }
        
    def update(self, epoch, train_loss, train_acc, val_loss, val_acc, 
              grad_norm=None, weight_norm=None, lr=None):
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
        
        if grad_norm is not None:
            self.metrics['grad_norms'].append(grad_norm)
        if weight_norm is not None:
            self.metrics['weight_norms'].append(weight_norm)
        if lr is not None:
            self.metrics['lr_values'].append(lr)
            
        # Save metrics after each update
        with open(self.save_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f)
    
    def save_config(self, config):
        with open(self.save_dir / 'config.json', 'w') as f:
            json.dump(config, f)
            
    def save_model(self, model, epoch):
        torch.save(model.state_dict(), self.save_dir / f'model_epoch_{epoch}.pth')


def get_dataloaders(dataset_name, batch_size=128):
    """Set up dataloaders for training and validation."""
    if dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                         download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                        download=True, transform=transform_test)
        
    elif dataset_name == 'fashionmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = datasets.FashionMNIST(root='./data', train=True, 
                                             download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, 
                                            download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # For tracking gradient norms
    grad_norms = []
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Calculate gradient norm
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        grad_norms.append(grad_norm)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    avg_grad_norm = sum(grad_norms) / len(grad_norms)
    
    return avg_loss, accuracy, avg_grad_norm


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def calculate_weight_norm(model):
    """Calculate the L2 norm of model weights."""
    weight_norm = 0.0
    for param in model.parameters():
        weight_norm += param.data.norm(2).item() ** 2
    return weight_norm ** 0.5