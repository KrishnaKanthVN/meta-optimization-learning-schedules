import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
import pandas as pd


def load_experiment_results(exp_dir):
    """Load metrics from an experiment directory."""
    metrics_path = Path(exp_dir) / 'metrics.json'
    config_path = Path(exp_dir) / 'config.json'
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    return metrics, config


def plot_training_curves(metrics, config, save_dir):
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'Loss Curves: {config["optimizer"]} + {config["scheduler"]}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, metrics['val_acc'], 'r-', label='Validation Accuracy')
    plt.title(f'Accuracy Curves: {config["optimizer"]} + {config["scheduler"]}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    plt.close()


def plot_gradient_weight_norms(metrics, config, save_dir):
    epochs = range(1, len(metrics['grad_norms']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Gradient Norms
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['grad_norms'], 'g-')
    plt.title(f'Gradient Norms: {config["optimizer"]} + {config["scheduler"]}')
    plt.xlabel('Epochs')
    plt.ylabel('L2 Norm')
    plt.grid(True, alpha=0.3)
    
    # Weight Norms
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics['weight_norms'], 'm-')
    plt.title(f'Weight Norms: {config["optimizer"]} + {config["scheduler"]}')
    plt.xlabel('Epochs')
    plt.ylabel('L2 Norm')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'norm_curves.png'), dpi=300)
    plt.close()


def plot_learning_rates(metrics, config, save_dir):
    epochs = range(1, len(metrics['lr_values']) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics['lr_values'], 'r-')
    plt.title(f'Learning Rate Schedule: {config["scheduler"]}')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lr_schedule.png'), dpi=300)
    plt.close()


def plot_generalization_gap(metrics, config, save_dir):
    epochs = range(1, len(metrics['train_acc']) + 1)
    gen_gap = [train - val for train, val in zip(metrics['train_acc'], metrics['val_acc'])]
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, gen_gap, 'purple')
    plt.title(f'Generalization Gap: {config["optimizer"]} + {config["scheduler"]}')
    plt.xlabel('Epochs')
    plt.ylabel('Train Acc - Val Acc (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'generalization_gap.png'), dpi=300)
    plt.close()


def compare_experiments(exp_dirs, metric_name, summary_dir, xlabel='Epochs', ylabel=None, title=None):
    """Compare a specific metric across multiple experiments."""
    plt.figure(figsize=(12, 7))
    
    for exp_dir in exp_dirs:
        metrics, config = load_experiment_results(exp_dir)
        epochs = range(1, len(metrics[metric_name]) + 1)
        label = f"{config['optimizer']} + {config['scheduler']}"
        plt.plot(epochs, metrics[metric_name], label=label)
    
    if title is None:
        title = f'Comparison of {metric_name}'
    if ylabel is None:
        ylabel = metric_name.replace('_', ' ').title()
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(summary_dir, exist_ok=True)
    plt.savefig(os.path.join(summary_dir, f'{metric_name}_comparison.png'), dpi=300)
    plt.close()


def create_summary_table(results_dir, summary_dir):
    exp_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    summary_data = []
    
    for exp_dir in exp_dirs:
        try:
            full_path = os.path.join(results_dir, exp_dir)
            metrics, config = load_experiment_results(full_path)
            final_val_acc = metrics['val_acc'][-1]
            final_train_acc = metrics['train_acc'][-1]
            best_val_acc = max(metrics['val_acc'])
            best_val_epoch = metrics['val_acc'].index(best_val_acc) + 1
            threshold = 0.9 * best_val_acc
            convergence_epochs = next((i+1 for i, acc in enumerate(metrics['val_acc']) if acc >= threshold), len(metrics['val_acc']))
            gen_gap = final_train_acc - final_val_acc
            
            summary_data.append({
                'Experiment': exp_dir,
                'Optimizer': config['optimizer'],
                'Scheduler': config['scheduler'],
                'Final Val Acc (%)': round(final_val_acc, 2),
                'Best Val Acc (%)': round(best_val_acc, 2),
                'Best Val Epoch': best_val_epoch,
                'Convergence Epochs': convergence_epochs,
                'Gen Gap (%)': round(gen_gap, 2)
            })
        except Exception as e:
            print(f"Error processing {exp_dir}: {e}")
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values('Best Val Acc (%)', ascending=False)
    
    os.makedirs(summary_dir, exist_ok=True)
    df.to_csv(os.path.join(summary_dir, 'summary_table.csv'), index=False)
    
    return df


if __name__ == "__main__":
    results_dir = "./results"
    summary_dir = "./results_summary/overall"
    os.makedirs(summary_dir, exist_ok=True)
    
    exp_dirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) 
                if os.path.isdir(os.path.join(results_dir, d))]
    
    if exp_dirs:
        for exp_dir in exp_dirs:
            metrics, config = load_experiment_results(exp_dir)
            
            # Unique save dir per experiment
            exp_name = config["experiment_name"]
            save_dir = os.path.join("./results_summary", f"{config['optimizer']}_{config['scheduler']}_{exp_name}")
            os.makedirs(save_dir, exist_ok=True)
            
            # Generate and save plots for this experiment
            plot_training_curves(metrics, config, save_dir)
            plot_gradient_weight_norms(metrics, config, save_dir)
            plot_learning_rates(metrics, config, save_dir)
            plot_generalization_gap(metrics, config, save_dir)
        
        # Generate comparison plot and summary table
        compare_experiments(exp_dirs, 'val_acc', summary_dir,
                            ylabel='Validation Accuracy (%)',
                            title='Validation Accuracy Across Experiments')
        
        summary_df = create_summary_table(results_dir, summary_dir)
