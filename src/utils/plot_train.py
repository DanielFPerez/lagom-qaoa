# Add these imports at the top of your script
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
from typing import Dict

matplotlib.use('Agg')  # Use non-interactive backend for headless servers

SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels


matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["DejaVu Serif"]

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)


# Add these plotting functions after your class definitions and before train_rl_qaoa

def plot_training_progress(metrics: Dict, dst_dir_path: str, p: int):
    """Plot training progress showing train and test rewards over epochs."""
    
    # Extract training data
    train_epochs = [entry['epoch'] for entry in metrics['train_rewards']]
    train_rewards = [entry['mean_reward'] for entry in metrics['train_rewards']]
    train_stds = [entry['std_reward'] for entry in metrics['train_rewards']]
    
    # Extract test data
    test_epochs = [entry['epoch'] for entry in metrics['test_evaluations']]
    test_rewards = [entry['mean_reward'] for entry in metrics['test_evaluations']]
    test_stds = [entry['std_reward'] for entry in metrics['test_evaluations']]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot training rewards
    plt.subplot(2, 1, 1)
    plt.plot(train_epochs, train_rewards, 'b-', linewidth=2, label='Train Mean Reward', alpha=0.8)
    plt.fill_between(train_epochs, 
                     np.array(train_rewards) - np.array(train_stds),
                     np.array(train_rewards) + np.array(train_stds),
                     alpha=0.3, color='blue', label='Train ± 1 std')
    
    plt.xlabel('Epoch')
    plt.ylabel('Mean Reward')
    plt.title(f'Training Progress - QAOA p={p} (Training Set)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot test rewards
    plt.subplot(2, 1, 2)
    plt.plot(test_epochs, test_rewards, 'r-', linewidth=2, label='Test Mean Reward', marker='o', markersize=4)
    plt.fill_between(test_epochs,
                     np.array(test_rewards) - np.array(test_stds),
                     np.array(test_rewards) + np.array(test_stds),
                     alpha=0.3, color='red', label='Test ± 1 std')
    
    # Mark best test performance
    if test_rewards:
        best_idx = np.argmax(test_rewards)
        best_epoch = test_epochs[best_idx]
        best_reward = test_rewards[best_idx]
        plt.scatter([best_epoch], [best_reward], color='gold', s=100, zorder=5, 
                   label=f'Best Test (Epoch {best_epoch})', edgecolors='black')
    
    plt.xlabel('Epoch')
    plt.ylabel('Mean Reward')
    plt.title(f'Test Set Evaluation - QAOA p={p}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(dst_dir_path, f'training_progress_p{p}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"Training progress plot saved to: {plot_path}")


def plot_combined_progress(metrics: Dict, dst_dir_path: str, p: int):
    """Plot both training and test rewards on the same plot for easier comparison."""
    
    # Extract data
    train_epochs = [entry['epoch'] for entry in metrics['train_rewards']]
    train_rewards = [entry['mean_reward'] for entry in metrics['train_rewards']]
    train_stds = [entry['std_reward'] for entry in metrics['train_rewards']]
    
    test_epochs = [entry['epoch'] for entry in metrics['test_evaluations']]
    test_rewards = [entry['mean_reward'] for entry in metrics['test_evaluations']]
    test_stds = [entry['std_reward'] for entry in metrics['test_evaluations']]
    
    plt.figure(figsize=(14, 8))
    
    # Plot training rewards (line)
    plt.plot(train_epochs, train_rewards, 'b-', linewidth=2, label='Train Mean Reward', alpha=0.8)
    plt.fill_between(train_epochs, 
                     np.array(train_rewards) - np.array(train_stds),
                     np.array(train_rewards) + np.array(train_stds),
                     alpha=0.2, color='blue')
    
    # Plot test rewards (markers + line)
    plt.plot(test_epochs, test_rewards, 'r-', linewidth=2, label='Test Mean Reward', 
             marker='o', markersize=6, markerfacecolor='white', markeredgecolor='red', markeredgewidth=2)
    plt.fill_between(test_epochs,
                     np.array(test_rewards) - np.array(test_stds),
                     np.array(test_rewards) + np.array(test_stds),
                     alpha=0.2, color='red')
    
    # Mark best test performance
    if test_rewards:
        best_idx = np.argmax(test_rewards)
        best_epoch = test_epochs[best_idx]
        best_reward = test_rewards[best_idx]
        plt.scatter([best_epoch], [best_reward], color='gold', s=150, zorder=5, 
                   label=f'Best Test (Epoch {best_epoch}: {best_reward:.4f})', 
                   edgecolors='black', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Reward', fontsize=12)
    plt.title(f'Training Progress Comparison - QAOA p={p}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add summary text
    if metrics['train_rewards'] and metrics['test_evaluations']:
        final_train = metrics['train_rewards'][-1]['mean_reward']
        final_test = metrics['test_evaluations'][-1]['mean_reward']
        best_test = max([entry['mean_reward'] for entry in metrics['test_evaluations']])
        
        summary_text = f"Final Train: {final_train:.4f}\nFinal Test: {final_test:.4f}\nBest Test: {best_test:.4f}"
        plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(dst_dir_path, f'combined_progress_p{p}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined progress plot saved to: {plot_path}")


def plot_live_progress(metrics: Dict, dst_dir_path: str, p: int, epoch: int):
    """Create a live-updating plot during training (called after each test evaluation)."""
    
    if not metrics['train_rewards'] or not metrics['test_evaluations']:
        return
    
    # Create a simple live plot
    train_epochs = [entry['epoch'] for entry in metrics['train_rewards']]
    train_rewards = [entry['mean_reward'] for entry in metrics['train_rewards']]
    
    test_epochs = [entry['epoch'] for entry in metrics['test_evaluations']]
    test_rewards = [entry['mean_reward'] for entry in metrics['test_evaluations']]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_epochs, train_rewards, 'b-', linewidth=1.5, label='Train', alpha=0.7)
    plt.plot(test_epochs, test_rewards, 'r-o', linewidth=2, markersize=4, label='Test')
    
    # Mark current best
    if test_rewards:
        best_reward = max(test_rewards)
        best_idx = test_rewards.index(best_reward)
        best_epoch = test_epochs[best_idx]
        plt.scatter([best_epoch], [best_reward], color='gold', s=80, zorder=5, 
                   edgecolors='black', linewidth=1)
    
    plt.xlabel('Epoch')
    plt.ylabel('Mean Reward')
    plt.title(f'QAOA p={p} - Training Progress (Epoch {epoch})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save live plot
    live_plot_path = os.path.join(dst_dir_path, f'live_progress_p{p}.png')
    plt.savefig(live_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
