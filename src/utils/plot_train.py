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
    """Plot training progress showing expected discounted rewards and final QAOA values."""
    
    # Extract training data for expected discounted rewards
    train_epochs = [entry['epoch'] for entry in metrics['train_expected_discounted_rewards']]
    train_edr = [entry['mean_reward'] for entry in metrics['train_expected_discounted_rewards']]
    train_edr_stds = [entry['std_reward'] for entry in metrics['train_expected_discounted_rewards']]
    
    # Extract test data for expected discounted rewards
    test_epochs = [entry['epoch'] for entry in metrics['test_expected_discounted_rewards']]
    test_edr = [entry['mean_discounted_reward'] for entry in metrics['test_expected_discounted_rewards']]
    test_edr_stds = [entry['std_discounted_reward'] for entry in metrics['test_expected_discounted_rewards']]
    
    # Extract final QAOA values
    train_values = [entry['mean_value'] for entry in metrics['train_mean_final_values']]
    train_values_stds = [entry['std_value'] for entry in metrics['train_mean_final_values']]
    
    test_values = [entry['mean_value'] for entry in metrics['test_mean_final_values']]
    test_values_stds = [entry['std_value'] for entry in metrics['test_mean_final_values']]
    
    # Create the plot with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Expected Discounted Rewards
    ax1.plot(train_epochs, train_edr, 'b-', linewidth=2, label='Train EDR', alpha=0.8)
    ax1.fill_between(train_epochs, 
                     np.array(train_edr) - np.array(train_edr_stds),
                     np.array(train_edr) + np.array(train_edr_stds),
                     alpha=0.3, color='blue', label='Train ± 1 std')
    
    ax1.plot(test_epochs, test_edr, 'r-', linewidth=2, label='Test EDR', 
             marker='o', markersize=6, markerfacecolor='white', markeredgecolor='red', markeredgewidth=2)
    ax1.fill_between(test_epochs,
                     np.array(test_edr) - np.array(test_edr_stds),
                     np.array(test_edr) + np.array(test_edr_stds),
                     alpha=0.3, color='red', label='Test ± 1 std')
    
    # Mark best test EDR
    if test_edr:
        best_idx = np.argmax(test_edr)
        best_epoch = test_epochs[best_idx]
        best_edr = test_edr[best_idx]
        ax1.scatter([best_epoch], [best_edr], color='gold', s=100, zorder=5, 
                   label=f'Best Test EDR (Epoch {best_epoch})', edgecolors='black')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Expected Discounted Reward', fontsize=12)
    ax1.set_title(f'Expected Discounted Rewards - QAOA p={p}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final QAOA Values
    ax2.plot(train_epochs, train_values, 'b-', linewidth=2, label='Train Final Values', alpha=0.8)
    ax2.fill_between(train_epochs, 
                     np.array(train_values) - np.array(train_values_stds),
                     np.array(train_values) + np.array(train_values_stds),
                     alpha=0.3, color='blue')
    
    ax2.plot(test_epochs, test_values, 'r-', linewidth=2, label='Test Final Values',
             marker='o', markersize=6, markerfacecolor='white',
             markeredgecolor='red', markeredgewidth=2)
    ax2.fill_between(test_epochs,
                     np.array(test_values) - np.array(test_values_stds),
                     np.array(test_values) + np.array(test_values_stds),
                     alpha=0.3, color='red')
    
    # Mark best test final value
    if test_values:
        best_idx = np.argmax(test_values)
        best_epoch = test_epochs[best_idx]
        best_value = test_values[best_idx]
        ax2.scatter([best_epoch], [best_value], color='gold', s=100, zorder=5,
                   label=f'Best Test Value (Epoch {best_epoch})', edgecolors='black')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Final QAOA Objective Value', fontsize=12)
    ax2.set_title(f'Final QAOA Values - QAOA p={p}', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(dst_dir_path, f'training_progress_p{p}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training progress plot saved to: {plot_path}")



def plot_combined_progress(metrics: Dict, dst_dir_path: str, p: int):
    """Plot expected discounted rewards for train and test on the same plot."""
    
    # Extract data
    train_epochs = [entry['epoch'] for entry in metrics['train_expected_discounted_rewards']]
    train_edr = [entry['mean_reward'] for entry in metrics['train_expected_discounted_rewards']]
    train_edr_stds = [entry['std_reward'] for entry in metrics['train_expected_discounted_rewards']]
    
    test_epochs = [entry['epoch'] for entry in metrics['test_expected_discounted_rewards']]
    test_edr = [entry['mean_discounted_reward'] for entry in metrics['test_expected_discounted_rewards']]
    test_edr_stds = [entry['std_discounted_reward'] for entry in metrics['test_expected_discounted_rewards']]
    
    plt.figure(figsize=(14, 8))
    
    # Plot training EDR (line)
    plt.plot(train_epochs, train_edr, 'b-', linewidth=2.5, label='Train Expected Discounted Reward', alpha=0.8)
    plt.fill_between(train_epochs, 
                     np.array(train_edr) - np.array(train_edr_stds),
                     np.array(train_edr) + np.array(train_edr_stds),
                     alpha=0.2, color='blue')
    
    # Plot test EDR (markers + line)
    plt.plot(test_epochs, test_edr, 'r-', linewidth=2.5, label='Test Expected Discounted Reward', 
             marker='o', markersize=8, markerfacecolor='white', markeredgecolor='red', markeredgewidth=2)
    plt.fill_between(test_epochs,
                     np.array(test_edr) - np.array(test_edr_stds),
                     np.array(test_edr) + np.array(test_edr_stds),
                     alpha=0.2, color='red')
    
    # Mark best test performance
    if test_edr:
        best_idx = np.argmax(test_edr)
        best_epoch = test_epochs[best_idx]
        best_edr = test_edr[best_idx]
        plt.scatter([best_epoch], [best_edr], color='gold', s=200, zorder=5, 
                   label=f'Best Test EDR (Epoch {best_epoch}: {best_edr:.4f})', 
                   edgecolors='black', linewidth=2)
    
    # Add zero line for reference
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Expected Discounted Reward', fontsize=14)
    plt.title(f'Training Progress - QAOA p={p} (Expected Discounted Rewards)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add summary text box
    if metrics['train_expected_discounted_rewards'] and metrics['test_expected_discounted_rewards']:
        final_train_edr = metrics['train_expected_discounted_rewards'][-1]['mean_reward']
        final_test_edr = metrics['test_expected_discounted_rewards'][-1]['mean_discounted_reward']
        best_test_edr = max([entry['mean_discounted_reward'] for entry in metrics['test_expected_discounted_rewards']])
        
        # Also get final values if available
        if metrics.get('test_mean_final_values'):
            final_test_value = metrics['test_mean_final_values'][-1]['mean_value']
            best_test_value = max([entry['mean_value'] for entry in metrics['test_mean_final_values']])
            
            summary_text = (f"Expected Discounted Rewards:\n"
                          f"  Final Train: {final_train_edr:.4f}\n"
                          f"  Final Test: {final_test_edr:.4f}\n"
                          f"  Best Test: {best_test_edr:.4f}\n\n"
                          f"Final QAOA Values:\n"
                          f"  Final Test: {final_test_value:.4f}\n"
                          f"  Best Test: {best_test_value:.4f}")
        else:
            summary_text = (f"Final Train EDR: {final_train_edr:.4f}\n"
                          f"Final Test EDR: {final_test_edr:.4f}\n"
                          f"Best Test EDR: {best_test_edr:.4f}")
        
        plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add GNN type to the plot if available
    if 'gnn_type' in metrics:
        plt.text(0.98, 0.02, f"GNN: {metrics['gnn_type']}", transform=plt.gca().transAxes, 
                fontsize=12, horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
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



def plot_expected_discounted_rewards(metrics, dst_dir, p, epoch):
    """Plot expected discounted rewards instead of mean rewards"""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot expected discounted rewards
    train_epochs = [m['epoch'] for m in metrics['train_expected_discounted_rewards']]
    train_rewards = [m['mean_reward'] for m in metrics['train_expected_discounted_rewards']]
    
    test_epochs = [m['epoch'] for m in metrics['test_expected_discounted_rewards']]
    test_rewards = [m['mean_discounted_reward'] for m in metrics['test_expected_discounted_rewards']]
    
    ax1.plot(train_epochs, train_rewards, 'b-', label='Train', alpha=0.7)
    ax1.plot(test_epochs, test_rewards, 'r-o', label='Test', markersize=6)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Expected Discounted Reward')
    ax1.set_title(f'QAOA p={p} - Expected Discounted Rewards (Epoch {epoch})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot final QAOA values
    train_values = [m['mean_value'] for m in metrics['train_mean_final_values']]
    test_values = [m['mean_value'] for m in metrics['test_mean_final_values']]
    
    ax2.plot(train_epochs, train_values, 'b-', label='Train', alpha=0.7)
    ax2.plot(test_epochs, test_values, 'r-o', label='Test', markersize=6)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Final QAOA Value')
    ax2.set_title(f'QAOA p={p} - Final Objective Values (Epoch {epoch})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(dst_dir, f'live_progress_p{p}.png'))
    plt.close()




def plot_training_with_init(metrics, dst_dir_path, p, current_epoch):
    """Plot training progress including initialization quality"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Expected Discounted Rewards
        if metrics['train_expected_discounted_rewards']:
            train_epochs = [x['epoch'] for x in metrics['train_expected_discounted_rewards']]
            train_rewards = [x['mean_reward'] for x in metrics['train_expected_discounted_rewards']]
            axes[0, 0].plot(train_epochs, train_rewards, label='Train', color='blue')
        
        if metrics['test_expected_discounted_rewards']:
            test_epochs = [x['epoch'] for x in metrics['test_expected_discounted_rewards']]
            test_rewards = [x['mean_discounted_reward'] for x in metrics['test_expected_discounted_rewards']]
            axes[0, 0].plot(test_epochs, test_rewards, label='Test', color='red')
        
        axes[0, 0].set_title('Expected Discounted Rewards')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Final QAOA Values
        if metrics['train_mean_final_values']:
            train_epochs = [x['epoch'] for x in metrics['train_mean_final_values']]
            train_values = [x['mean_value'] for x in metrics['train_mean_final_values']]
            axes[0, 1].plot(train_epochs, train_values, label='Train', color='blue')
        
        if metrics['test_mean_final_values']:
            test_epochs = [x['epoch'] for x in metrics['test_mean_final_values']]
            test_values = [x['mean_value'] for x in metrics['test_mean_final_values']]
            axes[0, 1].plot(test_epochs, test_values, label='Test', color='red')
        
        axes[0, 1].set_title('Final QAOA Values (Normalized)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Initialization Quality
        if metrics['initialization_quality']:
            init_epochs = [x['epoch'] for x in metrics['initialization_quality']]
            init_improvements = [x['mean_improvement'] for x in metrics['initialization_quality']]
            axes[1, 0].plot(init_epochs, init_improvements, label='GNN vs Random Init', color='green')
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        axes[1, 0].set_title('Initialization Quality')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Improvement over Random')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 4: Test Initialization Improvement
        if metrics['test_expected_discounted_rewards']:
            test_epochs = [x['epoch'] for x in metrics['test_expected_discounted_rewards']]
            test_init_improvements = [x.get('init_improvement', 0) for x in metrics['test_expected_discounted_rewards']]
            if any(x != 0 for x in test_init_improvements):
                axes[1, 1].plot(test_epochs, test_init_improvements, label='Test Init Improvement', color='purple')
                axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        axes[1, 1].set_title('Test Initialization Improvement')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Improvement')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(dst_dir_path, f'live_training_progress_with_init.png'))
        plt.close()
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to create initialization plot: {e}")