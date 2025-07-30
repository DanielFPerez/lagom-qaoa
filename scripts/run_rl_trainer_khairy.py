import os
import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import networkx as nx
import cirq
from typing import List, Dict, Tuple, Any
from collections import deque
import json
import time
import argparse
import logging
import random

import src.qaoa.qaoa_model as qaoa_model
import src.utils.graphs as graph_utils
import src.utils.config as config_utils
import src.solvers.classic_nlopt as classic_nlopt
from src.utils.logger import setup_logger
import src.utils.plot_train as plot_train



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * -6.0)

    def forward(self, state):
        mu = 0.1 * self.net(state)
        std = torch.exp(self.log_std)
        return mu, std


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)

class QAOAEnv(gym.Env):
    def __init__(self, graph: nx.Graph, p: int, reps: int = 128, history_len: int = 4, n_samples_normalization: int = 100):
        self.graph = graph
        self.p = p
        self.reps = reps
        self.L = history_len
        self.qubits = cirq.LineQubit.range(len(graph.nodes))
        self.param_dim = 2 * p
        self.history = deque(maxlen=self.L)
        self.normalization = qaoa_model.estimate_random_average_energy(graph, p, reps=reps, n_samples=n_samples_normalization)
        self.reset()

    def step(self, action):
        new_params = self.params + action
        f_new = self.eval_qaoa(new_params)
        reward = (f_new - self.f_val) / self.normalization 
        self.update_state(new_params, f_new)
        done = False
        return self.state, reward, done, {'f_val': self.f_val}

    def reset(self):
        beta_params = np.random.uniform(0, np.pi, self.p)
        gamma_params = np.random.uniform(0, 2*np.pi, self.p)
        self.params = np.concatenate([gamma_params, beta_params])
        
        self.f_val = self.eval_qaoa(self.params)
        self.history.clear()
        self.state = self.build_state()
        return self.state

    def update_state(self, new_params, new_f_val):
        delta_f = new_f_val - self.f_val
        delta_params = new_params - self.params
        self.f_val = new_f_val
        self.params = new_params
        self.history.append(np.concatenate(([delta_f], delta_params)))
        self.state = self.build_state()

    def build_state(self):
        state = np.zeros((self.L, self.param_dim + 1))
        for i, h in enumerate(self.history):
            state[i] = h
        return state.flatten()

    def eval_qaoa(self, params):
        gamma = params[:self.p]
        beta = params[self.p:]
        return qaoa_model.eval_circuit(self.graph, self.p, gamma, beta, self.reps)


# Define the PPO agent
class PPO:
    def __init__(self, state_dim, action_dim, clip_epsilon=0.2, gamma=0.99, lr=3e-4, hidden_dim = 64,
                 device: str = "cpu"):
        
        self.device = torch.device(device)
        
        self.actor = Actor(state_dim, action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic = Critic(state_dim, hidden_dim=hidden_dim).to(self.device)
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device)
        mu, std = self.actor(state)
        if deterministic:
            action = mu
            log_prob = torch.zeros(1)
        else:
            dist = Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach().cpu().numpy(), log_prob.detach().cpu().item()

    def evaluate(self, state, action):
        # Accept both numpy/arrays and tensors; ensure float32 on the right device
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device).float()

        if not torch.is_tensor(action):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        else:
            action = action.to(self.device).float()

        mu, std = self.actor(state)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(state).squeeze()
        return log_prob, entropy, value

    def update(self, memory):
        states, actions, log_probs_old, rewards, dones = zip(*memory)
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.FloatTensor(returns).to(self.device)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)

        for _ in range(5):
            log_probs, entropy, values = self.evaluate(states, actions)
            advantages = returns - values.detach()
            ratios = torch.exp(log_probs - log_probs_old)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()
            loss_critic = (returns - values).pow(2).mean()
            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            self.optimizer_critic.step()

    def save(self, path):
        """Save model to disk"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'clip_epsilon': self.clip_epsilon,
            'gamma': self.gamma,
            'hidden_dim': self.hidden_dim,
        }, path)

    def load(self, path):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])


@torch.no_grad()
def evaluate_expected_discounted_rewards(agent: PPO, test_graphs: List[Dict[str, Any]], 
                                       p: int, T: int = 64, reps: int = 128, 
                                       L: int = 4, n_samples_normalization: int = 25) -> Dict[str, float]:
    """Evaluate agent using expected discounted rewards"""
    logger = logging.getLogger(__name__)
    all_discounted_rewards = []
    all_final_values = []
    early_stop_patience = T//2  # Early stopping patience for no improvement
    
    for i, elem in enumerate(test_graphs):
        logger.debug(f"Evaluating graph {i+1}/{len(test_graphs)}: {elem['id']}")
        graph = graph_utils.read_graph_from_dict(elem["graph_dict"])
        env = QAOAEnv(graph, p=p, reps=reps, history_len=L,
                      n_samples_normalization=n_samples_normalization)
        state = env.reset()
        
        # Track rewards for this episode
        episode_rewards = []
        best_f_val = -float('inf')
        steps_without_improvement = 0
        
        for t in range(T):
            action, _ = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            episode_rewards.append(reward)

            if info['f_val'] > best_f_val:
                best_f_val = info['f_val']
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                
            if steps_without_improvement >= early_stop_patience:
                logger.debug(f"Early stopping after {t} steps")
                break

            state = next_state
            if done:
                break
        
        # Compute expected discounted reward
        discounted_reward = 0
        discount_factor = agent.gamma
        for i, r in enumerate(episode_rewards):
            discounted_reward += (discount_factor ** i) * r
            
        all_discounted_rewards.append(discounted_reward)
        all_final_values.append(info['f_val']/env.normalization)
    
    return {
        'mean_discounted_reward': np.mean(all_discounted_rewards),
        'std_discounted_reward': np.std(all_discounted_rewards),
        'mean_final_value': np.mean(all_final_values),
        'std_final_value': np.std(all_final_values)
    }



def train_rl_qaoa(GTrain: List[Dict[str, Any]], GTest: List[Dict[str, Any]], 
                  p: int, dst_dir_path: str, epochs: int = 750, 
                  episodes_per_epoch: int = 128, T: int = 64, graphs_per_episode: int = 50,
                  reps: int = 128, L: int = 4, patience: int = 40, seed: int = 42,
                  hidden_dim: int = 64, device: str = "cpu") -> Dict:
    """Train RL agent with round-robin on training graphs"""

    logger = logging.getLogger(__name__)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info(f"Training Khairy-based agent with parameters: p={p}, epochs={epochs} ")
    logger.info(f"Placing computation in DEVICE: {device}")
    # Create destination directory
    os.makedirs(dst_dir_path, exist_ok=True)
    
    # Initialize agent
    state_dim = (2 * p + 1) * L
    action_dim = 2 * p
    
    logger.info(f"Creating the PPO agent with state_dim: {state_dim}, action_dim: {action_dim}, hidden_dim: {hidden_dim}")
    agent = PPO(state_dim, action_dim, hidden_dim=hidden_dim, device=device)
    
    # Training metrics storage
    # Training metrics storage
    metrics = {
        'train_expected_discounted_rewards': [],  # Changed metric name
        'test_expected_discounted_rewards': [],   # Changed metric name
        'train_mean_final_values': [],           # Track final QAOA values
        'test_mean_final_values': [],            # Track final QAOA values
        'best_test_reward': -float('inf'),
        'best_epoch': 0,
    }

    patience_counter = 0
    graphs_per_epoch = min(len(GTrain), graphs_per_episode)
    epochs_per_test = 5
    
    try:
        # Training loop
        for epoch in range(epochs):
            epoch_discounted_rewards = []  # Store discounted rewards
            epoch_final_values = []         # Store final QAOA values
            batch_memory = []
            GTrain_subset = random.sample(GTrain, graphs_per_epoch)
            t_start = time.time()
            # Collect trajectories with round-robin
            for ep in range(episodes_per_epoch):
                # Round-robin graph selection
                graph_idx = ep % len(GTrain_subset)
                if ep % 10 == 0:
                    logger.info(f"[Epoch {epoch}] Training on graph {graph_idx} with hash {GTrain_subset[graph_idx]['id']}")
                graph = graph_utils.read_graph_from_dict(GTrain_subset[graph_idx]["graph_dict"])
                
                env = QAOAEnv(graph, p=p, reps=reps, history_len=L, n_samples_normalization=100)
                memory = []
                state = env.reset()
                episode_rewards = []
                
                for t in range(T):
                    action, log_prob = agent.select_action(state)
                    next_state, reward, done, info = env.step(action)
                    memory.append((state, action, log_prob, reward, done))
                    episode_rewards.append(reward)
                    state = next_state
                    if done:
                        break
                
                # Compute expected discounted reward for this episode
                discounted_reward = 0
                discount_factor = agent.gamma
                for i, r in enumerate(episode_rewards):
                    discounted_reward += (discount_factor ** i) * r
                
                epoch_discounted_rewards.append(discounted_reward)
                epoch_final_values.append(info['f_val']/env.normalization)
                batch_memory.extend(memory)
            
            # Update agent with collected batch
            agent.update(batch_memory)
            
            # Store training metrics (expected discounted rewards)
            mean_discounted_reward = np.mean(epoch_discounted_rewards)
            mean_final_value = np.mean(epoch_final_values)

            metrics['train_expected_discounted_rewards'].append({
                'epoch': epoch,
                'mean_reward': mean_discounted_reward,
                'std_reward': np.std(epoch_discounted_rewards)
            })
            metrics['train_mean_final_values'].append({
                'epoch': epoch,
                'mean_value': mean_final_value,
                'std_value': np.std(epoch_final_values)
            })
            
            logger.info(f"[Epoch {epoch}] took {(time.time() - t_start)/60:.2f} minutes.")
            
            # Evaluate on test set 
            if epoch % epochs_per_test == 0:
                GTest_sample = random.sample(GTest, min(len(GTest), graphs_per_episode))
                logger.info(f"Evaluating on test set with {len(GTest_sample)} graphs...")

                t_start_test = time.time()
                test_results = evaluate_expected_discounted_rewards(agent, GTest_sample, p=p, T=T, reps=reps, L=L)
                
                logger.info(f"[Epoch {epoch}] Test evaluation took {(time.time() - t_start_test)/60:.2f} minutes.")
                
                test_results['epoch'] = epoch
                metrics['test_expected_discounted_rewards'].append(test_results)
                metrics['test_mean_final_values'].append({
                    'epoch': epoch,
                    'mean_value': test_results['mean_final_value'],
                    'std_value': test_results['std_final_value']
                })
                
                # Save best model
                if test_results['mean_discounted_reward'] > metrics['best_test_reward']:
                    patience_counter = 0
                    metrics['best_test_reward'] = test_results['mean_discounted_reward']
                    metrics['best_epoch'] = epoch
                    model_path = os.path.join(dst_dir_path, f'best_model.pth')
                    agent.save(model_path)
                    logger.info(f"[Epoch {epoch}] New best model saved!")
                else:
                    patience_counter += epochs_per_test
                
                logger.info(f"[Epoch {epoch}] Train EDR: {mean_discounted_reward:.4f}, "
                           f"Test EDR: {test_results['mean_discounted_reward']:.4f} ± {test_results['std_discounted_reward']:.4f}")
                
                if patience_counter >= patience and epoch > 100:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                try:
                    # Modified plotting to show expected discounted rewards
                    plot_train.plot_expected_discounted_rewards(metrics, dst_dir_path, p, epoch)
                    
                    # Save metrics periodically
                    metrics_path = os.path.join(dst_dir_path, f'training_metrics_{gnn_type}.json')
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to create live plot at epoch {epoch}: {e}")

            else:
                logger.info(f"[Epoch {epoch}] Train EDR: {mean_discounted_reward:.4f}")
    
    # catch keyboard interrupt to save final model
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving current model...")
    
    # Save final model and metrics
    final_model_path = os.path.join(dst_dir_path, 'final_model.pth')
    agent.save(final_model_path)
    
    metrics_path = os.path.join(dst_dir_path, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    try:
        plot_train.plot_training_progress(metrics, dst_dir_path, p)
        plot_train.plot_combined_progress(metrics, dst_dir_path, p)
        logger.info("Final training plots created successfully")
    except Exception as e:
        logger.error(f"Failed to create final plots: {e}")

    return metrics


class QAOAOptimizer:
    """Class for QAOA optimization using trained RL agent"""
    
    def __init__(self, model_path: str, p: int, L: int = 4, hidden_dim = 64, device: str = "cpu"):
        self.p = p
        self.L = L
        self.state_dim = (2 * p + 1) * L
        self.action_dim = 2 * p
        
        # Load trained agent
        self.agent = PPO(self.state_dim, self.action_dim, hidden_dim=hidden_dim, device=device)
        self.agent.load(model_path)
        self.agent.actor.eval()
        self.agent.critic.eval()
    
    def optimize_rl_only(self, graph: nx.Graph, T: int = 64, 
                        reps: int = 128) -> Tuple[np.ndarray, float]:
        """Optimize using only RL agent"""
        env = QAOAEnv(graph, self.p, reps=reps, history_len=self.L)
        state = env.reset()
        
        best_params = env.params.copy()
        best_value = env.f_val
        
        for t in range(T):
            action, _ = self.agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            
            if info['f_val'] > best_value:
                best_value = info['f_val']
                best_params = env.params.copy()
            
            state = next_state
            if done:
                break
        
        return best_params, best_value
    
    def optimize_rl_then_nlopt(self, graph: nx.Graph, 
                              T_rl: int = 32, reps: int = 128,
                              method: str = 'Nelder-Mead', 
                              in_xtol_rel: float = 1e-4, 
                              in_ftol_abs: float = 1e-3) -> Tuple[np.ndarray, float]:
        """Optimize using RL for T_rl steps, then continue with NLopt"""
        
        # First use RL to get to a good region
        env = QAOAEnv(graph, self.p, reps=reps, history_len=self.L)
        state = env.reset()
        
        for t in range(T_rl):
            action, _ = self.agent.select_action(state, deterministic=True)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done:
                break
        
        # Get current parameters from RL
        rl_params = env.params
        gamma_init = rl_params[:self.p]
        beta_init = rl_params[self.p:]
        # print(f"Initial parameters found: {rl_params}")
        # Continue with NLopt optimization
        final_params, final_opt_value, status_code  = classic_nlopt.optimize_qaoa_nlopt(
            graph=graph,
            p=self.p,
            method=method,
            in_xtol_rel=in_xtol_rel,
            in_ftol_abs=in_ftol_abs,
            gamma_init=gamma_init,
            beta_init=beta_init,
            enforce_bouds=False 
        )
        
        return final_params, final_opt_value, status_code



def get_parser():

    parser = argparse.ArgumentParser(description="Train QAOA RL agent")

    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to place the computations for torch.")
    parser.add_argument("--graphs_dir",type=str, default="data/optimized_graphs_classic",
                        help="Directory with graph datasets (expects train_results.json / test_results.json).")
    parser.add_argument("--p", type=int, default=1, 
                        help="QAOA depth (number of (γ, β) layers).")
    parser.add_argument("--dst_dir", type=str, default="./outputs/rl_model_khairy",
                        help="Destination directory for checkpoints and metrics.",)
    parser.add_argument("--n_epochs", type=int, default=600, 
                        help="Number of training epochs.")
    parser.add_argument("--eps_per_epoch", type=int, default=128, 
                        help="Number of episodes per epoch.")
    parser.add_argument("--graphs_per_episode", type=int, default=50,
                        help="Number of graphs to sample per episode (round-robin).")
    parser.add_argument("--T", type=int , default=64,
                        help="Episode length (time steps per episode).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--patience", type=int, default=40,
                        help="Patience for early stopping (number of epochs without improvement).")
    parser.add_argument("--hidden_dim", type=int, default=64, 
                        help="Hidden dimension for actor and critic network.")
    
    # Logging parameters
    parser.add_argument("--log_filename", type=str, default="qaoa_solver.log",
                        help="Name of the log file")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--log_format", type=str, default="", 
                        help="Format for log messages.")
    return parser

# Example usage:
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    log_path = os.path.join(config_utils.get_project_root(), f"outputs/logs/{args.log_filename}")

    # log level from arguments
    log_level = getattr(logging, args.log_level, logging.INFO)

    # Setup logging with arguments
    setup_logger(log_file_path=log_path, log_format=args.log_format,
                 console_level=log_level, 
                 file_level=log_level)
    
    logger = logging.getLogger(__name__)

    src_read_dir = os.path.join(config_utils.get_project_root(), args.graphs_dir)

    GTrain = graph_utils.open_merged_graph_json(os.path.join(src_read_dir, "train_results.json"))
    logger.info(f"Training set has {len(GTrain)} instances")

    GTest = graph_utils.open_merged_graph_json(os.path.join(src_read_dir, "test_results.json"))
    logger.info(f"Test set has {len(GTest)} instances")
    
    # Train the model
    p = args.p
    dst_dir = args.dst_dir
    n_epochs = args.n_epochs
    eps_per_epoch = args.eps_per_epoch
    T = args.T
    
    logger.info(f"Initializing training with parameters p: {p}, n_epochs: {n_epochs} eps_per_epoch: {eps_per_epoch}, T: {T}")
    logger.info(f"Destination directory: {dst_dir}")
    
    logger.info("Starting training...")
    metrics = train_rl_qaoa(GTrain, GTest, p, dst_dir, epochs=n_epochs, episodes_per_epoch=eps_per_epoch, T=T, 
                            seed=args.seed, patience=args.patience, graphs_per_episode=args.graphs_per_episode,
                            hidden_dim=args.hidden_dim,
                            device=args.device) 
    
    # Load and use the trained model
    # logger.info("\nLoading best model for inference...")
    # model_path = os.path.join(dst_dir, 'best_model.pth')
    # optimizer = QAOAOptimizer(model_path, p, hidden_dim=args.hidden_dim)
    
    # # Example: optimize a new graph
    # test_graph = nx.erdos_renyi_graph(10, 0.5)
    
    # # RL-only optimization
    # params_rl, value_rl = optimizer.optimize_rl_only(test_graph)
    # logger.info(f"\nRL-only optimization: value = {value_rl:.4f}")
    
    # # RL + NLopt optimization (uncomment when you have optimize_qaoa_nlopt)
    # params_hybrid, value_hybrid, status_hybrid = optimizer.optimize_rl_then_nlopt(
    #     test_graph, 
    #     optimize_qaoa_nlopt,
    #     T_rl=32
    # )
    # logger.info(f"RL+NLopt optimization: value = {value_hybrid:.4f}")



# @torch.no_grad()
# def evaluate_on_test_set(agent: PPO, test_graphs: List[Dict[str, Any]], p: int, 
#                          T: int = 64, reps: int = 128, L: int = 4, early_stop_patience: int = 16, n_samples_normalization: int = 25) -> Dict[str, float]:
#     """Evaluate the agent on test set"""
#     all_rewards = []
#     all_final_values = []
    
#     for i, elem in enumerate(test_graphs):
#         logger.debug(f"Evaluating graph {i+1}/{len(test_graphs)}: {elem['id']}")
#         graph = graph_utils.read_graph_from_dict(elem["graph_dict"])
#         env = QAOAEnv(graph, p=p, reps=reps, history_len=L, n_samples_normalization=n_samples_normalization)
#         state = env.reset()
#         episode_reward = 0

#         # Track performance for early stopping
#         best_f_val = -float('inf')
#         steps_without_improvement = 0
        
#         for t in range(T):
#             action, _ = agent.select_action(state, deterministic=True)
#             next_state, reward, done, info = env.step(action)
#             episode_reward += reward

#             if info['f_val'] > best_f_val:
#                 best_f_val = info['f_val']
#                 steps_without_improvement = 0
#             else:
#                 steps_without_improvement += 1
                
#             if steps_without_improvement >= early_stop_patience:
#                 logger.debug(f"Early stopping after {t} steps due to {steps_without_improvement} steps without improvement.")
#                 break

#             state = next_state
#             if done:
#                 break
        
#         all_rewards.append(episode_reward)
#         all_final_values.append(info['f_val'])
    
#     return {
#         'mean_reward': np.mean(all_rewards),
#         'std_reward': np.std(all_rewards),
#         'mean_final_value': np.mean(all_final_values),
#         'std_final_value': np.std(all_final_values)
#     }