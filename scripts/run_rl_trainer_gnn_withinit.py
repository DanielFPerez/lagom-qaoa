import os
import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import networkx as nx
import cirq
from typing import List, Dict, Tuple, Any, Optional
from collections import deque
import json
import time
import argparse
import logging
import random

from torch_geometric.data import Data, Batch

import src.qaoa.qaoa_model as qaoa_model
import src.utils.graphs as graph_utils
import src.utils.config as config_utils
import src.solvers.classic_nlopt as classic_nlopt
from src.utils.logger import setup_logger
import src.utils.plot_train as plot_train

from src.gnns.encoder import GNNEncoder


class GNNActorWithInit(nn.Module):
    """Enhanced GNN-based Actor that handles both initialization and parameter updates"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 64,
                 gnn_hidden_dim: int = 64,
                 gnn_output_dim: int = 128,
                 gnn_type: str = "GIN",
                 gnn_num_layers: int = 3,
                 dropout: float = 0.0):
        super(GNNActorWithInit, self).__init__()
        
        self.action_dim = action_dim
        self.p = action_dim // 2  # Number of QAOA layers
        self.state_dim = state_dim
        
        # GNN encoder for processing graph structure (shared)
        self.gnn_encoder = GNNEncoder(
            node_dim=gnn_hidden_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            num_layers=gnn_num_layers,
            gnn_type=gnn_type,
            dropout=dropout
        )
        
        # Process optimization state (history of parameter changes)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Combine graph and state information for action updates
        # Graph embedding is 2 * gnn_output_dim due to concatenation of mean and max pooling
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * gnn_output_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output mean actions (parameter updates)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Initialization head that only uses graph embedding
        self.init_head = nn.Sequential(
            nn.Linear(2 * gnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Learnable log standard deviation for actions
        self.log_std = nn.Parameter(torch.ones(action_dim) * -6.0)
        
        # Learnable log standard deviation for initialization (higher variance)
        self.init_log_std = nn.Parameter(torch.ones(action_dim) * -2.0)
        
    def forward(self, state, graph_data, mode='action'):
        """
        Args:
            state: Optimization state tensor (batch_size, state_dim) or (state_dim,)
            graph_data: PyTorch Geometric Data object or Batch object
            mode: 'action' for parameter updates, 'init' for initial parameters
        """
        # Handle single state case
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_graph = True
        else:
            single_graph = False
            
        # Encode graph structure (shared encoder)
        if hasattr(graph_data, 'batch'):
            # Batched graphs
            graph_embedding, node_embeddings = self.gnn_encoder(
                graph_data.x, 
                graph_data.edge_index,
                graph_data.batch
            )
        else:
            # Single graph
            graph_embedding, node_embeddings = self.gnn_encoder(
                graph_data.x,
                graph_data.edge_index
            )
        
        if mode == 'init':
            # For initialization, only use graph embedding
            init_params = self.init_head(graph_embedding)
            
            # Apply domain-specific transformations
            # Split into gamma and beta parameters
            gamma_params = init_params[:, :self.p]
            beta_params = init_params[:, self.p:]
            
            # Apply appropriate ranges: gamma in [0, 2π], beta in [0, π]
            gamma_params = torch.sigmoid(gamma_params) * 2 * np.pi
            beta_params = torch.sigmoid(beta_params) * np.pi
            
            mu = torch.cat([gamma_params, beta_params], dim=1)
            std = torch.exp(self.init_log_std)
            
            if single_graph:
                mu = mu.squeeze(0)
                
            return mu, std
            
        elif mode == 'action':
            # For actions, use both graph and state information
            # Encode optimization state
            state_embedding = self.state_encoder(state)
            
            # Fuse graph and state information
            combined = torch.cat([graph_embedding, state_embedding], dim=1)
            fused = self.fusion_layer(combined)
            
            # Generate action parameters (small updates)
            mu = 0.1 * self.action_head(fused)
            std = torch.exp(self.log_std)
            
            if single_graph:
                mu = mu.squeeze(0)
                
            return mu, std
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def get_initial_parameters(self, graph_data, deterministic=False):
        """Get initial parameters for a graph"""
        with torch.no_grad():
            # Create dummy state (not used for initialization)
            device = next(self.parameters()).device
            dummy_state = torch.zeros(1, self.state_dim, dtype=torch.float32, device=device)
            # dummy_state = torch.zeros(1, dtype=torch.float32, device=device)
            
            mu, std = self.forward(dummy_state, graph_data, mode='init')
            
            if deterministic:
                return mu.detach().cpu().numpy()
            else:
                dist = Normal(mu, std)
                sample = dist.sample()
                return sample.detach().cpu().numpy()
            

class Critic(nn.Module):
    """Standard Critic network (unchanged from original)"""
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
    

# #### QAOA Environment with GNN Initialization ####
class QAOAEnvWithGraphInitGNN(gym.Env):
    """Extended QAOA environment that supports GNN-based initialization"""
    
    def __init__(self, graph: nx.Graph, p: int, reps: int = 128, 
                 history_len: int = 4, n_samples_normalization: int = 100,
                 gnn_actor=None):
        self.graph = graph
        self.p = p
        self.reps = reps
        self.L = history_len
        self.qubits = cirq.LineQubit.range(len(graph.nodes))
        self.param_dim = 2 * p
        self.history = deque(maxlen=self.L)
        self.normalization = qaoa_model.estimate_random_average_energy(
            graph, p, reps=reps, n_samples=n_samples_normalization
        )
        self.best_value_seen = -float('inf')
        
        # NEW: Optional GNN actor for initialization
        self.gnn_actor = gnn_actor
        
        # Convert NetworkX graph to PyTorch Geometric format
        self.graph_data = self._create_graph_data()
        
        self.reset()
    
    def _create_graph_data(self):
        """Convert NetworkX graph to PyTorch Geometric Data object"""
        # Create node features (just constant 1 for each node)
        num_nodes = len(self.graph.nodes)
        x = torch.ones((num_nodes, 1), dtype=torch.float32)
        
        # Convert edges
        edge_list = list(self.graph.edges())
        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            # Add reverse edges for undirected graph
            edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create edge weights if they exist
        edge_attr = None
        if nx.get_edge_attributes(self.graph, 'weight'):
            weights = []
            for u, v in edge_list:
                weights.append(self.graph[u][v].get('weight', 1.0))
                weights.append(self.graph[u][v].get('weight', 1.0))  # Reverse edge
            edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def reset(self, use_gnn_init=True):
        """Reset environment with optional GNN-based initialization"""
        if use_gnn_init and self.gnn_actor is not None:
            # Use GNN to propose initial parameters
            try:
                self.params = self.gnn_actor.get_initial_parameters(
                    self.graph_data, deterministic=False
                )
                # Ensure params is 1D numpy array
                if len(self.params.shape) > 1:
                    self.params = self.params.squeeze()
            except Exception as e:
                # Fallback to random initialization if GNN fails
                self.params = self._random_initialization()
        else:
            # Use random initialization
            self.params = self._random_initialization()
        
        self.f_val = self.eval_qaoa(self.params)
        self.best_value_seen = self.f_val  # Reset best value
        self.history.clear()
        self.state = self.build_state()
        return self.state
    
    def _random_initialization(self):
        """Original random initialization method"""
        beta_params = np.random.uniform(0, np.pi, self.p)
        gamma_params = np.random.uniform(0, 2*np.pi, self.p)
        return np.concatenate([gamma_params, beta_params])

    def step(self, action):
        new_params = self.params + action
        f_new = self.eval_qaoa(new_params)
        # Improved reward: combination of improvement and distance to best
        improvement = f_new - self.f_val
        normalized_improvement = improvement / (abs(self.normalization) + 1e-8)
        # Bonus for reaching new best
        best_bonus = 0.1 if f_new > self.best_value_seen * 0.99 else 0
        # update best value after checking for bonus
        self.best_value_seen = max(self.best_value_seen, f_new)
        reward = normalized_improvement + best_bonus
        self.update_state(new_params, f_new)
        done = False
        return self.state, reward, done, {'f_val': self.f_val}

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
    


# #### GNN-based PPO Agent with Initialization Support ####
class GNNPPOWithInit:
    """Enhanced PPO agent with GNN-based actor supporting initialization"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 clip_epsilon: float = 0.2,
                 gamma: float = 0.99,
                 lr: float = 3e-4,
                 hidden_dim: int = 64,
                 gnn_type: str = "GIN",
                 gnn_hidden_dim: int = 64,
                 gnn_num_layers: int = 3,
                 device: str = "cpu"):
        
        self.device = torch.device(device)
        
        # Enhanced GNN-based actor with initialization capability
        self.actor = GNNActorWithInit(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_hidden_dim,
            gnn_type=gnn_type,
            gnn_num_layers=gnn_num_layers
        ).to(self.device)

        self.hidden_dim = hidden_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_num_layers = gnn_num_layers
        
        # Standard critic (doesn't need graph information)
        self.critic = Critic(state_dim, hidden_dim=hidden_dim).to(self.device)
        
        # self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_actor = optim.Adam(
            list(self.actor.gnn_encoder.parameters()) + 
            list(self.actor.state_encoder.parameters()) +
            list(self.actor.fusion_layer.parameters()) +
            list(self.actor.action_head.parameters()) +
            [self.actor.log_std], 
            lr=lr
        )

        # Separate optimizer for initialization components
        self.optimizer_init = optim.Adam(
            list(self.actor.init_head.parameters()) + 
            [self.actor.init_log_std],
            lr=5e-4  # Higher learning rate for initialization head
        )

        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gnn_type = gnn_type

    def select_action(self, state, graph_data, deterministic=False):
        state = torch.FloatTensor(state).to(self.device)
        graph_data = graph_data.to(self.device)
        
        # Use 'action' mode for parameter updates
        mu, std = self.actor(state, graph_data, mode='action')
        
        if deterministic:
            action = mu
            log_prob = torch.zeros(1)
        else:
            dist = Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
        return action.detach().cpu().numpy(), log_prob.detach().cpu().item()

    def evaluate(self, state, action, graph_data):
        # Convert inputs to tensors on the right device
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device).float()

        if not torch.is_tensor(action):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        else:
            action = action.to(self.device).float()

        # Move graph data to device
        graph_data = graph_data.to(self.device)

        # Use 'action' mode for evaluation
        mu, std = self.actor(state, graph_data, mode='action')
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(state).squeeze()
        
        return log_prob, entropy, value

    def update(self, memory):
        states, actions, log_probs_old, rewards, dones, graph_datas = zip(*memory)
        
        # Compute returns
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
        
        # Batch graph data
        graph_batch = Batch.from_data_list(graph_datas).to(self.device)

        # PPO update
        for _ in range(5):
            log_probs, entropy, values = self.evaluate(states, actions, graph_batch)
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
            'optimizer_init_state_dict': self.optimizer_init.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'clip_epsilon': self.clip_epsilon,
            'gamma': self.gamma,
            'hidden_dim': self.hidden_dim,
            'gnn_type': self.gnn_type, 
            'gnn_hidden_dim': self.gnn_hidden_dim,
            'gnn_num_layers': self.gnn_num_layers,
        }, path)

    def load(self, path):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        self.optimizer_init.load_state_dict(checkpoint['optimizer_init_state_dict'])
        self.state_dim = checkpoint['state_dim']


# #### Training Function with Initialization Support ####
def train_gnn_rl_qaoa_with_init(GTrain: List[Dict[str, Any]], GTest: List[Dict[str, Any]], 
                                p: int, dst_dir_path: str, gnn_type: str = "GIN",
                                epochs: int = 750, episodes_per_epoch: int = 128, 
                                T: int = 64, in_graphs_per_epoch: int = 50,
                                reps: int = 128, L: int = 4, patience: int = 40, 
                                seed: int = 42, hidden_dim: int = 64, 
                                gnn_hidden_dim: int = 64, gnn_num_layers: int = 3,
                                device: str = "cpu", 
                                training_strategy: str = "joint",
                                init_training_epochs: int = 50,
                                evalinit: bool = False) -> Dict:
    """Train GNN-based RL agent with initialization capability"""
    logger = logging.getLogger(__name__)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info(f"Training GNN-based agent with {gnn_type} architecture and {training_strategy} strategy")
    logger.info(f"Placing computation in DEVICE: {device}")
    
    # Create destination directory
    os.makedirs(dst_dir_path, exist_ok=True)
    
    # Initialize agent
    state_dim = (2 * p + 1) * L
    action_dim = 2 * p
    
    logger.info(f"Creating enhanced GNN-PPO agent with state_dim: {state_dim}, action_dim: {action_dim}")
    agent = GNNPPOWithInit(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        gnn_type=gnn_type,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_num_layers=gnn_num_layers,
        device=device
    )
    
    # Training metrics storage
    metrics = {
        'train_expected_discounted_rewards': [],
        'test_expected_discounted_rewards': [],
        'train_mean_final_values': [],
        'test_mean_final_values': [],
        'initialization_quality': [],  # Track initialization improvements
        'best_test_reward': -float('inf'),
        'best_epoch': 0,
        'gnn_type': gnn_type,
        'training_strategy': training_strategy
    }
    
    patience_counter = 0
    graphs_per_epoch = min(len(GTrain), in_graphs_per_epoch)
    graphs_per_epoch_test = graphs_per_epoch // 2
    epochs_per_test = 5

    logger.info(f"Training with {graphs_per_epoch} training graphs and {graphs_per_epoch_test} test graphs PER EPOCH.")

    # Stage 1: Pre-train initialization (if using staged strategy)
    if training_strategy == "staged":
        raise NotImplementedError("Pre-training initialization is not implemented yet.")
    
    try:
        # Main training loop
        for epoch in range(epochs):
            epoch_discounted_rewards = []
            epoch_final_values = []
            epoch_init_improvements = []  # Track initialization quality
            batch_memory = []
            GTrain_subset = random.sample(GTrain, graphs_per_epoch)
            t_start = time.time()
            
            # Collect trajectories
            for ep in range(episodes_per_epoch):
                # Round-robin graph selection
                graph_idx = ep % len(GTrain_subset)
                if ep % 10 == 0:
                    logger.info(f"[Epoch {epoch}] Training on graph {graph_idx}")
                    
                graph = graph_utils.read_graph_from_dict(GTrain_subset[graph_idx]["graph_dict"])
                # Create environment with GNN actor for initialization
                env = QAOAEnvWithGraphInitGNN(graph, p=p, reps=reps, history_len=L, 
                                              n_samples_normalization=100, gnn_actor=agent.actor)
                
                # Track initialization quality
                if evalinit and training_strategy == "joint" and epoch % 5 == 0:
                    init_improvement = evaluate_single_graph_init_quality(agent.actor, graph, p, reps, L)
                    epoch_init_improvements.append(init_improvement)
                
                memory = []
                state = env.reset(use_gnn_init=True)  # Use GNN initialization
                episode_rewards = []
                
                for t in range(T):
                    action, log_prob = agent.select_action(state, env.graph_data)
                    next_state, reward, done, info = env.step(action)
                    memory.append((state, action, log_prob, reward, done, env.graph_data))
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
            
            # Additional initialization training for joint strategy
            if training_strategy == "joint" and epoch % 2 == 0:
                init_loss, init_mean_improvement = train_initialization_step(agent, GTrain_subset, p, reps//2, L)
                logger.info(f"[Epoch {epoch}] Init-params training - Loss: {init_loss:.4f}, Mean improvement: {init_mean_improvement:.4f}")
                # Optionally store in metrics
                if 'init_training_loss' not in metrics:
                    metrics['init_training_loss'] = []
                metrics['init_training_loss'].append({
                    'epoch': epoch,
                    'loss': init_loss,
                    'improvement': init_mean_improvement
                })
            
            # Store training metrics
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
            
            # Store initialization quality metrics
            if evalinit and epoch_init_improvements:
                metrics['initialization_quality'].append({
                    'epoch': epoch,
                    'mean_improvement': np.mean(epoch_init_improvements),
                    'std_improvement': np.std(epoch_init_improvements)
                })
            
            logger.info(f"[Epoch {epoch}] took {(time.time() - t_start)/60:.2f} minutes.")
            
            # Evaluate on test set
            if epoch % epochs_per_test == 0:
                GTest_sample = random.sample(GTest, min(len(GTest), graphs_per_epoch_test))
                logger.info(f"Evaluating on test set with {len(GTest_sample)} graphs...")

                t_start_test = time.time()
                test_results = evaluate_expected_discounted_rewards_with_init(
                    agent, GTest_sample, p=p, T=T, reps=reps//2, L=L, evalinit=evalinit, n_samples_normalization=25
                )

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
                    model_path = os.path.join(dst_dir_path, f'best_model_{gnn_type}_with_init.pth')
                    agent.save(model_path)
                    logger.info(f"[Epoch {epoch}] New best model saved!")
                else:
                    patience_counter += epochs_per_test
                
                logger.info(f"[Epoch {epoch}] Train EDR: {mean_discounted_reward:.4f}, "
                           f"Test EDR: {test_results['mean_discounted_reward']:.4f} ± {test_results['std_discounted_reward']:.4f}")
                
                if evalinit and 'init_improvement' in test_results:
                    logger.info(f"[Epoch {epoch}] Init-params improvement: {test_results['init_improvement']:.4f}")
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                try:
                    # Plot training progress including initialization quality
                    if evalinit:
                        plot_train.plot_training_with_init(metrics, dst_dir_path, p, epoch)
                    else: 
                        plot_train.plot_expected_discounted_rewards(metrics, dst_dir_path, p, epoch)
                    
                    # Save metrics periodically
                    metrics_path = os.path.join(dst_dir_path, f'training_metrics_{gnn_type}_with_init.json')
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to create live plot at epoch {epoch}: {e}")
            else:
                logger.info(f"[Epoch {epoch}] Train EDR: {mean_discounted_reward:.4f}")
    
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving current model...")
    
    # Save final model and metrics
    final_model_path = os.path.join(dst_dir_path, f'final_model_{gnn_type}_with_init.pth')
    agent.save(final_model_path)
    
    metrics_path = os.path.join(dst_dir_path, f'training_metrics_{gnn_type}_with_init.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    try:
        plot_train.plot_training_progress(metrics, dst_dir_path, p)
        plot_train.plot_combined_progress(metrics, dst_dir_path, p)
        logger.info("Final training plots created successfully")
    except Exception as e:
        logger.error(f"Failed to create final plots: {e}")

    return metrics


def train_initialization_step(agent: GNNPPOWithInit, graph_subset: List[Dict[str, Any]], 
                             p: int, reps: int, L: int):
    """Train initialization using policy gradient with improvement as reward"""
    logger = logging.getLogger(__name__)
    logger.info("Training GNN initialization with policy gradient...")
    improvements = []
    log_probs = []

    # Create proper dummy state with correct dimensions
    state_dim = (2 * p + 1) * L
    t_start = time.time()
    for graph_data in random.sample(graph_subset, min(50, len(graph_subset))):
        graph = graph_utils.read_graph_from_dict(graph_data["graph_dict"])
        
        # Get GNN initialization with log probability
        env = QAOAEnvWithGraphInitGNN(graph, p=p, reps=reps, history_len=L, 
                                      gnn_actor=agent.actor)
        
        # Get initial parameters from GNN (with gradient tracking)
        dummy_state = torch.zeros(1, state_dim, dtype=torch.float32, device=agent.device)
        mu, std = agent.actor(dummy_state, env.graph_data.to(agent.device), mode='init')
        dist = Normal(mu, std)
        init_params = dist.sample()
        log_prob = dist.log_prob(init_params).sum()

        # Evaluate GNN initialization
        init_params_numpy = init_params.squeeze(0).detach().cpu().numpy()
        gnn_value = env.eval_qaoa(init_params_numpy)
        
        # Evaluate random initialization for comparison
        random_params = env._random_initialization()
        random_value = env.eval_qaoa(random_params)
        
        improvement = (gnn_value - random_value) / (abs(env.normalization) + 1e-8)
        
        improvements.append(improvement)
        log_probs.append(log_prob)
    
    if improvements:
        # Convert to tensors
        improvements_tensor = torch.tensor(improvements, device=agent.device)
        log_probs_tensor = torch.stack(log_probs)
        
        # Normalize advantages (improvements)
        advantages = (improvements_tensor - improvements_tensor.mean()) / (improvements_tensor.std() + 1e-8)
        
        # Policy gradient loss: -log_prob * advantage
        loss = -(log_probs_tensor * advantages).mean()
        
        agent.optimizer_init.zero_grad()
        loss.backward()
        agent.optimizer_init.step()
        logger.info(f"Initialization training step completed in {(time.time() - t_start)/60:.2f} minutes.")
        
        return loss.item(), improvements_tensor.mean().item()
    else:
        return 0.0, 0.0


def evaluate_single_graph_init_quality(actor, graph: nx.Graph, p: int, 
                                      reps: int, L: int) -> float:
    """Evaluate initialization quality for a single graph"""
    
    # Random initialization
    env_random = QAOAEnvWithGraphInitGNN(graph, p=p, reps=reps, history_len=L)
    env_random.reset(use_gnn_init=False)
    random_value = env_random.f_val
    
    # GNN initialization
    env_gnn = QAOAEnvWithGraphInitGNN(graph, p=p, reps=reps, history_len=L, gnn_actor=actor)
    env_gnn.reset(use_gnn_init=True)
    gnn_value = env_gnn.f_val
    
    # Return normalized improvement
    improvement = (gnn_value - random_value) / (abs(env_random.normalization) + 1e-8)
    return improvement


@torch.no_grad()
def evaluate_expected_discounted_rewards_with_init(agent: GNNPPOWithInit, test_graphs: List[Dict[str, Any]], 
                                                  p: int, T: int = 64, reps: int = 128, 
                                                  L: int = 4, n_samples_normalization: int = 50,
                                                  evalinit: bool = False) -> Dict[str, float]:
    """Evaluate agent with initialization quality tracking"""
    logger = logging.getLogger(__name__)
    all_discounted_rewards = []
    all_final_values = []
    all_init_improvements = []
    early_stop_patience = T//2
    
    for i, elem in enumerate(test_graphs):
        logger.debug(f"Evaluating graph {i+1}/{len(test_graphs)}: {elem['id']}")
        graph = graph_utils.read_graph_from_dict(elem["graph_dict"])
        
        # Evaluate initialization quality
        # if evalinit and i % 5 == 0:
        #     init_improvement = evaluate_single_graph_init_quality(agent.actor, graph, p, reps//2, L)
        #     all_init_improvements.append(init_improvement)
        
        # Run optimization with GNN initialization
        env = QAOAEnvWithGraphInitGNN(graph, p=p, reps=reps, history_len=L, 
                                      n_samples_normalization=n_samples_normalization,
                                      gnn_actor=agent.actor)
        state = env.reset(use_gnn_init=True)
        
        # Track rewards for this episode
        episode_rewards = []
        best_f_val = -float('inf')
        steps_without_improvement = 0
        
        for t in range(T):
            action, _ = agent.select_action(state, env.graph_data, deterministic=True)
            next_state, reward, done, info = env.step(action)
            episode_rewards.append(reward)

            if info['f_val'] > best_f_val:
                best_f_val = info['f_val']
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                
            if steps_without_improvement >= early_stop_patience:
                logger.info(f"Early stopping for graph {i+1}/{len(test_graphs)} at step {t+1}")
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
    
    results = {
        'mean_discounted_reward': np.mean(all_discounted_rewards),
        'std_discounted_reward': np.std(all_discounted_rewards),
        'mean_final_value': np.mean(all_final_values),
        'std_final_value': np.std(all_final_values),
        'init_improvement': np.mean(all_init_improvements) if len(all_init_improvements) > 0 else 0,
        'init_improvement_std': np.std(all_init_improvements) if len(all_init_improvements) > 0 else 0
    }
    
    return results



# Enhanced GNNQAOAOptimizer with initialization support
class GNNQAOAOptimizer:
    """Enhanced QAOA optimizer using trained GNN-based RL agent with initialization"""
    
    def __init__(self, model_path: str, p: int, L: int = 4, 
                 gnn_type: str = "GIN", gnn_hidden_dim: int = 64,
                 gnn_num_layers: int = 3, hidden_dim: int = 64, 
                 device: str = "cpu"):
        self.p = p
        self.L = L
        self.state_dim = (2 * p + 1) * L
        self.action_dim = 2 * p
        
        # Load trained agent with initialization capability
        self.agent = GNNPPOWithInit(
            self.state_dim, 
            self.action_dim,
            hidden_dim=hidden_dim,
            gnn_type=gnn_type,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_layers=gnn_num_layers,
            device=device
        )
        self.agent.load(model_path)
        self.agent.actor.eval()
        self.agent.critic.eval()
    
    def optimize_rl_with_gnn_init(self, graph: nx.Graph, T: int = 64, 
                                 reps: int = 128) -> Tuple[np.ndarray, float]:
        """Optimize using GNN initialization + GNN-based RL agent"""
        env = QAOAEnvWithGraphInitGNN(graph, self.p, reps=reps, history_len=self.L,
                               gnn_actor=self.agent.actor)
        state = env.reset(use_gnn_init=True)  # Use GNN initialization
        
        best_params = env.params.copy()
        best_value = env.f_val
        
        for t in range(T):
            action, _ = self.agent.select_action(state, env.graph_data, deterministic=True)
            next_state, reward, done, info = env.step(action)
            
            if info['f_val'] > best_value:
                best_value = info['f_val']
                best_params = env.params.copy()
            
            state = next_state
            if done:
                break
        
        return best_params, best_value
    


def get_parser():
    parser = argparse.ArgumentParser(description="Train GNN-based QAOA RL agent with initialization")

    # GNN-specific arguments
    parser.add_argument("--gnn_type", type=str, default="GIN",
                        choices=["GIN", "GCN", "TCN"],
                        help="Type of GNN architecture to use")
    parser.add_argument("--gnn_hidden_dim", type=int, default=64,
                        help="Hidden dimension for GNN layers")
    parser.add_argument("--gnn_num_layers", type=int, default=3,
                        help="Number of GNN layers")
    
    # NEW: Initialization-specific arguments
    parser.add_argument("--training_strategy", type=str, default="joint",
                        choices=["joint", "staged"],
                        help="Training strategy: 'joint' trains both simultaneously, 'staged' pretrains initialization")
    parser.add_argument("--init_training_epochs", type=int, default=50,
                        help="Number of epochs for initialization pre-training (if using staged strategy)")
    parser.add_argument("--use_gnn_init", action="store_true", default=True,
                        help="Use GNN-based initialization during training")
    parser.add_argument("--evalinit", action="store_true", default=False,
                        help="Evaluate initialization quality on test set after training")
    
    # Original arguments
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to place the computations for torch.")
    parser.add_argument("--graphs_dir", type=str, default="data/optimized_graphs_classic",
                        help="Directory with graph datasets")
    parser.add_argument("--p", type=int, default=1, 
                        help="QAOA depth (number of (γ, β) layers)")
    parser.add_argument("--dst_dir", type=str, default="./outputs/gnn_rl_model_with_init",
                        help="Destination directory for checkpoints and metrics")
    parser.add_argument("--n_epochs", type=int, default=600, 
                        help="Number of training epochs")
    parser.add_argument("--eps_per_epoch", type=int, default=128, 
                        help="Number of episodes per epoch")
    parser.add_argument("--graphs_per_epoch", type=int, default=50,
                        help="Number of graphs to sample per epoch")
    parser.add_argument("--T", type=int, default=64,
                        help="Episode length (time steps per episode)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--patience", type=int, default=50,
                        help="Patience for early stopping")
    parser.add_argument("--hidden_dim", type=int, default=64, 
                        help="Hidden dimension for actor and critic network")
    
    # Logging parameters
    parser.add_argument("--log_filename", type=str, default="gnn_qaoa_solver_with_init.log",
                        help="Name of the log file")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--log_format", type=str, default="", 
                        help="Format for log messages")
    
    return parser


# Example usage:
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    log_path = os.path.join(config_utils.get_project_root(), f"outputs/logs/{args.log_filename}")

    # Log level from arguments
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
    
    # Train the model with initialization support
    p = args.p
    dst_dir = args.dst_dir
    n_epochs = args.n_epochs
    eps_per_epoch = args.eps_per_epoch
    T = args.T
    
    logger.info(f"Initializing enhanced GNN-based training with:")
    logger.info(f"  GNN type: {args.gnn_type}")
    logger.info(f"  GNN hidden dim: {args.gnn_hidden_dim}")
    logger.info(f"  GNN layers: {args.gnn_num_layers}")
    logger.info(f"  Training strategy: {args.training_strategy}")
    logger.info(f"  Use GNN initialization: {args.use_gnn_init}")
    logger.info(f"  p: {p}, n_epochs: {n_epochs}, eps_per_epoch: {eps_per_epoch}, T: {T}")
    logger.info(f"  Destination directory: {dst_dir}")
    
    logger.info("Starting enhanced GNN-based training with initialization...")
    metrics = train_gnn_rl_qaoa_with_init(
        GTrain, GTest, p, dst_dir, 
        gnn_type=args.gnn_type,
        epochs=n_epochs, 
        episodes_per_epoch=eps_per_epoch, 
        T=T,
        seed=args.seed, 
        patience=args.patience, 
        in_graphs_per_epoch=args.graphs_per_epoch,
        hidden_dim=args.hidden_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_num_layers=args.gnn_num_layers,
        device=args.device,
        training_strategy=args.training_strategy,
        init_training_epochs=args.init_training_epochs, 
        evalinit=args.evalinit
    )
    
    logger.info("Training completed successfully!")
    logger.info(f"Best test reward: {metrics['best_test_reward']:.4f} at epoch {metrics['best_epoch']}")
    
    # # Optional: Evaluate initialization quality on test set
    # logger.info("Evaluating final initialization quality...")
    
    # # Load best model for final evaluation
    # best_model_path = os.path.join(dst_dir, f'best_model_{args.gnn_type}_with_init.pth')
    # if os.path.exists(best_model_path):
    #     final_agent = GNNPPOWithInit(
    #         state_dim=(2 * p + 1) * args.L if hasattr(args, 'L') else (2 * p + 1) * 4,
    #         action_dim=2 * p,
    #         hidden_dim=args.hidden_dim,
    #         gnn_type=args.gnn_type,
    #         gnn_hidden_dim=args.gnn_hidden_dim,
    #         gnn_num_layers=args.gnn_num_layers,
    #         device=args.device
    #     )
    #     final_agent.load(best_model_path)
        
    #     # Evaluate initialization quality on a subset of test graphs
    #     test_sample = random.sample(GTest, min(20, len(GTest)))
    #     init_improvements = []
        
    #     for graph_data in test_sample:
    #         graph = graph_utils.read_graph_from_dict(graph_data["graph_dict"])
    #         improvement = evaluate_single_graph_init_quality(
    #             final_agent.actor, graph, p, reps=128, L=4
    #         )
    #         init_improvements.append(improvement)
        
    #     mean_init_improvement = np.mean(init_improvements)
    #     logger.info(f"Final initialization improvement over random: {mean_init_improvement:.4f} ± {np.std(init_improvements):.4f}")
        
    #     # Save final initialization quality results
    #     init_results = {
    #         'mean_improvement': mean_init_improvement,
    #         'std_improvement': np.std(init_improvements),
    #         'all_improvements': init_improvements
    #     }
        
    #     init_results_path = os.path.join(dst_dir, 'final_initialization_quality.json')
    #     with open(init_results_path, 'w') as f:
    #         json.dump(init_results, f, indent=2)
        
    #     logger.info(f"Final initialization quality results saved to {init_results_path}")
    # else:
    #     logger.warning("Best model not found for final evaluation")