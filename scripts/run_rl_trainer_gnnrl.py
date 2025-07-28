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

# PyTorch Geometric imports
import torch_geometric
from torch_geometric.nn import GINConv, GCNConv, TransformerConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx

import src.qaoa.qaoa_model as qaoa_model
import src.utils.graphs as graph_utils
import src.utils.config as config_utils
import src.solvers.classic_nlopt as classic_nlopt
from src.utils.logger import setup_logger
import src.utils.plot_train as plot_train


class GNNEncoder(nn.Module):
    """Graph Neural Network encoder that processes graph topology with Transformer-style blocks"""
    
    def __init__(self, 
                 node_dim: int = 64,
                 hidden_dim: int = 64,
                 output_dim: int = 64,
                 num_layers: int = 3,
                 gnn_type: str = "GIN",
                 dropout: float = 0.0,
                 num_heads: int = 4,
                 feedforward_dim: int = 256):
        super(GNNEncoder, self).__init__()
        
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.num_heads = num_heads
        
        # Initial node embedding (since our nodes don't have features, we use learnable embeddings)
        self.node_encoder = nn.Sequential(
            nn.Linear(1, node_dim),  # Each node gets a constant feature of 1
            nn.ReLU()
        )
        
        # Build Transformer-style GNN blocks
        self.gnn_blocks = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = node_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            
            # Create a transformer-style block
            block = TransformerGNNBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                gnn_type=gnn_type,
                dropout=dropout,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim
            )
            self.gnn_blocks.append(block)
        
    def forward(self, x, edge_index, batch=None):
        # Initial node features
        x = self.node_encoder(x)
        
        # Apply Transformer-style GNN blocks
        for block in self.gnn_blocks:
            x = block(x, edge_index)
        
        # Global pooling to get graph-level representation
        if batch is not None:
            graph_embedding = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        else:
            # Single graph case
            graph_embedding = torch.cat([
                x.mean(dim=0, keepdim=True),
                x.max(dim=0, keepdim=True)[0]
            ], dim=1)
            
        return graph_embedding, x


class TransformerGNNBlock(nn.Module):
    """Transformer-style block: GNN -> AddNorm -> FeedForward -> AddNorm"""
    
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 gnn_type: str,
                 dropout: float = 0.0,
                 num_heads: int = 4,
                 feedforward_dim: int = 256):
        super(TransformerGNNBlock, self).__init__()
        
        self.gnn_type = gnn_type
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # GNN layer (attention mechanism)
        if gnn_type == "GIN":
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            self.gnn_layer = GINConv(mlp)
            self.gnn_out_dim = out_dim
            
        elif gnn_type == "GCN":
            self.gnn_layer = GCNConv(in_dim, out_dim)
            self.gnn_out_dim = out_dim
            
        elif gnn_type == "TransformerConv":
            # With concat=True and num_heads heads, output dimension is out_dim * num_heads
            self.gnn_layer = TransformerConv(
                in_dim, 
                out_dim, 
                heads=num_heads, 
                concat=True  # Changed to True as requested
            )
            self.gnn_out_dim = out_dim * num_heads
            # Project back to expected dimension
            self.projection = nn.Linear(out_dim * num_heads, out_dim)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Layer normalization for residual connections
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(out_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, out_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Linear projection for residual connection if dimensions don't match
        if in_dim != out_dim:
            self.residual_projection = nn.Linear(in_dim, out_dim)
        else:
            self.residual_projection = None
    
    def forward(self, x, edge_index):
        # Store input for residual connection
        residual = x
        
        # Apply GNN layer (attention mechanism)
        x = self.gnn_layer(x, edge_index)
        
        # Handle TransformerConv concatenated output
        if self.gnn_type == "TransformerConv":
            x = self.projection(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # First Add & Norm (residual connection)
        if self.residual_projection is not None:
            residual = self.residual_projection(residual)
        x = self.norm1(x + residual)
        
        # Store for second residual connection
        residual = x
        
        # Feedforward network
        ff_output = self.feedforward(x)
        
        # Second Add & Norm (residual connection)
        x = self.norm2(ff_output + residual)
        
        return x


class GNNActor(nn.Module):
    """GNN-based Actor that uses graph topology to propose QAOA parameter updates"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 64,
                 gnn_hidden_dim: int = 64,
                 gnn_output_dim: int = 128,
                 gnn_type: str = "GIN",
                 gnn_num_layers: int = 3,
                 dropout: float = 0.0):
        super(GNNActor, self).__init__()
        
        self.action_dim = action_dim
        
        # GNN encoder for processing graph structure
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
        
        # Combine graph and state information
        # Graph embedding is 2 * gnn_output_dim due to concatenation of mean and max pooling
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * gnn_output_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output mean actions
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.ones(action_dim) * -6.0)
        
    def forward(self, state, graph_data):
        """
        Args:
            state: Optimization state tensor (batch_size, state_dim) or (state_dim,)
            graph_data: PyTorch Geometric Data object or Batch object
        """
        # Handle single state case
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_graph = True
        else:
            single_graph = False
            
        # Encode graph structure
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
            
        # Encode optimization state
        state_embedding = self.state_encoder(state)
        
        # Fuse graph and state information
        combined = torch.cat([graph_embedding, state_embedding], dim=1)
        fused = self.fusion_layer(combined)
        
        # Generate action parameters
        mu = 0.1 * self.action_head(fused)
        std = torch.exp(self.log_std)
        
        if single_graph:
            mu = mu.squeeze(0)
            
        return mu, std


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


class QAOAEnvWithGraph(gym.Env):
    """Extended QAOA environment that also provides graph data"""
    
    def __init__(self, graph: nx.Graph, p: int, reps: int = 128, 
                 history_len: int = 4, n_samples_normalization: int = 100):
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


class GNNPPO:
    """PPO agent with GNN-based actor"""
    
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
        
        # GNN-based actor
        self.actor = GNNActor(
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
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gnn_type = gnn_type

    def select_action(self, state, graph_data, deterministic=False):
        state = torch.FloatTensor(state).to(self.device)
        graph_data = graph_data.to(self.device)
        
        mu, std = self.actor(state, graph_data)
        
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

        mu, std = self.actor(state, graph_data)
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
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'clip_epsilon': self.clip_epsilon,
            'gamma': self.gamma,
            'gnn_type': self.gnn_type, 
            'hidden_dim': self.hidden_dim,
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


@torch.no_grad()
def evaluate_on_test_set(agent: GNNPPO, test_graphs: List[Dict[str, Any]], p: int, 
                         T: int = 64, reps: int = 128, L: int = 4, 
                         early_stop_patience: int = 16, 
                         n_samples_normalization: int = 25) -> Dict[str, float]:
    """Evaluate the GNN agent on test set"""
    logger = logging.getLogger(__name__)
    all_rewards = []
    all_final_values = []
    
    for i, elem in enumerate(test_graphs):
        logger.debug(f"Evaluating graph {i+1}/{len(test_graphs)}: {elem['id']}")
        graph = graph_utils.read_graph_from_dict(elem["graph_dict"])
        env = QAOAEnvWithGraph(graph, p=p, reps=reps, history_len=L, 
                               n_samples_normalization=n_samples_normalization)
        state = env.reset()
        episode_reward = 0

        # Track performance for early stopping
        best_f_val = env.f_val
        steps_without_improvement = 0
        
        for t in range(T):
            action, _ = agent.select_action(state, env.graph_data, deterministic=True)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

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
        
        all_rewards.append(episode_reward)
        all_final_values.append(info['f_val'])
    
    return {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_final_value': np.mean(all_final_values),
        'std_final_value': np.std(all_final_values)
    }


def train_gnn_rl_qaoa(GTrain: List[Dict[str, Any]], GTest: List[Dict[str, Any]], 
                      p: int, dst_dir_path: str, gnn_type: str = "GIN",
                      epochs: int = 750, episodes_per_epoch: int = 128, 
                      T: int = 64, graphs_per_episode: int = 50,
                      reps: int = 128, L: int = 4, patience: int = 40, 
                      seed: int = 42, hidden_dim: int = 64, 
                      gnn_hidden_dim: int = 64, gnn_num_layers: int = 3,
                      device: str = "cpu") -> Dict:
    """Train GNN-based RL agent"""
    
    logger = logging.getLogger(__name__)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info(f"Training GNN-based agent with {gnn_type} architecture")
    logger.info(f"Placing computation in DEVICE: {device}")
    
    # Create destination directory
    os.makedirs(dst_dir_path, exist_ok=True)
    
    # Initialize agent
    state_dim = (2 * p + 1) * L
    action_dim = 2 * p
    
    logger.info(f"Creating GNN-PPO agent with state_dim: {state_dim}, action_dim: {action_dim}")
    agent = GNNPPO(
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
        'train_rewards': [],
        'test_evaluations': [],
        'best_test_reward': -float('inf'),
        'best_epoch': 0,
        'gnn_type': gnn_type
    }
    
    patience_counter = 0
    graphs_per_epoch = min(len(GTrain), graphs_per_episode)
    epochs_per_test = 5
    
    try:
        # Training loop
        for epoch in range(epochs):
            epoch_rewards = []
            batch_memory = []
            GTrain_subset = random.sample(GTrain, graphs_per_epoch)
            t_start = time.time()
            
            # Collect trajectories with round-robin
            for ep in range(episodes_per_epoch):
                # Round-robin graph selection
                graph_idx = ep % len(GTrain_subset)
                if ep % 10 == 0:
                    logger.info(f"[Epoch {epoch}] Training on graph {graph_idx}")
                    
                graph = graph_utils.read_graph_from_dict(GTrain_subset[graph_idx]["graph_dict"])
                env = QAOAEnvWithGraph(graph, p=p, reps=reps, history_len=L, 
                                      n_samples_normalization=100)
                
                memory = []
                state = env.reset()
                episode_reward = 0
                
                for t in range(T):
                    action, log_prob = agent.select_action(state, env.graph_data)
                    next_state, reward, done, _ = env.step(action)
                    memory.append((state, action, log_prob, reward, done, env.graph_data))
                    episode_reward += reward
                    state = next_state
                    if done:
                        break
                
                batch_memory.extend(memory)
                epoch_rewards.append(episode_reward)
            
            # Update agent with collected batch
            agent.update(batch_memory)
            
            # Store training metrics
            mean_train_reward = np.mean(epoch_rewards)
            metrics['train_rewards'].append({
                'epoch': epoch,
                'mean_reward': mean_train_reward,
                'std_reward': np.std(epoch_rewards)
            })
            
            logger.info(f"[Epoch {epoch}] took {(time.time() - t_start)/60:.2f} minutes.")
            
            # Evaluate on test set
            if epoch % epochs_per_test == 0:
                GTest_sample = random.sample(GTest, min(len(GTest), graphs_per_episode))
                logger.info(f"Evaluating on test set with {len(GTest_sample)} graphs...")
                t_start_test = time.time()
                test_results = evaluate_on_test_set(agent, GTest_sample, p=p, T=T, 
                                                   reps=reps, L=L)
                logger.info(f"[Epoch {epoch}] Test evaluation took {(time.time() - t_start_test)/60:.2f} minutes.")
                test_results['epoch'] = epoch
                metrics['test_evaluations'].append(test_results)
                
                # Save best model
                if test_results['mean_reward'] > metrics['best_test_reward']:
                    patience_counter = 0
                    metrics['best_test_reward'] = test_results['mean_reward']
                    metrics['best_epoch'] = epoch
                    model_path = os.path.join(dst_dir_path, f'best_model_{gnn_type}.pth')
                    agent.save(model_path)
                    logger.info(f"[Epoch {epoch}] New best model saved!")
                else:
                    patience_counter += epochs_per_test
                
                logger.info(f"[Epoch {epoch}] Train reward: {mean_train_reward:.4f}, "
                           f"Test reward: {test_results['mean_reward']:.4f} ± {test_results['std_reward']:.4f}")
                
                if patience_counter >= patience and epoch > 100:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                try:
                    plot_train.plot_live_progress(metrics, dst_dir_path, p, epoch)
                    # Save metrics periodically
                    metrics_path = os.path.join(dst_dir_path, f'training_metrics_{gnn_type}.json')
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to create live plot: {e}")
            else:
                logger.info(f"[Epoch {epoch}] Train reward: {mean_train_reward:.4f}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving current model...")
    
    # Save final model and metrics
    final_model_path = os.path.join(dst_dir_path, f'final_model_{gnn_type}.pth')
    agent.save(final_model_path)
    
    metrics_path = os.path.join(dst_dir_path, f'training_metrics_{gnn_type}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    try:
        plot_train.plot_training_progress(metrics, dst_dir_path, p)
        plot_train.plot_combined_progress(metrics, dst_dir_path, p)
        logger.info("Final training plots created successfully")
    except Exception as e:
        logger.error(f"Failed to create final plots: {e}")

    return metrics


class GNNQAOAOptimizer:
    """Class for QAOA optimization using trained GNN-based RL agent"""
    
    def __init__(self, model_path: str, p: int, L: int = 4, 
                 gnn_type: str = "GIN", gnn_hidden_dim: int = 64,
                 gnn_num_layers: int = 3, hidden_dim: int = 64, 
                 device: str = "cpu"):
        self.p = p
        self.L = L
        self.state_dim = (2 * p + 1) * L
        self.action_dim = 2 * p
        
        # Load trained agent
        self.agent = GNNPPO(
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
    
    def optimize_rl_only(self, graph: nx.Graph, T: int = 64, 
                        reps: int = 128) -> Tuple[np.ndarray, float]:
        """Optimize using only GNN-based RL agent"""
        env = QAOAEnvWithGraph(graph, self.p, reps=reps, history_len=self.L)
        state = env.reset()
        
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
    
    def optimize_rl_then_nlopt(self, graph: nx.Graph, optimize_qaoa_nlopt, 
                              T_rl: int = 32, reps: int = 128,
                              method: str = 'Nelder-Mead', 
                              in_xtol_rel: float = 1e-4, 
                              in_ftol_abs: float = 1e-3) -> Tuple[np.ndarray, float]:
        """Optimize using GNN-RL for T_rl steps, then continue with NLopt"""
        
        # First use GNN-RL to get to a good region
        env = QAOAEnvWithGraph(graph, self.p, reps=reps, history_len=self.L)
        state = env.reset()
        
        for t in range(T_rl):
            action, _ = self.agent.select_action(state, env.graph_data, deterministic=True)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done:
                break
        
        # Get current parameters from RL
        rl_params = env.params
        gamma_init = rl_params[:self.p]
        beta_init = rl_params[self.p:]
        
        # Continue with NLopt optimization
        final_params, final_opt_value, status_code = classic_nlopt.optimize_qaoa_nlopt(
            graph=graph,
            p=self.p,
            method=method,
            in_xtol_rel=in_xtol_rel,
            in_ftol_abs=in_ftol_abs,
            gamma_init=gamma_init,
            beta_init=beta_init
        )
        
        return final_params, final_opt_value, status_code


def get_parser():
    parser = argparse.ArgumentParser(description="Train GNN-based QAOA RL agent")

    # GNN-specific arguments
    parser.add_argument("--gnn_type", type=str, default="GIN",
                        choices=["GIN", "GCN", "TransformerConv"],
                        help="Type of GNN architecture to use")
    parser.add_argument("--gnn_hidden_dim", type=int, default=64,
                        help="Hidden dimension for GNN layers")
    parser.add_argument("--gnn_num_layers", type=int, default=3,
                        help="Number of GNN layers")
    
    # Original arguments
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to place the computations for torch.")
    parser.add_argument("--graphs_dir", type=str, default="data/optimized_graphs_classic",
                        help="Directory with graph datasets")
    parser.add_argument("--p", type=int, default=1, 
                        help="QAOA depth (number of (γ, β) layers)")
    parser.add_argument("--dst_dir", type=str, default="./outputs/gnn_rl_model",
                        help="Destination directory for checkpoints and metrics")
    parser.add_argument("--n_epochs", type=int, default=600, 
                        help="Number of training epochs")
    parser.add_argument("--eps_per_epoch", type=int, default=128, 
                        help="Number of episodes per epoch")
    parser.add_argument("--graphs_per_episode", type=int, default=50,
                        help="Number of graphs to sample per episode")
    parser.add_argument("--T", type=int, default=64,
                        help="Episode length (time steps per episode)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--patience", type=int, default=50,
                        help="Patience for early stopping")
    parser.add_argument("--hidden_dim", type=int, default=64, 
                        help="Hidden dimension for actor and critic network")
    
    # Logging parameters
    parser.add_argument("--log_filename", type=str, default="gnn_qaoa_solver.log",
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
    
    # Train the model
    p = args.p
    dst_dir = os.path.join(args.dst_dir, args.gnn_type)
    n_epochs = args.n_epochs
    eps_per_epoch = args.eps_per_epoch
    T = args.T
    
    logger.info(f"Initializing GNN-based training with:")
    logger.info(f"  GNN type: {args.gnn_type}")
    logger.info(f"  GNN hidden dim: {args.gnn_hidden_dim}")
    logger.info(f"  GNN layers: {args.gnn_num_layers}")
    logger.info(f"  p: {p}, n_epochs: {n_epochs}, eps_per_epoch: {eps_per_epoch}, T: {T}")
    logger.info(f"  Destination directory: {dst_dir}")
    
    logger.info("Starting GNN-based training...")
    metrics = train_gnn_rl_qaoa(
        GTrain, GTest, p, dst_dir, 
        gnn_type=args.gnn_type,
        epochs=n_epochs, 
        episodes_per_epoch=eps_per_epoch, 
        T=T,
        seed=args.seed, 
        patience=args.patience, 
        graphs_per_episode=args.graphs_per_episode,
        hidden_dim=args.hidden_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_num_layers=args.gnn_num_layers,
        device=args.device
    )