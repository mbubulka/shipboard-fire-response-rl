#!/usr/bin/env python3
"""
Enhanced DQN Training Script
Trains DQN agent using comprehensive fire response scenarios
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

from ..core.enhanced_dqn import (
    EnhancedFireResponseEnvironment,
    EnhancedDQNAgent
)


class TrainingManager:
    """Manages the training process for the Enhanced DQN"""
    
    def __init__(self, save_dir: str = "./models"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize environment and agent
        self.env = EnhancedFireResponseEnvironment()
        self.agent = EnhancedDQNAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            lr=2e-4,  # Lower learning rate for stability
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.997,
            epsilon_min=0.05
        )
        
        # Training metrics
        self.training_history = {
            "episode_rewards": [],
            "episode_losses": [],
            "source_performance": {},
            "epsilon_history": [],
            "training_time": []
        }
        
    def train(self, num_episodes: int = 1000, save_interval: int = 200) -> Dict:
        """Train enhanced DQN agent with comprehensive scenarios"""

        print("🔥 Enhanced Shipboard Fire Response DQN Training")
        print("=" * 60)
        print(f"📊 Environment: {self.env.state_dim} states, {self.env.action_dim} actions")
        print(f"🎯 Training scenarios: {len(self.env.scenarios)}")
        print(f"📚 Training sources: {list(self.env.source_map.keys())}")
        print(f"🔄 Training episodes: {num_episodes}")
        print()
        
        # Initialize source performance tracking
        for source in self.env.source_map.keys():
            self.training_history["source_performance"][source] = []
        
        # Training loop
        for episode in range(num_episodes):
            episode_reward, episode_loss, source_name = self._run_episode()
            
            # Record metrics
            self.training_history["episode_rewards"].append(episode_reward)
            self.training_history["episode_losses"].append(episode_loss)
            self.training_history["epsilon_history"].append(self.agent.epsilon)
            
            # Track source-specific performance
            if source_name in self.training_history["source_performance"]:
                self.training_history["source_performance"][source_name].append(episode_reward)
            
            # Update target network periodically
            if episode % 10 == 0:
                self.agent.update_target_network()
            
            # Progress reporting
            if episode % 50 == 0:
                avg_reward = np.mean(self.training_history["episode_rewards"][-50:])
                avg_loss = np.mean([l for l in self.training_history["episode_losses"][-50:] if l is not None])
                print(f"Episode {episode:4d}: Avg Reward: {avg_reward:6.2f}, "
                      f"Avg Loss: {avg_loss:6.4f}, Epsilon: {self.agent.epsilon:.3f}")
            
            # Save model periodically
            if episode % save_interval == 0 and episode > 0:
                self._save_checkpoint(episode)
        
        # Final save
        self._save_checkpoint(num_episodes, final=True)
        
        # Generate training report
        training_report = self._generate_training_report()
        
        print(f"\n🎉 Training completed!")
        print(f"📈 Final average reward: {np.mean(self.training_history['episode_rewards'][-100:]):.2f}")
        print(f"🎯 Final epsilon: {self.agent.epsilon:.3f}")
        
        return training_report
    
    def _run_episode(self) -> Tuple[float, Optional[float], str]:
        """Run a single training episode"""
        # Reset environment
        state, source_id = self.env.reset()
        episode_reward = 0
        episode_loss = 0
        loss_count = 0
        steps = 0
        
        # Get source name for tracking
        source_names = list(self.env.source_map.keys())
        current_source = (
            source_names[source_id]
            if source_id < len(source_names)
            else "unknown"
        )

        while steps < 100:  # Max steps per episode
            # Select action
            action = self.agent.select_action(state, source_id)

            # Take step
            next_state, reward, done, info = self.env.step(action, source_id)
            next_source_id = source_id  # Source stays same within episode

            # Store experience
            self.agent.store_experience(
                state, action, reward, next_state, done,
                source_id, next_source_id
            )
            
            # Train agent
            loss = self.agent.train()
            if loss is not None:
                episode_loss += loss
                loss_count += 1
            
            # Update state
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        # Calculate average loss for episode
        avg_loss = episode_loss / loss_count if loss_count > 0 else None
        
        return episode_reward, avg_loss, current_source
    
    def _save_checkpoint(self, episode: int, final: bool = False):
        """Save model checkpoint"""
        if final:
            filepath = self.save_dir / "enhanced_dqn_final.pth"
        else:
            filepath = self.save_dir / f"enhanced_dqn_episode_{episode}.pth"
        
        self.agent.save_model(str(filepath))
        
        # Save training history
        history_file = self.save_dir / "training_history.json"
        with open(history_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = {}
            for key, value in self.training_history.items():
                if isinstance(value, dict):
                    serializable_history[key] = {k: [float(x) for x in v] for k, v in value.items()}
                else:
                    serializable_history[key] = [float(x) if x is not None else None for x in value]
            json.dump(serializable_history, f, indent=2)
    
    def _generate_training_report(self) -> Dict:
        """Generate comprehensive training report"""
        episode_rewards = self.training_history["episode_rewards"]
        episode_losses = [l for l in self.training_history["episode_losses"] if l is not None]
        
        report = {
            "training_summary": {
                "total_episodes": len(episode_rewards),
                "final_average_reward": float(np.mean(episode_rewards[-100:])),
                "best_episode_reward": float(np.max(episode_rewards)),
                "final_epsilon": float(self.agent.epsilon),
                "average_loss": float(np.mean(episode_losses)) if episode_losses else 0.0
            },
            "source_performance": {},
            "learning_progress": {
                "reward_improvement": float(np.mean(episode_rewards[-100:]) - np.mean(episode_rewards[:100])),
                "loss_reduction": float(np.mean(episode_losses[:100]) - np.mean(episode_losses[-100:])) if len(episode_losses) > 100 else 0.0
            }
        }
        
        # Source-specific performance
        for source, rewards in self.training_history["source_performance"].items():
            if rewards:
                report["source_performance"][source] = {
                    "average_reward": float(np.mean(rewards)),
                    "episodes_count": len(rewards),
                    "improvement": float(np.mean(rewards[-10:]) - np.mean(rewards[:10])) if len(rewards) > 10 else 0.0
                }
        
        return report
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.training_history["episode_rewards"])
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        
        # Episode losses
        losses = [l for l in self.training_history["episode_losses"] if l is not None]
        axes[0, 1].plot(losses)
        axes[0, 1].set_title("Training Loss")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Loss")
        
        # Epsilon decay
        axes[1, 0].plot(self.training_history["epsilon_history"])
        axes[1, 0].set_title("Epsilon Decay")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Epsilon")
        
        # Source performance
        for source, rewards in self.training_history["source_performance"].items():
            if rewards:
                # Calculate moving average
                window = min(50, len(rewards) // 10)
                if window > 1:
                    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    axes[1, 1].plot(moving_avg, label=source)
        
        axes[1, 1].set_title("Source-Specific Performance")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Avg Reward")
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"📊 Training plots saved to {save_path}")
        else:
            plt.show()
    
    def evaluate_model(self, num_episodes: int = 100) -> Dict:
        """Evaluate trained model performance"""
        print(f"🧪 Evaluating model over {num_episodes} episodes...")
        
        # Set epsilon to 0 for evaluation (no exploration)
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        
        evaluation_rewards = []
        source_performance = {source: [] for source in self.env.source_map.keys()}
        
        for episode in range(num_episodes):
            state, source_id = self.env.reset()
            episode_reward = 0
            steps = 0
            
            source_names = list(self.env.source_map.keys())
            current_source = (
                source_names[source_id]
                if source_id < len(source_names)
                else "unknown"
            )
            
            while steps < 100:
                action = self.agent.select_action(state, source_id)
                state, reward, done, _ = self.env.step(action, source_id)
                episode_reward += reward
                steps += 1
                
                if done:
                    break
            
            evaluation_rewards.append(episode_reward)
            if current_source in source_performance:
                source_performance[current_source].append(episode_reward)
        
        # Restore original epsilon
        self.agent.epsilon = original_epsilon
        
        # Calculate evaluation metrics
        evaluation_report = {
            "average_reward": float(np.mean(evaluation_rewards)),
            "std_reward": float(np.std(evaluation_rewards)),
            "min_reward": float(np.min(evaluation_rewards)),
            "max_reward": float(np.max(evaluation_rewards)),
            "source_performance": {}
        }
        
        for source, rewards in source_performance.items():
            if rewards:
                evaluation_report["source_performance"][source] = {
                    "average_reward": float(np.mean(rewards)),
                    "episodes": len(rewards)
                }
        
        print(f"📊 Evaluation complete!")
        print(f"   Average reward: {evaluation_report['average_reward']:.2f} ± {evaluation_report['std_reward']:.2f}")
        print(f"   Range: [{evaluation_report['min_reward']:.2f}, {evaluation_report['max_reward']:.2f}]")
        
        return evaluation_report


def train_enhanced_dqn(num_episodes: int = 1000, save_interval: int = 200) -> Dict:
    """Convenience function to train enhanced DQN"""
    trainer = TrainingManager()
    return trainer.train(num_episodes, save_interval)


if __name__ == "__main__":
    # Train the model
    training_report = train_enhanced_dqn(num_episodes=1000)
    
    # Print final report
    print("\n📋 FINAL TRAINING REPORT")
    print("=" * 50)
    for key, value in training_report["training_summary"].items():
        print(f"{key}: {value}")
    
    print("\n📊 Source Performance:")
    for source, metrics in training_report["source_performance"].items():
        print(f"  {source}: {metrics['average_reward']:.2f} avg reward ({metrics['episodes_count']} episodes)")
