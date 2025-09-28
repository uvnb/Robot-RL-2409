#!/usr/bin/env python3
"""
🎨 DDPG+HER Training for 4-DOF Robot Drawing Task
Optimized for 2D trajectory following with enhanced rewards and stability
"""

import numpy as np
import gymnasium as gym
import sys
import os
import time
import random
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_4dof_env import Robot4DOFDrawingEnv
from agents.ddpg import DDPGAgent
from utils.HER import her_augmentation
import matplotlib.pyplot as plt

def main():
    print("🎨 === 4-DOF ROBOT DRAWING TRAINING WITH DDPG + HER ===")
    
    # Enhanced Training parameters for FASTER learning
    MAX_EPISODES = 200  # Reduced from 300
    EVALUATION_EPISODES = 20
    OPT_STEPS_PER_EPISODE = 40  # Reduced from 80 for speed
    HER_REPLAY_RATIO = 2  # Reduced from 4 for speed
    
    # Exploration and stability parameters
    NOISE_DECAY_START = 50
    NOISE_DECAY_RATE = 0.998
    MIN_NOISE = 0.02
    
    # Early stopping and performance tracking
    EARLY_STOP_THRESHOLD = 0.6  # 60% success rate
    EARLY_STOP_WINDOW = 30
    PERFORMANCE_EVAL_FREQ = 25
    
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    print(f"🔧 Training Configuration:")
    print(f"   Max episodes: {MAX_EPISODES}")
    print(f"   Optimization steps: {OPT_STEPS_PER_EPISODE}")
    print(f"   HER replay ratio: {HER_REPLAY_RATIO}")
    print(f"   Early stop threshold: {EARLY_STOP_THRESHOLD}")
    
    # Create drawing environment with EASIER settings
    env = Robot4DOFDrawingEnv(
        max_episode_steps=50,  # REDUCED from 100 to 50
        success_threshold=0.08,  # INCREASED from 0.02 to 8cm - much easier
        enable_domain_randomization=False  # DISABLE for initial learning
    )
    
    # Start with SMALL, SIMPLE circle trajectory
    env.set_trajectory("circle", radius=0.08, num_points=8)  # Much smaller and fewer points
    
    print(f"🤖 Environment Configuration:")
    print(f"   Action space: {env.action_space.shape}")
    print(f"   Observation space: {env.observation_space['observation'].shape}")
    print(f"   Success threshold: {env.success_threshold}")
    print(f"   Initial trajectory: Circle (25 points)")
    
    # Initialize DDPG agent with optimized parameters
    state_dim = 12 + 3 + 3  # observation + achieved_goal + desired_goal = 18
    agent = DDPGAgent(
        env=env, 
        input_dims=state_dim,
        alpha=0.001,  # Reduced learning rate for stability
        beta=0.002,
        gamma=0.98,   # Slightly lower discount for drawing tasks
        tau=0.005,    # Soft update rate
        batch_size=128,
        noise_factor=0.2
    )
    
    print(f"🧠 Agent Configuration:")
    print(f"   State dimension: {state_dim}")
    print(f"   Learning rates: α={agent.alpha}, β={agent.beta}")
    print(f"   Batch size: {agent.batch_size}")
    print(f"   Initial noise: {agent.noise_factor}")
    
    # Tracking variables
    episode_scores = []
    success_history = []
    distance_history = []
    trajectory_progress_history = []
    actor_losses = []
    critic_losses = []
    avg_scores = []
    
    best_performance = {"score": -float('inf'), "success_rate": 0}
    total_successes = 0
    
    # Training loop
    print(f"\n🚀 Starting training...")
    start_time = time.time()
    
    for episode in range(MAX_EPISODES):
        # Episode setup
        obs_array = []
        actions_array = []
        new_obs_array = []
        episode_score = 0
        steps = 0
        
        # Reset environment
        observation, info = env.reset()
        
        # Episode loop
        while True:
            # Prepare state
            state = np.concatenate([
                observation['observation'],
                observation['achieved_goal'],
                observation['desired_goal']
            ])
            
            # Choose action with exploration
            action = agent.choose_action(state, add_noise=True)
            
            # Execute action
            new_observation, reward, terminated, truncated, step_info = env.step(action)
            
            # Prepare next state
            next_state = np.concatenate([
                new_observation['observation'],
                new_observation['achieved_goal'], 
                new_observation['desired_goal']
            ])
            
            # Store experience
            agent.remember(state, action, reward, next_state, terminated)
            
            # Store for HER
            obs_array.append(observation)
            actions_array.append(action)
            new_obs_array.append(new_observation)
            
            # Update
            observation = new_observation
            episode_score += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        # Episode statistics
        final_distance = step_info.get('distance_to_target', 1.0)
        is_success = step_info.get('is_success', False)
        trajectory_progress = step_info.get('trajectory_progress', 0.0)
        trajectory_complete = step_info.get('trajectory_complete', False)
        
        if is_success:
            total_successes += 1
        
        # Store episode data
        episode_scores.append(episode_score)
        success_history.append(int(is_success))
        distance_history.append(final_distance)
        trajectory_progress_history.append(trajectory_progress)
        
        # HER augmentation with increased replay
        for _ in range(HER_REPLAY_RATIO):
            her_augmentation(agent, obs_array, actions_array, new_obs_array)
        
        # Training with multiple optimization steps
        episode_actor_loss = None
        episode_critic_loss = None
        
        for _ in range(OPT_STEPS_PER_EPISODE):
            if agent.memory.counter > agent.batch_size:
                actor_loss, critic_loss = agent.learn()
                if actor_loss is not None:
                    episode_actor_loss = actor_loss
                if critic_loss is not None:
                    episode_critic_loss = critic_loss
        
        # Store losses
        actor_losses.append(episode_actor_loss)
        critic_losses.append(episode_critic_loss)
        
        # Noise decay
        if episode >= NOISE_DECAY_START:
            agent.noise_factor = max(agent.noise_factor * NOISE_DECAY_RATE, MIN_NOISE)
        
        # Calculate moving averages
        window = min(50, episode + 1)
        avg_score = np.mean(episode_scores[-window:])
        avg_scores.append(avg_score)
        recent_success_rate = np.mean(success_history[-min(20, len(success_history)):])
        
        # Progress logging - PRINT EVERY EPISODE
        status_icon = "✅" if is_success else "❌"
        complete_icon = "🎯" if trajectory_complete else ""
        
        print(f"Episode {episode:3d}: {status_icon}{complete_icon} "
              f"Score={episode_score:6.1f}, Steps={steps:2d}, "
              f"Distance={final_distance:.3f}, Progress={trajectory_progress:.2f}, "
              f"Success Rate={recent_success_rate:.2f}")
        
        # Performance evaluation
        if (episode + 1) % PERFORMANCE_EVAL_FREQ == 0:
            current_success_rate = total_successes / (episode + 1)
            
            print(f"\n📊 === PERFORMANCE UPDATE (Episode {episode + 1}) ===")
            print(f"   Overall Success Rate: {current_success_rate:.2f} ({total_successes}/{episode + 1})")
            print(f"   Recent Success Rate: {recent_success_rate:.2f}")
            print(f"   Average Score: {avg_score:.2f}")
            print(f"   Current Noise Factor: {agent.noise_factor:.4f}")
            print(f"   Average Distance: {np.mean(distance_history[-PERFORMANCE_EVAL_FREQ:]):.4f}")
            print(f"   Average Progress: {np.mean(trajectory_progress_history[-PERFORMANCE_EVAL_FREQ:]):.2f}")
            
            # Update best performance
            if recent_success_rate > best_performance["success_rate"]:
                best_performance = {
                    "score": avg_score,
                    "success_rate": recent_success_rate,
                    "episode": episode + 1
                }
                # Save best model
                agent.save_models()
                print(f"🏆 New best performance! Model saved.")
            
            print("=" * 60)
        
        # Early stopping check
        if episode >= EARLY_STOP_WINDOW:
            recent_success_rate_window = np.mean(success_history[-EARLY_STOP_WINDOW:])
            if recent_success_rate_window >= EARLY_STOP_THRESHOLD:
                print(f"\n🎯 EARLY STOPPING TRIGGERED!")
                print(f"Success rate {recent_success_rate_window:.2f} >= {EARLY_STOP_THRESHOLD}")
                print(f"over last {EARLY_STOP_WINDOW} episodes.")
                break
        
        # Progressive curriculum (optional)
        if episode == 150 and recent_success_rate > 0.4:
            print(f"\n🔄 Switching to square trajectory...")
            env.set_trajectory("square", size=0.18, num_points_per_side=12)
        
        if episode == 250 and recent_success_rate > 0.5:
            print(f"\n🔥 Switching to complex circle trajectory...")
            env.set_trajectory("circle", radius=0.15, num_points=40)
    
    # Training completed
    total_episodes = episode + 1
    elapsed_time = time.time() - start_time
    final_success_rate = total_successes / total_episodes
    
    print(f"\n🎉 === TRAINING COMPLETED ===")
    print(f"Total Episodes: {total_episodes}")
    print(f"Training Time: {elapsed_time/60:.1f} minutes")
    print(f"Final Success Rate: {final_success_rate:.2f} ({total_successes}/{total_episodes})")
    print(f"Best Performance: {best_performance['success_rate']:.2f} at episode {best_performance['episode']}")
    print(f"Average Final Score: {np.mean(episode_scores[-20:]):.2f}")
    
    # Save results
    results = {
        'episode_scores': episode_scores,
        'success_history': success_history,
        'distance_history': distance_history,
        'trajectory_progress_history': trajectory_progress_history,
        'actor_losses': [x for x in actor_losses if x is not None],
        'critic_losses': [x for x in critic_losses if x is not None],
        'avg_scores': avg_scores,
        'final_success_rate': final_success_rate,
        'total_episodes': total_episodes,
        'training_time': elapsed_time
    }
    
    np.savez('results_4dof_drawing.npz', **results)
    print(f"📁 Results saved to: results_4dof_drawing.npz")
    
    # Create visualization
    create_training_plots(results, total_episodes)
    
    print(f"\n✅ Training complete! Check results_4dof_drawing.npz and training plots.")

def create_training_plots(results, total_episodes):
    """Create comprehensive training visualization"""
    print(f"📈 Creating training plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Scores
    axes[0,0].plot(results['episode_scores'], alpha=0.6, label='Episode Score')
    axes[0,0].plot(results['avg_scores'], linewidth=2, label='Moving Average')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Score')
    axes[0,0].set_title('Training Scores')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Success Rate
    window = 20
    if len(results['success_history']) >= window:
        success_rate = [np.mean(results['success_history'][max(0, i-window):i+1]) 
                       for i in range(len(results['success_history']))]
        axes[0,1].plot(success_rate, linewidth=2, color='green')
    axes[0,1].scatter(range(len(results['success_history'])), results['success_history'], 
                     alpha=0.3, s=10)
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Success Rate')
    axes[0,1].set_title('Success Rate (Moving Average)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Distance to Target
    axes[0,2].plot(results['distance_history'])
    axes[0,2].axhline(y=0.02, color='r', linestyle='--', label='Success Threshold')
    axes[0,2].set_xlabel('Episode')
    axes[0,2].set_ylabel('Distance (m)')
    axes[0,2].set_title('Distance to Target')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Trajectory Progress
    axes[1,0].plot(results['trajectory_progress_history'])
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Progress Ratio')
    axes[1,0].set_title('Trajectory Completion Progress')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: Training Losses
    if results['actor_losses'] and results['critic_losses']:
        axes[1,1].plot(results['actor_losses'], label='Actor Loss', alpha=0.7)
        axes[1,1].plot(results['critic_losses'], label='Critic Loss', alpha=0.7)
        axes[1,1].set_xlabel('Episode')
        axes[1,1].set_ylabel('Loss')
        axes[1,1].set_title('Training Losses')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Performance Summary
    final_window = min(50, len(results['success_history']))
    final_success = np.mean(results['success_history'][-final_window:])
    final_distance = np.mean(results['distance_history'][-final_window:])
    final_progress = np.mean(results['trajectory_progress_history'][-final_window:])
    
    metrics = ['Success Rate', 'Avg Distance', 'Avg Progress']
    values = [final_success, final_distance, final_progress]
    colors = ['green', 'orange', 'blue']
    
    bars = axes[1,2].bar(metrics, values, color=colors, alpha=0.7)
    axes[1,2].set_title(f'Final Performance (Last {final_window} episodes)')
    axes[1,2].set_ylabel('Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{value:.3f}', ha='center', va='bottom')
    
    axes[1,2].grid(True, alpha=0.3)
    
    plt.suptitle(f'4-DOF Robot Drawing Training Results ({total_episodes} episodes)', fontsize=16)
    plt.tight_layout()
    plt.savefig('training_results_4dof_drawing.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training plots saved: training_results_4dof_drawing.png")

if __name__ == "__main__":
    main()
