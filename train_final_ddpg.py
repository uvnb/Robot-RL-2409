#!/usr/bin/env python3
"""
Final optimized DDPG training for 4-DOF robot drawing
Based on successful learning test results
"""

import numpy as np
import torch
import importlib.util
from agents.ddpg import DDPGAgent
from replay_memory.ReplayBuffer import ReplayBuffer
import matplotlib.pyplot as plt
import os
from datetime import datetime

def train_optimized_ddpg():
    """Run optimized DDPG training with proven learnable configuration"""
    print("🚀 OPTIMIZED DDPG TRAINING - Robot Drawing System")
    print("Configuration: 20cm threshold, 8-point square, 30 steps\n")
    
    # Load proven learnable environment
    spec = importlib.util.spec_from_file_location("learning_env", "robot_4dof_env_learning.py")
    learning_env_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(learning_env_module)
    
    # Create environment with same successful configuration
    env = learning_env_module.Robot4DOFDrawingEnv(
        max_episode_steps=30,          # Same as test
        success_threshold=0.20,        # Same 20cm threshold  
        enable_domain_randomization=False  # No noise initially
    )
    
    print(f"Environment: {env.success_threshold*100:.0f}cm threshold, {len(env.trajectory_points)} waypoints")
    
    # Training hyperparameters
    max_episodes = 100     # Should learn quickly given test results
    buffer_size = 50000
    batch_size = 128
    warmup_episodes = 10   # Collect some experience first
    
    # DDPG Agent setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    agent = DDPGAgent(
        env=env,
        input_dims=env.observation_space['observation'].shape[0],
        alpha=1e-3,        # Actor learning rate  
        beta=1e-3,         # Critic learning rate
        tau=0.005,         # Soft update
        gamma=0.99,        # Future reward discount
        batch_size=batch_size,
        noise_factor=0.2   # Exploration noise
    )
    
    # Initialize replay buffer (DDPG agent has its own memory)
    # replay_buffer = ReplayBuffer(buffer_size, device)
    
    # Training metrics
    episode_rewards = []
    success_rates = []
    progress_rates = []
    completion_times = []
    
    print(f"🎯 Starting DDPG training for {max_episodes} episodes...\n")
    
    best_success_rate = 0
    best_model_saved = False
    
    for episode in range(max_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_success = False
        max_progress = 0
        steps_to_complete = 0
        
        for step in range(env.max_episode_steps):
            # Extract observation array from dict
            obs_array = obs['observation'] if isinstance(obs, dict) else obs
            
            # Get action from agent
            if episode < warmup_episodes:
                # Random exploration during warmup
                action = env.action_space.sample()
            else:
                # Policy action with exploration noise
                action = agent.choose_action(obs_array)
            
            # Execute action
            next_obs, reward, done, truncated, next_info = env.step(action)
            next_obs_array = next_obs['observation'] if isinstance(next_obs, dict) else next_obs
            
            # Store transition in agent's memory
            agent.remember(obs_array, action, reward, next_obs_array, done or truncated)
            
            episode_reward += reward
            max_progress = max(max_progress, next_info['trajectory_progress'])
            
            if next_info['is_success']:
                episode_success = True
                
            # Train agent (after warmup)
            if episode >= warmup_episodes and agent.memory.counter > batch_size:
                agent.learn()
            
            obs = next_obs
            info = next_info
            steps_to_complete = step + 1
            
            if done or truncated:
                break
        
        # Track metrics
        episode_rewards.append(episode_reward)
        success_rates.append(1.0 if episode_success else 0.0)
        progress_rates.append(max_progress)
        completion_times.append(steps_to_complete)
        
        # Calculate rolling averages
        window = min(10, episode + 1)
        recent_success = np.mean(success_rates[-window:]) * 100
        recent_progress = np.mean(progress_rates[-window:]) * 100
        recent_reward = np.mean(episode_rewards[-window:])
        recent_steps = np.mean(completion_times[-window:])
        
        # Print progress
        if episode % 5 == 0 or episode < 10:
            print(f"Episode {episode+1:3d}: Reward={recent_reward:6.1f}, "
                  f"Success={recent_success:5.1f}%, Progress={recent_progress:5.1f}%, "
                  f"Steps={recent_steps:4.1f}")
        
        # Save best model
        if recent_success > best_success_rate and episode > warmup_episodes:
            best_success_rate = recent_success
            if not os.path.exists('ckp'):
                os.makedirs('ckp')
            agent.save_models()
            print(f"    💾 New best model saved! Success rate: {recent_success:.1f}%")
        
        # Early success check
        if episode > 20 and recent_success >= 95.0:
            print(f"\n🎉 TRAINING SUCCESS! Achieved {recent_success:.1f}% success rate")
            break
    
    # Final results
    final_episodes = min(10, len(success_rates))
    final_success_rate = np.mean(success_rates[-final_episodes:]) * 100
    final_progress_rate = np.mean(progress_rates[-final_episodes:]) * 100
    final_reward = np.mean(episode_rewards[-final_episodes:])
    final_steps = np.mean(completion_times[-final_episodes:])
    
    print(f"\n📊 FINAL RESULTS (last {final_episodes} episodes):")
    print(f"   Success Rate: {final_success_rate:.1f}%")
    print(f"   Progress Rate: {final_progress_rate:.1f}%")
    print(f"   Average Reward: {final_reward:.1f}")
    print(f"   Average Steps: {final_steps:.1f}")
    
    # Analysis and recommendations
    if final_success_rate >= 90:
        print("✅ EXCELLENT TRAINING! Robot learned to draw successfully")
        print("➡️  Ready for: harder trajectories, domain randomization, real robot")
    elif final_success_rate >= 70:
        print("🟡 GOOD TRAINING! Robot mostly successful")  
        print("➡️  Consider: more episodes, hyperparameter tuning")
    elif final_progress_rate >= 50:
        print("🟠 MODERATE TRAINING! Some learning occurred")
        print("➡️  Need: reward tuning, network architecture changes")
    else:
        print("❌ POOR TRAINING! Little learning occurred") 
        print("➡️  Debug: reward function, network, environment")
    
    # Save final model
    if not best_model_saved:
        if not os.path.exists('ckp'):
            os.makedirs('ckp')
        agent.save_models()
        print(f"\n💾 Final model saved")
    
    # Plot results
    if len(episode_rewards) >= 10:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        ax1.plot(episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Success rate (rolling average)
        rolling_success = [np.mean(success_rates[max(0, i-9):i+1]) for i in range(len(success_rates))]
        ax2.plot(rolling_success)
        ax2.set_title('Success Rate (10-episode rolling avg)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate')
        ax2.set_ylim(0, 1)
        ax2.grid(True)
        
        # Progress rate  
        rolling_progress = [np.mean(progress_rates[max(0, i-9):i+1]) for i in range(len(progress_rates))]
        ax3.plot(rolling_progress)
        ax3.set_title('Trajectory Progress (10-episode rolling avg)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Progress')
        ax3.set_ylim(0, 1)
        ax3.grid(True)
        
        # Completion steps
        rolling_steps = [np.mean(completion_times[max(0, i-9):i+1]) for i in range(len(completion_times))]
        ax4.plot(rolling_steps)
        ax4.set_title('Steps to Complete (10-episode rolling avg)')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Steps')
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f'ddpg_training_results_{timestamp}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"📈 Training plots saved to: {plot_filename}")
        
        plt.show()
    
    return agent, {
        'success_rates': success_rates,
        'progress_rates': progress_rates, 
        'episode_rewards': episode_rewards,
        'completion_times': completion_times
    }

if __name__ == "__main__":
    agent, results = train_optimized_ddpg()
