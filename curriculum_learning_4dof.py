#!/usr/bin/env python3
"""
Curriculum Learning for 4-DOF Robot Drawing
Progressive difficulty from easy to hard to prevent overfitting
"""

import numpy as np
import torch
import importlib.util
from agents.ddpg import DDPGAgent
import matplotlib.pyplot as plt
import os
from datetime import datetime

def create_curriculum_stages():
    """Define curriculum learning stages"""
    return [
        # Stage 1: Super Easy
        {
            'name': 'SUPER_EASY',
            'success_threshold': 0.25,  # 25cm
            'max_episode_steps': 20,
            'episodes': 20,
            'trajectory_points': 4,  # Very simple
            'noise_factor': 0.1
        },
        # Stage 2: Easy
        {
            'name': 'EASY', 
            'success_threshold': 0.20,  # 20cm
            'max_episode_steps': 25,
            'episodes': 30,
            'trajectory_points': 8,
            'noise_factor': 0.15
        },
        # Stage 3: Medium
        {
            'name': 'MEDIUM',
            'success_threshold': 0.15,  # 15cm
            'max_episode_steps': 30,
            'episodes': 40,
            'trajectory_points': 12,
            'noise_factor': 0.2
        },
        # Stage 4: Hard
        {
            'name': 'HARD',
            'success_threshold': 0.10,  # 10cm
            'max_episode_steps': 40,
            'episodes': 50,
            'trajectory_points': 16,
            'noise_factor': 0.25
        },
        # Stage 5: Expert
        {
            'name': 'EXPERT',
            'success_threshold': 0.05,  # 5cm
            'max_episode_steps': 50,
            'episodes': 60,
            'trajectory_points': 20,
            'noise_factor': 0.3
        }
    ]

def train_curriculum_stage(agent, env, stage_config, stage_idx):
    """Train agent on one curriculum stage"""
    print(f"\n🎯 STAGE {stage_idx+1}: {stage_config['name']}")
    print(f"   Threshold: {stage_config['success_threshold']*100:.0f}cm")
    print(f"   Episodes: {stage_config['episodes']}")
    print(f"   Trajectory points: {stage_config['trajectory_points']}")
    
    # Update environment parameters
    env.success_threshold = stage_config['success_threshold']
    env.max_episode_steps = stage_config['max_episode_steps']
    agent.noise_factor = stage_config['noise_factor']
    
    # Generate new trajectory for this stage
    if stage_config['trajectory_points'] == 4:
        # Very simple 4-point trajectory
        center_x, center_z = 0.3, 0.325
        size = 0.1  # Smaller size for easier learning
        points = [
            np.array([center_x - size/2, env.drawing_plane_y, center_z - size/2]),
            np.array([center_x + size/2, env.drawing_plane_y, center_z - size/2]),
            np.array([center_x + size/2, env.drawing_plane_y, center_z + size/2]),
            np.array([center_x - size/2, env.drawing_plane_y, center_z + size/2])
        ]
        env.trajectory_points = points
        print(f"   📍 Generated 4-point simple square trajectory")
    elif stage_config['trajectory_points'] <= 8:
        env._generate_square_trajectory()
    else:
        env._generate_circle_trajectory(n_points=stage_config['trajectory_points'])
    
    stage_rewards = []
    stage_success_rates = []
    stage_progress_rates = []
    
    success_threshold_for_advancement = 0.8  # 80% success needed
    consecutive_success_count = 0
    min_episodes_before_check = max(10, stage_config['episodes'] // 4)
    
    for episode in range(stage_config['episodes']):
        obs, info = env.reset()
        episode_reward = 0
        episode_success = False
        max_progress = 0
        
        for step in range(env.max_episode_steps):
            obs_array = obs['observation'] if isinstance(obs, dict) else obs
            action = agent.choose_action(obs_array)
            
            next_obs, reward, done, truncated, next_info = env.step(action)
            next_obs_array = next_obs['observation'] if isinstance(next_obs, dict) else next_obs
            
            agent.remember(obs_array, action, reward, next_obs_array, done or truncated)
            
            episode_reward += reward
            max_progress = max(max_progress, next_info['trajectory_progress'])
            
            if next_info['is_success']:
                episode_success = True
            
            # Train agent
            if agent.memory.counter > 128:
                agent.learn()
            
            obs = next_obs
            info = next_info
            
            if done or truncated:
                break
        
        stage_rewards.append(episode_reward)
        stage_success_rates.append(1.0 if episode_success else 0.0)
        stage_progress_rates.append(max_progress)
        
        # Check for early advancement
        if episode >= min_episodes_before_check:
            recent_success = np.mean(stage_success_rates[-10:])
            if recent_success >= success_threshold_for_advancement:
                consecutive_success_count += 1
                if consecutive_success_count >= 5:  # 5 consecutive good episodes
                    print(f"   ✅ Stage mastered early! ({recent_success*100:.1f}% success)")
                    break
            else:
                consecutive_success_count = 0
        
        # Progress reporting
        if (episode + 1) % 10 == 0 or episode < 5:
            recent_success = np.mean(stage_success_rates[-min(10, episode+1):]) * 100
            recent_progress = np.mean(stage_progress_rates[-min(10, episode+1):]) * 100
            recent_reward = np.mean(stage_rewards[-min(10, episode+1):])
            print(f"   Episode {episode+1:3d}: Success={recent_success:5.1f}%, "
                  f"Progress={recent_progress:5.1f}%, Reward={recent_reward:6.1f}")
    
    # Stage completion metrics
    final_success = np.mean(stage_success_rates[-10:]) * 100
    final_progress = np.mean(stage_progress_rates[-10:]) * 100
    final_reward = np.mean(stage_rewards[-10:])
    
    print(f"   📊 Stage Result: Success={final_success:.1f}%, "
          f"Progress={final_progress:.1f}%, Reward={final_reward:.1f}")
    
    return {
        'rewards': stage_rewards,
        'success_rates': stage_success_rates,
        'progress_rates': stage_progress_rates,
        'final_success': final_success
    }

def curriculum_learning_training():
    """Main curriculum learning training loop"""
    print("🚀 CURRICULUM LEARNING - 4-DOF Robot Drawing")
    print("Training from easy to hard to prevent overfitting\n")
    
    # Load environment
    spec = importlib.util.spec_from_file_location("learning_env", "robot_4dof_env_learning.py")
    learning_env_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(learning_env_module)
    
    env = learning_env_module.Robot4DOFDrawingEnv(
        enable_domain_randomization=False  # Start without noise
    )
    
    # Initialize agent
    agent = DDPGAgent(
        env=env,
        input_dims=env.observation_space['observation'].shape[0],
        alpha=1e-3,
        beta=1e-3,
        tau=0.005,
        gamma=0.99,
        batch_size=128,
        noise_factor=0.1
    )
    
    # Get curriculum stages
    stages = create_curriculum_stages()
    all_results = []
    
    # Train through curriculum
    for stage_idx, stage_config in enumerate(stages):
        stage_results = train_curriculum_stage(agent, env, stage_config, stage_idx)
        all_results.append({
            'stage': stage_config['name'],
            'config': stage_config,
            'results': stage_results
        })
        
        # Check if stage was mastered
        if stage_results['final_success'] < 60.0:  # Less than 60% success
            print(f"   ❌ Stage {stage_config['name']} not mastered!")
            print(f"   🔄 Extending training...")
            
            # Extend training for difficult stage
            extended_results = train_curriculum_stage(agent, env, {
                **stage_config,
                'episodes': stage_config['episodes'] // 2,  # Extra episodes
                'name': f"{stage_config['name']}_EXTENDED"
            }, stage_idx)
            
            if extended_results['final_success'] < 50.0:
                print(f"   ⚠️  Still struggling with {stage_config['name']}")
                print("   💡 Consider adjusting curriculum or hyperparameters")
        
        # Save checkpoint after each stage
        if not os.path.exists('ckp'):
            os.makedirs('ckp')
        if not os.path.exists('ckp/ddpg'):
            os.makedirs('ckp/ddpg')
        agent.save_models()
        print(f"   💾 Checkpoint saved after {stage_config['name']}")
    
    # Final evaluation on all difficulty levels
    print(f"\n🏆 CURRICULUM COMPLETED! Evaluating on all levels...")
    
    final_evaluation = {}
    for stage_config in stages:
        env.success_threshold = stage_config['success_threshold']
        env.max_episode_steps = stage_config['max_episode_steps']
        
        if stage_config['trajectory_points'] == 4:
            # Very simple 4-point trajectory
            center_x, center_z = 0.3, 0.325
            size = 0.1
            points = [
                np.array([center_x - size/2, env.drawing_plane_y, center_z - size/2]),
                np.array([center_x + size/2, env.drawing_plane_y, center_z - size/2]),
                np.array([center_x + size/2, env.drawing_plane_y, center_z + size/2]),
                np.array([center_x - size/2, env.drawing_plane_y, center_z + size/2])
            ]
            env.trajectory_points = points
        elif stage_config['trajectory_points'] <= 8:
            env._generate_square_trajectory()
        else:
            env._generate_circle_trajectory(n_points=stage_config['trajectory_points'])
        
        # Test 10 episodes
        test_success = []
        test_progress = []
        test_rewards = []
        
        for test_ep in range(10):
            obs, _ = env.reset()
            episode_reward = 0
            max_progress = 0
            episode_success = False
            
            for step in range(env.max_episode_steps):
                obs_array = obs['observation'] if isinstance(obs, dict) else obs
                action = agent.choose_action(obs_array)  # No exploration noise
                
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                max_progress = max(max_progress, info['trajectory_progress'])
                
                if info['is_success']:
                    episode_success = True
                
                if done or truncated:
                    break
            
            test_success.append(episode_success)
            test_progress.append(max_progress)
            test_rewards.append(episode_reward)
        
        final_evaluation[stage_config['name']] = {
            'success_rate': np.mean(test_success) * 100,
            'progress_rate': np.mean(test_progress) * 100,
            'avg_reward': np.mean(test_rewards)
        }
    
    # Print final results
    print(f"\n📊 FINAL EVALUATION RESULTS:")
    for stage_name, results in final_evaluation.items():
        print(f"   {stage_name:12s}: Success={results['success_rate']:5.1f}%, "
              f"Progress={results['progress_rate']:5.1f}%, Reward={results['avg_reward']:6.1f}")
    
    # Plot results
    plot_curriculum_results(all_results, final_evaluation)
    
    return agent, all_results, final_evaluation

def plot_curriculum_results(all_results, final_evaluation):
    """Plot curriculum learning results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Success rates by stage during training
    for i, result in enumerate(all_results):
        stage_name = result['stage']
        success_rates = result['results']['success_rates']
        # Rolling average
        rolling_success = [np.mean(success_rates[max(0, j-4):j+1]) for j in range(len(success_rates))]
        ax1.plot(rolling_success, label=stage_name, linewidth=2)
    
    ax1.set_title('Success Rate During Training (5-episode rolling avg)')
    ax1.set_xlabel('Episode within Stage')
    ax1.set_ylabel('Success Rate')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Final evaluation comparison
    stages = list(final_evaluation.keys())
    success_rates = [final_evaluation[stage]['success_rate'] for stage in stages]
    
    bars = ax2.bar(range(len(stages)), success_rates, 
                   color=['green' if sr > 80 else 'orange' if sr > 60 else 'red' for sr in success_rates])
    ax2.set_title('Final Evaluation: Success Rate by Difficulty')
    ax2.set_xlabel('Difficulty Level')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_xticks(range(len(stages)))
    ax2.set_xticklabels(stages, rotation=45)
    ax2.set_ylim(0, 100)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, success_rates)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom')
    
    # Plot 3: Progress rates
    progress_rates = [final_evaluation[stage]['progress_rate'] for stage in stages]
    ax3.bar(range(len(stages)), progress_rates, color='skyblue')
    ax3.set_title('Final Evaluation: Progress Rate by Difficulty')
    ax3.set_xlabel('Difficulty Level')
    ax3.set_ylabel('Progress Rate (%)')
    ax3.set_xticks(range(len(stages)))
    ax3.set_xticklabels(stages, rotation=45)
    ax3.set_ylim(0, 100)
    
    # Plot 4: Rewards
    avg_rewards = [final_evaluation[stage]['avg_reward'] for stage in stages]
    ax4.bar(range(len(stages)), avg_rewards, color='lightcoral')
    ax4.set_title('Final Evaluation: Average Reward by Difficulty')
    ax4.set_xlabel('Difficulty Level')
    ax4.set_ylabel('Average Reward')
    ax4.set_xticks(range(len(stages)))
    ax4.set_xticklabels(stages, rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'curriculum_learning_results_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n📈 Results plotted and saved to: {filename}")
    
    plt.show()

if __name__ == "__main__":
    agent, results, evaluation = curriculum_learning_training()
