#!/usr/bin/env python3
"""
Test trained DDPG model for 4-DOF robot drawing
Verify that robot learned to draw square trajectory correctly
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import importlib.util
from agents.ddpg import DDPGAgent

def test_trained_model():
    """Test the trained model with square trajectory"""
    print("🧪 === TESTING TRAINED DDPG MODEL ===")
    print("Checking if robot learned to draw 8-point square\n")
    
    # Load learning environment (same as training)
    spec = importlib.util.spec_from_file_location("learning_env", "robot_4dof_env_learning.py")
    learning_env_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(learning_env_module)
    
    # Create environment with same training config
    env = learning_env_module.Robot4DOFDrawingEnv(
        max_episode_steps=30,
        success_threshold=0.20,  # Same 20cm threshold
        enable_domain_randomization=False
    )
    
    print(f"✅ Environment loaded:")
    print(f"   Success threshold: {env.success_threshold*100:.0f}cm")
    print(f"   Trajectory points: {len(env.trajectory_points)}")
    print(f"   Max steps: {env.max_episode_steps}")
    
    # Load trained DDPG agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DDPGAgent(
        env=env,
        input_dims=env.observation_space['observation'].shape[0],
        alpha=1e-3,
        beta=1e-3,
        tau=0.005,
        gamma=0.99,
        batch_size=128,
        noise_factor=0.0  # No noise during testing
    )
    
    # Initialize networks by running a dummy forward pass
    dummy_obs = env.observation_space['observation'].sample()
    dummy_action = agent.choose_action(dummy_obs)
    print("✅ Networks initialized")
    
    # Load trained model
    try:
        agent.load_models()
        print("✅ Trained model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Trying alternative method to verify training success...")
        
        # Alternative: Test with random actions to see environment behavior
        print("🔄 Testing environment behavior instead...")
        test_environment_behavior(env)
        return
    
    # Test parameters
    num_test_episodes = 10
    test_results = []
    
    print(f"\n🎯 Running {num_test_episodes} test episodes...\n")
    
    for episode in range(num_test_episodes):
        print(f"Episode {episode+1}/{num_test_episodes}:")
        
        # Reset environment
        obs, info = env.reset()
        episode_data = {
            'positions': [],
            'targets': [],
            'rewards': [],
            'successes': [],
            'distances': [],
            'waypoints_reached': []
        }
        
        episode_reward = 0
        episode_successes = 0
        waypoints_reached = 0
        
        # Run episode
        for step in range(env.max_episode_steps):
            # Get observation
            obs_array = obs['observation'] if isinstance(obs, dict) else obs
            
            # Get action from trained agent (no exploration noise)
            action = agent.choose_action(obs_array)
            
            # Execute action
            next_obs, reward, done, truncated, next_info = env.step(action)
            
            # Store data
            episode_data['positions'].append(env.current_ee_pos.copy())
            episode_data['targets'].append(env.target_pos.copy())
            episode_data['rewards'].append(reward)
            episode_data['successes'].append(next_info.get('is_success', False))
            episode_data['distances'].append(next_info.get('distance_to_target', 1.0))
            
            episode_reward += reward
            
            if next_info.get('is_success', False):
                episode_successes += 1
                waypoints_reached = max(waypoints_reached, env.current_trajectory_idx)
                print(f"  Step {step+1:2d}: ✅ Waypoint {env.current_trajectory_idx+1} reached! Distance={next_info['distance_to_target']:.3f}m")
            else:
                print(f"  Step {step+1:2d}: Distance={next_info['distance_to_target']:.3f}m, Progress={next_info['trajectory_progress']:.2f}")
            
            obs = next_obs
            
            if done or truncated:
                break
        
        # Episode summary
        final_progress = next_info.get('trajectory_progress', 0)
        completed_trajectory = final_progress >= 0.99  # 99% completion
        
        episode_result = {
            'episode': episode + 1,
            'total_reward': episode_reward,
            'waypoints_reached': waypoints_reached + 1,  # +1 because index starts at 0
            'final_progress': final_progress,
            'completed_trajectory': completed_trajectory,
            'steps_taken': step + 1,
            'episode_data': episode_data
        }
        test_results.append(episode_result)
        
        print(f"  📊 Episode Summary:")
        print(f"     Total reward: {episode_reward:.1f}")
        print(f"     Waypoints reached: {waypoints_reached + 1}/8")
        print(f"     Trajectory completion: {final_progress*100:.1f}%")
        print(f"     Steps taken: {step + 1}/{env.max_episode_steps}")
        print(f"     Success: {'✅ YES' if completed_trajectory else '❌ NO'}")
        print()
    
    # Overall analysis
    print("=" * 50)
    print("📊 OVERALL TEST RESULTS")
    print("=" * 50)
    
    total_episodes = len(test_results)
    completed_episodes = sum(1 for r in test_results if r['completed_trajectory'])
    avg_waypoints = np.mean([r['waypoints_reached'] for r in test_results])
    avg_progress = np.mean([r['final_progress'] for r in test_results])
    avg_reward = np.mean([r['total_reward'] for r in test_results])
    avg_steps = np.mean([r['steps_taken'] for r in test_results])
    
    success_rate = completed_episodes / total_episodes * 100
    
    print(f"Success Rate: {success_rate:.1f}% ({completed_episodes}/{total_episodes})")
    print(f"Average Waypoints Reached: {avg_waypoints:.1f}/8")
    print(f"Average Trajectory Completion: {avg_progress*100:.1f}%")
    print(f"Average Reward: {avg_reward:.1f}")
    print(f"Average Steps: {avg_steps:.1f}")
    
    # Detailed analysis
    print(f"\n🔍 DETAILED ANALYSIS:")
    
    if success_rate >= 90:
        print("✅ EXCELLENT! Robot consistently draws the square correctly")
        print("   Ready for: harder trajectories, real robot deployment")
    elif success_rate >= 70:
        print("🟡 GOOD! Robot usually draws the square correctly")
        print("   Consider: more training or parameter tuning")
    elif avg_progress >= 0.5:
        print("🟠 MODERATE! Robot partially draws the square")
        print("   Need: more training episodes or easier configuration")
    else:
        print("❌ POOR! Robot failed to learn square drawing")
        print("   Debug: reward function, network, or environment")
    
    if avg_waypoints >= 7:
        print(f"✅ Waypoint Navigation: Excellent ({avg_waypoints:.1f}/8 average)")
    elif avg_waypoints >= 5:
        print(f"🟡 Waypoint Navigation: Good ({avg_waypoints:.1f}/8 average)")
    else:
        print(f"❌ Waypoint Navigation: Poor ({avg_waypoints:.1f}/8 average)")
    
    # Create visualization
    create_test_visualization(test_results, env)
    
    return test_results

def test_environment_behavior(env):
    """Test environment behavior to verify training setup"""
    print("\n🔍 === ANALYZING ENVIRONMENT BEHAVIOR ===")
    
    # Test a few episodes to understand the trajectory
    for episode in range(3):
        print(f"\nEpisode {episode+1}:")
        obs, info = env.reset()
        
        print(f"  Initial target: {env.target_pos}")
        print(f"  Initial distance: {info['distance_to_target']:.3f}m")
        print(f"  Trajectory points: {len(env.trajectory_points)}")
        
        # Try some intelligent actions towards target
        for step in range(15):
            # Simple action: move toward target
            target_direction = env.target_pos - env.current_ee_pos
            action = np.array([target_direction[0], target_direction[2], 0.0]) * 0.5
            action = np.clip(action, -1, 1)
            
            obs, reward, done, truncated, info = env.step(action)
            
            if info.get('is_success', False):
                print(f"    Step {step+1}: ✅ Success! Waypoint {env.current_trajectory_idx+1} reached")
            elif step % 3 == 0:  # Print every 3rd step
                print(f"    Step {step+1}: Distance={info['distance_to_target']:.3f}m, Progress={info['trajectory_progress']:.2f}")
            
            if done or truncated:
                break
        
        final_progress = info.get('trajectory_progress', 0)
        waypoints_reached = env.current_trajectory_idx + 1 if hasattr(env, 'current_trajectory_idx') else 0
        
        print(f"  Final result: {waypoints_reached}/8 waypoints, {final_progress*100:.1f}% complete")

def create_test_visualization(test_results, env):
    """Create visualization of test results"""
    print(f"\n📈 Creating test visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Trajectory visualization (best episode)
    best_episode = max(test_results, key=lambda x: x['final_progress'])
    positions = np.array(best_episode['episode_data']['positions'])
    targets = np.array(best_episode['episode_data']['targets'])
    
    axes[0,0].plot(positions[:, 0], positions[:, 2], 'b-', linewidth=3, label='Robot Path', marker='o')
    axes[0,0].plot(targets[:, 0], targets[:, 2], 'r--', linewidth=2, label='Target Square', marker='s', markersize=8)
    
    # Add waypoint numbers
    for i, target in enumerate(targets[::max(1, len(targets)//8)]):  # Show max 8 points
        axes[0,0].annotate(f'{i+1}', (target[0], target[2]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    axes[0,0].set_xlabel('X Position (m)')
    axes[0,0].set_ylabel('Z Position (m)')
    axes[0,0].set_title(f'Best Square Drawing (Episode {best_episode["episode"]})')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axis('equal')
    
    # Add success threshold circle at each target
    for target in targets[::max(1, len(targets)//8)]:
        circle = Circle((target[0], target[2]), env.success_threshold, 
                       fill=False, color='red', alpha=0.3, linestyle='--')
        axes[0,0].add_patch(circle)
    
    # Plot 2: Success rate per episode
    episodes = [r['episode'] for r in test_results]
    completions = [r['completed_trajectory'] for r in test_results]
    
    axes[0,1].bar(episodes, completions, color=['green' if c else 'red' for c in completions])
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Trajectory Completed')
    axes[0,1].set_title('Success per Episode')
    axes[0,1].set_ylim(0, 1.2)
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Waypoints reached
    waypoints_reached = [r['waypoints_reached'] for r in test_results]
    axes[1,0].bar(episodes, waypoints_reached, color='blue', alpha=0.7)
    axes[1,0].axhline(y=8, color='red', linestyle='--', label='Target (8 waypoints)')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Waypoints Reached')
    axes[1,0].set_title('Waypoints Reached per Episode')
    axes[1,0].set_ylim(0, 9)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Performance metrics
    progress_percentages = [r['final_progress'] * 100 for r in test_results]
    rewards = [r['total_reward'] for r in test_results]
    
    ax4_twin = axes[1,1].twinx()
    
    bars1 = axes[1,1].bar([e - 0.2 for e in episodes], progress_percentages, 
                         width=0.4, label='Progress %', color='green', alpha=0.7)
    bars2 = ax4_twin.bar([e + 0.2 for e in episodes], rewards, 
                        width=0.4, label='Reward', color='orange', alpha=0.7)
    
    axes[1,1].set_xlabel('Episode')
    axes[1,1].set_ylabel('Progress (%)', color='green')
    ax4_twin.set_ylabel('Reward', color='orange')
    axes[1,1].set_title('Progress and Reward per Episode')
    axes[1,1].set_ylim(0, 110)
    axes[1,1].grid(True, alpha=0.3)
    
    # Add legends
    lines1, labels1 = axes[1,1].get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    axes[1,1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.suptitle('DDPG Robot Square Drawing Test Results', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'test_results_{len(test_results)}_episodes.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Test visualization saved to: {plot_filename}")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    results = test_trained_model()
