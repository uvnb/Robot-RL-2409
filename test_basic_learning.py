#!/usr/bin/env python3
"""
Simple test to see if robot can learn basic movements
No complex training, just check if environment works
"""

import numpy as np
import importlib.util

def test_basic_learning():
    """Test if robot can learn basic trajectory following"""
    print("🧪 Basic learning test - Simple trajectory following")
    
    # Load learning environment
    spec = importlib.util.spec_from_file_location("learning_env", "robot_4dof_env_learning.py")
    learning_env_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(learning_env_module)
    
    # Create environment
    env = learning_env_module.Robot4DOFDrawingEnv(
        max_episode_steps=20,          # Very short
        success_threshold=0.20,        # Very easy (20cm)
        enable_domain_randomization=False
    )
    
    print(f"Environment: {env.success_threshold*100:.0f}cm threshold, {len(env.trajectory_points)} points")
    
    # Test learning potential with simple policy
    print("\n🎯 Testing learning potential with simple heuristic policy...")
    
    total_episodes = 20
    successful_episodes = 0
    progress_history = []
    
    for episode in range(total_episodes):
        obs, info = env.reset()
        episode_reward = 0
        max_progress = 0
        
        print(f"\n🎮 Episode {episode+1}/{total_episodes}")
        print(f"Initial EE position: {env.current_ee_pos}")
        print(f"First target: {env.target_pos}")
        print(f"Distance: {info['distance_to_target']:.3f}m")
        
        for step in range(env.max_episode_steps):
            # Simple heuristic: Move towards target
            current_pos = env.current_ee_pos
            target_pos = env.target_pos
            
            # Calculate direction (only X and Z, Y is fixed)
            direction = target_pos - current_pos
            direction_xz = np.array([direction[0], direction[2], 0.0])  # [dx, dz, pen]
            
            # Scale action
            action_scale = min(0.3, np.linalg.norm(direction_xz[:2]))  # Adaptive scaling
            if np.linalg.norm(direction_xz[:2]) > 0:
                action = direction_xz / np.linalg.norm(direction_xz[:2]) * action_scale
            else:
                action = np.array([0.0, 0.0, 0.0])
            
            # Clip action
            action = np.clip(action, -1, 1)
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            max_progress = max(max_progress, info['trajectory_progress'])
            
            distance = info['distance_to_target']
            print(f"  Step {step+1:2d}: Action={action}, Distance={distance:.3f}m, "
                  f"Progress={info['trajectory_progress']:.2f}, Reward={reward:.1f}")
            
            if info['is_success']:
                print(f"    ✅ SUCCESS at waypoint {env.current_trajectory_idx}!")
            
            if done or truncated:
                print(f"  Episode finished: Done={done}, Truncated={truncated}")
                break
        
        progress_history.append(max_progress)
        
        if max_progress > 0.5:  # Made significant progress
            successful_episodes += 1
            
        print(f"Episode reward: {episode_reward:.1f}")
        print(f"Max progress: {max_progress:.3f} ({max_progress*100:.1f}%)")
    
    # Results analysis
    success_rate = successful_episodes / total_episodes * 100
    avg_progress = np.mean(progress_history) * 100
    max_progress_achieved = np.max(progress_history) * 100
    
    print(f"\n📊 LEARNING POTENTIAL ANALYSIS:")
    print(f"   Episodes with >50% progress: {successful_episodes}/{total_episodes} ({success_rate:.1f}%)")
    print(f"   Average progress: {avg_progress:.1f}%")
    print(f"   Best progress achieved: {max_progress_achieved:.1f}%")
    
    # Verdict
    if success_rate > 50:
        print("✅ EXCELLENT: Environment is highly learnable!")
        print("   Robot can consistently reach waypoints with simple policy")
        print("   ➡️  Ready for RL training")
    elif avg_progress > 30:
        print("🟡 GOOD: Environment shows learning potential")
        print("   Robot can make progress, but needs optimization")
        print("   ➡️  Proceed with RL but may need tuning")
    elif avg_progress > 10:
        print("🟠 MODERATE: Limited learning potential")
        print("   Robot struggles but shows some capability")
        print("   ➡️  Need easier configuration or better rewards")
    else:
        print("❌ POOR: Very difficult to learn")
        print("   Robot cannot make meaningful progress")
        print("   ➡️  Major changes needed")

if __name__ == "__main__":
    test_basic_learning()
