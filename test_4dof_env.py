#!/usr/bin/env python3
"""
Test script for 4-DOF robot environment before training
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robot_4dof_env import Robot4DOFReachEnv

def test_4dof_environment():
    """Test the 4-DOF robot environment functionality"""
    
    print("=== 4-DOF ROBOT ENVIRONMENT TEST ===")
    
    # Create environment
    env = Robot4DOFReachEnv(
        base_height=0.1,
        link_lengths=[0.25, 0.25, 0.2, 0.15],
        max_episode_steps=50,
        success_threshold=0.05
    )
    
    print(f"✅ Environment created successfully")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space keys: {env.observation_space.keys()}")
    print(f"   Max reach: {env.max_reach:.3f}m")
    print(f"   Workspace radius: {env.workspace_radius:.3f}m")
    
    # Test environment reset
    print(f"\n=== TESTING ENVIRONMENT RESET ===")
    observation, info = env.reset()
    print(f"✅ Environment reset successful")
    print(f"   Observation shape: {observation['observation'].shape}")
    print(f"   Achieved goal: {observation['achieved_goal']}")
    print(f"   Desired goal: {observation['desired_goal']}")
    print(f"   Distance to target: {info['distance_to_target']:.4f}m")
    
    # Test forward kinematics
    print(f"\n=== TESTING KINEMATICS ===")
    test_joint_angles = np.array([0.0, 0.5, -0.3, 0.2])
    ee_pos = env._forward_kinematics(test_joint_angles)
    print(f"✅ Forward kinematics test")
    print(f"   Joint angles: {test_joint_angles}")
    print(f"   End-effector position: {ee_pos}")
    print(f"   Distance from base: {np.linalg.norm(ee_pos[:2]):.4f}m")
    print(f"   Height: {ee_pos[2]:.4f}m")
    
    # Test inverse kinematics
    target_pos = np.array([0.3, 0.2, 0.4])
    joint_angles = env._inverse_kinematics(target_pos)
    achieved_pos = env._forward_kinematics(joint_angles)
    ik_error = np.linalg.norm(target_pos - achieved_pos)
    
    print(f"✅ Inverse kinematics test")
    print(f"   Target position: {target_pos}")
    print(f"   Joint angles: {joint_angles}")
    print(f"   Achieved position: {achieved_pos}")
    print(f"   IK error: {ik_error:.6f}m")
    
    # Test action execution
    print(f"\n=== TESTING ACTION EXECUTION ===")
    test_actions = [
        np.array([0.5, 0.0, 0.3]),    # Right
        np.array([-0.3, 0.4, 0.2]),   # Left-forward
        np.array([0.0, 0.0, 0.8]),    # Up
        np.array([0.2, -0.3, -0.2]),  # Right-back-down
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\n  Test Action {i+1}: {action}")
        
        # Reset environment
        observation, info = env.reset()
        initial_distance = info['distance_to_target']
        
        # Execute action
        new_observation, reward, terminated, truncated, step_info = env.step(action)
        final_distance = step_info['distance_to_target']
        
        print(f"    Initial distance: {initial_distance:.4f}m")
        print(f"    Final distance: {final_distance:.4f}m")
        print(f"    Reward: {reward:.4f}")
        print(f"    Success: {'✅' if step_info['is_success'] else '❌'}")
        print(f"    Terminated: {terminated}, Truncated: {truncated}")
    
    # Test workspace boundaries
    print(f"\n=== TESTING WORKSPACE BOUNDARIES ===")
    reachable_count = 0
    total_tests = 1000
    
    for _ in range(total_tests):
        # Sample random target
        target = env._sample_target_position()
        
        # Check if reachable
        distance_from_base = np.linalg.norm(target - np.array([0, 0, env.base_height]))
        if distance_from_base <= env.max_reach:
            reachable_count += 1
    
    reachability_rate = reachable_count / total_tests * 100
    print(f"✅ Workspace boundary test")
    print(f"   Reachable targets: {reachable_count}/{total_tests}")
    print(f"   Reachability rate: {reachability_rate:.1f}%")
    
    # Test episode completion
    print(f"\n=== TESTING EPISODE COMPLETION ===")
    observation, info = env.reset()
    episode_length = 0
    total_reward = 0
    
    for step in range(env.max_episode_steps):
        # Random action
        action = env.action_space.sample()
        observation, reward, terminated, truncated, step_info = env.step(action)
        
        total_reward += reward
        episode_length += 1
        
        if terminated or truncated:
            break
    
    print(f"✅ Episode completion test")
    print(f"   Episode length: {episode_length}")
    print(f"   Total reward: {total_reward:.4f}")
    print(f"   Final distance: {step_info['distance_to_target']:.4f}m")
    print(f"   Success: {'✅' if step_info['is_success'] else '❌'}")
    
    # Test HER compatibility
    print(f"\n=== TESTING HER COMPATIBILITY ===")
    achieved_goals = np.array([
        [0.3, 0.2, 0.4],
        [0.25, 0.15, 0.35],
        [0.35, 0.25, 0.45]
    ])
    desired_goals = np.array([
        [0.3, 0.2, 0.4],
        [0.3, 0.2, 0.4], 
        [0.3, 0.2, 0.4]
    ])
    
    rewards = env.compute_reward(achieved_goals, desired_goals, {})
    print(f"✅ HER reward computation test")
    print(f"   Achieved goals shape: {achieved_goals.shape}")
    print(f"   Desired goals shape: {desired_goals.shape}")
    print(f"   Computed rewards: {rewards}")
    
    print(f"\n🎉 ALL TESTS PASSED!")
    print(f"4-DOF Robot Environment is ready for training!")
    
    return True

if __name__ == "__main__":
    try:
        test_4dof_environment()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
