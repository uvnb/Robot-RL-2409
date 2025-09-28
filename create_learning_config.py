#!/usr/bin/env python3
"""
Optimized robot configuration for successful learning
- Progressive difficulty: start easy, get harder
- Simple 8-point trajectory 
- Adaptive success threshold
- Proper reward scaling
"""

import numpy as np
from robot_4dof_env import Robot4DOFDrawingEnv

def modify_env_for_learning():
    """Create learning-optimized version of environment"""
    print("🛠️ Creating LEARNING OPTIMIZED environment...")
    
    # Read current environment file
    with open('robot_4dof_env.py', 'r') as f:
        content = f.read()
    
    # Key modifications for better learning
    modifications = [
        # 1. Easier success threshold progression
        ('success_threshold=0.05', 'success_threshold=0.15'),  # Start with 15cm instead of 5cm
        
        # 2. Shorter episodes for faster learning
        ('max_episode_steps=100', 'max_episode_steps=30'),
        
        # 3. Simple trajectory
        ('self._generate_circle_trajectory()', 'self._generate_simple_trajectory()'),
    ]
    
    modified_content = content
    for old, new in modifications:
        if old in modified_content:
            modified_content = modified_content.replace(old, new)
            print(f"✅ Modified: {old} -> {new}")
        else:
            print(f"⚠️ Not found: {old}")
    
    # Add simple trajectory generation method
    simple_trajectory_method = '''
    def _generate_simple_trajectory(self):
        """Generate simple 8-point trajectory for learning"""
        points = []
        
        # Simple square path - easy to learn
        center_x, center_z = 0.3, 0.4
        size = 0.15  # 15cm square
        
        # 8 points around square
        positions = [
            (center_x - size/2, center_z - size/2),  # Bottom-left
            (center_x, center_z - size/2),           # Bottom-middle
            (center_x + size/2, center_z - size/2),  # Bottom-right
            (center_x + size/2, center_z),           # Right-middle
            (center_x + size/2, center_z + size/2),  # Top-right
            (center_x, center_z + size/2),           # Top-middle
            (center_x - size/2, center_z + size/2),  # Top-left
            (center_x - size/2, center_z),           # Left-middle
        ]
        
        for x, z in positions:
            points.append(np.array([x, self.drawing_plane_y, z]))
        
        self.trajectory_points = points
        print(f"📍 Generated simple {len(points)}-point square trajectory")
        print(f"   Center: ({center_x}, {center_z}), Size: {size}m")'''
    
    # Insert the method before the last class closing
    if 'def _generate_simple_trajectory' not in modified_content:
        # Find a good place to insert (before _apply_domain_randomization)
        insert_pos = modified_content.find('    def _apply_domain_randomization(self)')
        if insert_pos != -1:
            modified_content = modified_content[:insert_pos] + simple_trajectory_method + '\n\n    ' + modified_content[insert_pos:]
            print("✅ Added _generate_simple_trajectory method")
        else:
            print("⚠️ Could not find insertion point for simple trajectory method")
    
    # Save modified environment
    with open('robot_4dof_env_learning.py', 'w') as f:
        f.write(modified_content)
    
    print("✅ Created robot_4dof_env_learning.py")
    return 'robot_4dof_env_learning.py'

def test_learning_config():
    """Test the learning-optimized configuration"""
    print("\n🧪 Testing LEARNING OPTIMIZED configuration...")
    
    # Import the modified environment
    import importlib.util
    spec = importlib.util.spec_from_file_location("robot_4dof_env_learning", "robot_4dof_env_learning.py")
    learning_env_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(learning_env_module)
    
    # Create environment
    env = learning_env_module.Robot4DOFDrawingEnv(
        max_episode_steps=30,
        success_threshold=0.15,  # 15cm - much easier than 5cm
        enable_domain_randomization=False
    )
    
    print(f"Success threshold: {env.success_threshold}m = {env.success_threshold*100}cm")
    print(f"Max episode steps: {env.max_episode_steps}")  
    print(f"Trajectory points: {len(env.trajectory_points)}")
    
    # Test one episode
    obs, info = env.reset()
    episode_reward = 0
    
    print(f"\nInitial position: {env.current_ee_pos}")
    print(f"First target: {env.target_pos}")
    print(f"Initial distance: {info['distance_to_target']:.3f}m = {info['distance_to_target']*100:.1f}cm")
    
    for step in range(env.max_episode_steps):
        # Simple greedy action towards target
        target_dir = env.target_pos - env.current_ee_pos
        action = np.array([target_dir[0], target_dir[2], 0.0]) * 0.2
        action = np.clip(action, -1, 1)
        
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        
        distance = info['distance_to_target']
        progress = info['trajectory_progress']
        
        if step < 10 or info['is_success'] or step % 5 == 0:  # Print less frequently
            print(f"  Step {step+1:2d}: Distance={distance:.3f}m ({distance*100:.1f}cm), "
                  f"Progress={progress:.3f} ({progress*100:.1f}%), Reward={reward:.1f}")
        
        if info['is_success']:
            print(f"    🎯 SUCCESS at waypoint {env.current_trajectory_idx}!")
            
        if done or truncated:
            print(f"  Episode finished at step {step+1}: Done={done}, Truncated={truncated}")
            break
    
    print(f"\nFinal Results:")
    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  Final progress: {info['trajectory_progress']:.3f} ({info['trajectory_progress']*100:.1f}%)")
    print(f"  Waypoints reached: {env.current_trajectory_idx}/{len(env.trajectory_points)-1}")
    
    if info['trajectory_progress'] > 0.3:  # 30% progress
        print("✅ EXCELLENT! Robot can learn with this configuration")
    elif info['trajectory_progress'] > 0.1:  # 10% progress  
        print("✅ GOOD! Robot shows learning potential")
    else:
        print("❌ Still too difficult - needs easier configuration")

if __name__ == "__main__":
    modify_env_for_learning()
    test_learning_config()
