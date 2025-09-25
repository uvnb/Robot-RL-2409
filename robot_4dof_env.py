"""
4-DOF Robot Environment for training with DDPG + HER
Compatible with existing training scripts
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Any
import math

class Robot4DOFReachEnv(gym.Env):
    """
    4-DOF Robot Reach Environment for RL training
    Compatible with PandaReach-v3 interface
    """
    
    def __init__(self, 
                 base_height=0.1,
                 link_lengths=[0.25, 0.25, 0.2, 0.15],
                 max_episode_steps=50,
                 success_threshold=0.05):
        
        super().__init__()
        
        # Robot parameters
        self.base_height = base_height
        self.link_lengths = np.array(link_lengths)
        self.max_reach = sum(self.link_lengths)
        self.min_reach = abs(self.link_lengths[0] - sum(self.link_lengths[1:]))
        
        # Environment parameters
        self.max_episode_steps = max_episode_steps
        self.success_threshold = success_threshold
        
        # Joint limits (radians)
        self.joint_limits = np.array([
            [-np.pi, np.pi],        # Base rotation
            [-np.pi/2, np.pi/2],    # Shoulder
            [-np.pi, np.pi],        # Elbow
            [-np.pi, np.pi]         # Wrist
        ])
        
        # Action space: 3D end-effector position commands (normalized)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Observation space: compatible with PandaReach-v3
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(
                low=-10.0, high=10.0, shape=(8,), dtype=np.float32  # 4 joint pos + 4 joint vel
            ),
            'achieved_goal': gym.spaces.Box(
                low=-10.0, high=10.0, shape=(3,), dtype=np.float32
            ),
            'desired_goal': gym.spaces.Box(
                low=-10.0, high=10.0, shape=(3,), dtype=np.float32
            )
        })
        
        # State variables
        self.current_joint_pos = np.zeros(4)
        self.current_joint_vel = np.zeros(4)
        self.current_ee_pos = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.step_count = 0
        
        # Workspace bounds (safe area for training)
        self.workspace_center = np.array([0.0, 0.0, self.base_height + 0.3])
        self.workspace_radius = min(self.max_reach * 0.8, 0.6)  # Conservative workspace
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset robot to home position
        self.current_joint_pos = np.array([0.0, 0.0, 0.0, 0.0])
        self.current_joint_vel = np.zeros(4)
        self.step_count = 0
        
        # Calculate initial end-effector position
        self.current_ee_pos = self._forward_kinematics(self.current_joint_pos)
        
        # Generate random target within workspace
        self.target_pos = self._sample_target_position()
        
        observation = self._get_observation()
        info = {
            'is_success': False,
            'distance_to_target': np.linalg.norm(self.current_ee_pos - self.target_pos)
        }
        
        return observation, info
    
    def step(self, action):
        """Execute action and return new state"""
        # Scale action from [-1, 1] to workspace
        target_ee_pos = self._scale_action(action)
        
        # Inverse kinematics to get target joint angles
        target_joint_pos = self._inverse_kinematics(target_ee_pos)
        
        # Simple robot dynamics simulation
        self._simulate_robot_motion(target_joint_pos)
        
        # Update end-effector position
        self.current_ee_pos = self._forward_kinematics(self.current_joint_pos)
        
        # Calculate reward
        reward = self._compute_reward()
        
        # Check termination conditions
        self.step_count += 1
        distance_to_target = np.linalg.norm(self.current_ee_pos - self.target_pos)
        is_success = distance_to_target < self.success_threshold
        
        terminated = is_success
        truncated = self.step_count >= self.max_episode_steps
        
        observation = self._get_observation()
        info = {
            'is_success': is_success,
            'distance_to_target': distance_to_target
        }
        
        return observation, reward, terminated, truncated, info
    
    def _forward_kinematics(self, joint_angles):
        """Calculate end-effector position from joint angles"""
        q1, q2, q3, q4 = joint_angles
        L1, L2, L3, L4 = self.link_lengths
        
        # Forward kinematics for 4-DOF planar robot
        # Assuming robot moves in vertical plane after base rotation
        
        # Position in vertical plane (r, z)
        r = L1*np.cos(q2) + L2*np.cos(q2+q3) + L3*np.cos(q2+q3+q4)
        z = self.base_height + L1*np.sin(q2) + L2*np.sin(q2+q3) + L3*np.sin(q2+q3+q4)
        
        # Convert to 3D coordinates with base rotation
        x = r * np.cos(q1)
        y = r * np.sin(q1)
        
        return np.array([x, y, z])
    
    def _inverse_kinematics(self, target_pos):
        """Calculate joint angles from target end-effector position"""
        x, y, z = target_pos
        
        # Base rotation
        q1 = np.arctan2(y, x)
        
        # Distance in XY plane and adjust z for base height
        r = np.sqrt(x**2 + y**2)
        z_adj = z - self.base_height
        
        L1, L2, L3, L4 = self.link_lengths
        
        # Check reachability
        target_dist = np.sqrt(r**2 + z_adj**2)
        if target_dist > self.max_reach:
            # Scale to maximum reach
            scale = self.max_reach / target_dist
            r *= scale
            z_adj *= scale
            target_dist = self.max_reach
        
        # Inverse kinematics for planar 3-DOF (q2, q3, q4)
        # Simplified approach - treat last 3 links as 2-link system
        L23 = L2 + L3  # Combine links 2&3
        
        # Two-link inverse kinematics
        cos_q3_temp = (r**2 + z_adj**2 - L1**2 - L23**2) / (2*L1*L23)
        cos_q3_temp = np.clip(cos_q3_temp, -1, 1)
        
        q3_temp = np.arccos(cos_q3_temp)
        q2 = np.arctan2(z_adj, r) - np.arctan2(L23*np.sin(q3_temp), L1 + L23*np.cos(q3_temp))
        
        # Distribute q3_temp between actual q3 and q4
        q3 = q3_temp * 0.6  # Assign 60% to q3
        q4 = q3_temp * 0.4  # Assign 40% to q4
        
        joint_angles = np.array([q1, q2, q3, q4])
        
        # Apply joint limits
        for i in range(4):
            joint_angles[i] = np.clip(joint_angles[i], 
                                    self.joint_limits[i, 0], 
                                    self.joint_limits[i, 1])
        
        return joint_angles
    
    def _simulate_robot_motion(self, target_joint_pos):
        """Simulate robot motion towards target joint positions"""
        # Simple first-order dynamics
        max_joint_vel = 0.5  # rad/step
        
        # Calculate desired joint velocities
        joint_error = target_joint_pos - self.current_joint_pos
        desired_vel = joint_error * 0.3  # Proportional control
        
        # Limit joint velocities
        self.current_joint_vel = np.clip(desired_vel, -max_joint_vel, max_joint_vel)
        
        # Update joint positions
        self.current_joint_pos += self.current_joint_vel
        
        # Apply joint limits
        for i in range(4):
            self.current_joint_pos[i] = np.clip(self.current_joint_pos[i],
                                               self.joint_limits[i, 0],
                                               self.joint_limits[i, 1])
    
    def _scale_action(self, action):
        """Scale normalized action to workspace coordinates"""
        center = self.workspace_center
        radius = self.workspace_radius
        
        # Scale from [-1, 1] to workspace sphere
        scaled_pos = center + action * radius
        
        # Ensure within workspace bounds
        scaled_pos[2] = np.clip(scaled_pos[2], 
                               self.base_height + 0.1, 
                               self.base_height + self.max_reach)
        
        return scaled_pos
    
    def _sample_target_position(self):
        """Sample random target position within workspace"""
        # Sample within workspace sphere
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi/2)  # Upper hemisphere only
        r = np.random.uniform(0.2, self.workspace_radius)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta) 
        z = self.workspace_center[2] + r * np.cos(phi) * 0.5  # Bias towards reachable height
        
        return np.array([x, y, z])
    
    def _compute_reward(self):
        """Compute reward based on distance to target"""
        distance = np.linalg.norm(self.current_ee_pos - self.target_pos)
        
        # Dense reward (same as PandaReach-v3)
        reward = 1.0 - distance
        
        # Bonus for reaching target
        if distance < self.success_threshold:
            reward += 10.0
        
        # Small penalty for joint movement (encourage efficiency)
        joint_movement_penalty = 0.01 * np.sum(np.abs(self.current_joint_vel))
        reward -= joint_movement_penalty
        
        return reward
    
    def _get_observation(self):
        """Get current observation in PandaReach-v3 format"""
        # Observation: joint positions and velocities
        obs = np.concatenate([
            self.current_joint_pos,    # 4D joint positions
            self.current_joint_vel     # 4D joint velocities  
        ])
        
        return {
            'observation': obs.astype(np.float32),
            'achieved_goal': self.current_ee_pos.astype(np.float32),
            'desired_goal': self.target_pos.astype(np.float32)
        }
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute reward for HER (vectorized version)"""
        distances = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        rewards = 1.0 - distances
        
        # Add success bonus
        success_bonus = (distances < self.success_threshold) * 10.0
        rewards += success_bonus
        
        return rewards

# Register the environment
gym.register(
    id='Robot4DOFReach-v1',
    entry_point='robot_4dof_env:Robot4DOFReachEnv',
    max_episode_steps=50,
)
