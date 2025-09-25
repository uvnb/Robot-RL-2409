"""
4-DOF Robot Adapter for DDPG+HER trained model
Converts task-space commands to joint-space commands for 4-DOF robot
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any

class Robot4DOFAdapter:
    """
    Adapter class to use trained DDPG model with 4-DOF robot
    """
    
    def __init__(self, base_height=0.1, link_lengths=[0.2, 0.2, 0.15, 0.1]):
        """
        Initialize 4-DOF robot adapter
        
        Args:
            base_height: Height of robot base
            link_lengths: Lengths of robot links [L1, L2, L3, L4]
        """
        self.base_height = base_height
        self.link_lengths = link_lengths
        self.joint_limits = {
            'lower': np.array([-np.pi, -np.pi/2, -np.pi, -np.pi]),
            'upper': np.array([np.pi, np.pi/2, np.pi, np.pi])
        }
        
    def inverse_kinematics(self, target_pos: np.ndarray) -> np.ndarray:
        """
        Convert 3D end-effector position to joint angles
        
        Args:
            target_pos: [x, y, z] end-effector target position
            
        Returns:
            joint_angles: [q1, q2, q3, q4] joint angles in radians
        """
        x, y, z = target_pos
        
        # Adjust for base height
        z_adj = z - self.base_height
        
        # Joint 1: Base rotation
        q1 = np.arctan2(y, x)
        
        # Distance in XY plane
        r = np.sqrt(x**2 + y**2)
        
        # For 4-DOF planar arm in vertical plane
        L1, L2, L3, L4 = self.link_lengths
        
        # Distance from base to target (in vertical plane)
        target_dist = np.sqrt(r**2 + z_adj**2)
        
        # Check reachability
        max_reach = sum(self.link_lengths)
        min_reach = abs(self.link_lengths[0] - sum(self.link_lengths[1:]))
        
        if target_dist > max_reach or target_dist < min_reach:
            # Target unreachable, return closest valid configuration
            if target_dist > max_reach:
                # Scale down to max reach
                scale = max_reach / target_dist
                r *= scale
                z_adj *= scale
                target_dist = max_reach
        
        # Inverse kinematics for planar 4-DOF
        # Simplified approach - you may need to adjust based on your robot
        
        # Joint 2: Shoulder
        cos_q2 = (r**2 + z_adj**2 - L1**2 - (L2+L3+L4)**2) / (2*L1*(L2+L3+L4))
        cos_q2 = np.clip(cos_q2, -1, 1)
        q2 = np.arccos(cos_q2)
        
        # Joint 3: Elbow
        q3 = np.arctan2(z_adj, r) - q2
        
        # Joint 4: Wrist (assume horizontal end-effector)
        q4 = 0  # Can be adjusted based on desired orientation
        
        joint_angles = np.array([q1, q2, q3, q4])
        
        # Apply joint limits
        joint_angles = np.clip(joint_angles, 
                             self.joint_limits['lower'], 
                             self.joint_limits['upper'])
        
        return joint_angles
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Convert joint angles to end-effector position
        
        Args:
            joint_angles: [q1, q2, q3, q4] joint angles in radians
            
        Returns:
            end_effector_pos: [x, y, z] end-effector position
        """
        q1, q2, q3, q4 = joint_angles
        L1, L2, L3, L4 = self.link_lengths
        
        # Forward kinematics calculation
        # This is simplified - adjust based on your robot's DH parameters
        
        x = (L1*np.cos(q2) + L2*np.cos(q2+q3) + L3*np.cos(q2+q3+q4)) * np.cos(q1)
        y = (L1*np.cos(q2) + L2*np.cos(q2+q3) + L3*np.cos(q2+q3+q4)) * np.sin(q1)
        z = self.base_height + L1*np.sin(q2) + L2*np.sin(q2+q3) + L3*np.sin(q2+q3+q4)
        
        return np.array([x, y, z])
    
    def get_workspace_bounds(self) -> Dict[str, np.ndarray]:
        """
        Calculate workspace bounds for 4-DOF robot
        
        Returns:
            bounds: Dictionary with 'min' and 'max' bounds for [x, y, z]
        """
        max_reach = sum(self.link_lengths)
        min_reach = abs(self.link_lengths[0] - sum(self.link_lengths[1:]))
        
        bounds = {
            'min': np.array([-max_reach, -max_reach, self.base_height - max_reach]),
            'max': np.array([max_reach, max_reach, self.base_height + max_reach])
        }
        
        return bounds

class Robot4DOFEnvironment(gym.Env):
    """
    Custom Gym environment for 4-DOF robot compatible with trained DDPG model
    """
    
    def __init__(self, robot_adapter: Robot4DOFAdapter):
        super().__init__()
        
        self.robot = robot_adapter
        self.workspace_bounds = robot_adapter.get_workspace_bounds()
        
        # Action space: 3D end-effector position commands
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Observation space: compatible with PandaReach-v3
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(
                low=-10.0, high=10.0, shape=(6,), dtype=np.float32
            ),
            'achieved_goal': gym.spaces.Box(
                low=-10.0, high=10.0, shape=(3,), dtype=np.float32
            ),
            'desired_goal': gym.spaces.Box(
                low=-10.0, high=10.0, shape=(3,), dtype=np.float32
            )
        })
        
        # Initialize robot state
        self.current_joint_angles = np.zeros(4)
        self.current_position = self.robot.forward_kinematics(self.current_joint_angles)
        self.target_position = None
        
    def reset(self, seed=None, options=None):
        """Reset environment and return initial observation"""
        super().reset(seed=seed)
        
        # Reset robot to home position
        self.current_joint_angles = np.array([0, 0, 0, 0])
        self.current_position = self.robot.forward_kinematics(self.current_joint_angles)
        
        # Generate random target within workspace
        bounds = self.workspace_bounds
        self.target_position = np.random.uniform(
            bounds['min'] * 0.7,  # Stay within 70% of workspace
            bounds['max'] * 0.7,
            size=3
        )
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """Execute action and return observation, reward, done, info"""
        # Scale action from [-1, 1] to workspace bounds
        bounds = self.workspace_bounds
        scaled_action = bounds['min'] + (action + 1) * (bounds['max'] - bounds['min']) / 2
        
        # Convert to joint angles
        target_joint_angles = self.robot.inverse_kinematics(scaled_action)
        
        # Simulate robot movement (in real robot, send commands here)
        self.current_joint_angles = target_joint_angles
        self.current_position = self.robot.forward_kinematics(self.current_joint_angles)
        
        # Calculate reward (same as PandaReach-v3)
        reward = self.compute_reward(self.current_position, self.target_position, {})
        
        # Check if done
        distance_to_target = np.linalg.norm(self.current_position - self.target_position)
        done = distance_to_target < 0.05  # 5cm threshold
        
        observation = self._get_observation()
        info = {'distance_to_target': distance_to_target}
        
        return observation, reward, done, False, info
    
    def _get_observation(self):
        """Get current observation in PandaReach-v3 format"""
        # Create observation compatible with trained model
        # observation: [position, velocity] (6D)
        obs = np.concatenate([
            self.current_position,  # Current position (3D)
            np.zeros(3)  # Velocity (simplified as zeros)
        ])
        
        return {
            'observation': obs.astype(np.float32),
            'achieved_goal': self.current_position.astype(np.float32),
            'desired_goal': self.target_position.astype(np.float32)
        }
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute reward (same as original environment)"""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return 1.0 - distance

# Usage example:
if __name__ == "__main__":
    # Initialize 4-DOF robot adapter
    robot_adapter = Robot4DOFAdapter(
        base_height=0.1,
        link_lengths=[0.2, 0.2, 0.15, 0.1]  # Adjust to your robot
    )
    
    # Create environment
    env = Robot4DOFEnvironment(robot_adapter)
    
    # Test with trained agent
    # agent = DDPGAgent(env=env, input_dims=12)  # Same as before
    # agent.load_models()  # Load your trained weights
    
    print("4-DOF Robot Environment ready!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
