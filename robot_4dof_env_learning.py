"""
4-DOF Robot Environment for 2D Drawing/Trajectory Following
Optimized for DDPG + HER with continuous actions
Features: Dense reward, domain randomization, trajectory following
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Any, List
import math
import random

class Robot4DOFDrawingEnv(gym.Env):
    """
    4-DOF Robot Drawing Environment for RL training
    Optimized for 2D trajectory following on vertical plane (fixed Y)
    Features: Dense rewards, domain randomization, visual integration ready
    """
    
    def __init__(self, 
                 base_height=0.1,
                 link_lengths=[0.25, 0.25, 0.2, 0.15],
                 max_episode_steps=30,  # Increased for drawing tasks
                 success_threshold=0.15,  # INCREASED from 0.02 to 0.05 for easier learning
                 drawing_plane_y=0.4,    # Fixed Y coordinate for 2D drawing
                 enable_domain_randomization=True):
        
        super().__init__()
        
        # Robot parameters with randomization capability
        self.base_height = base_height
        self.nominal_link_lengths = np.array(link_lengths)
        self.link_lengths = self.nominal_link_lengths.copy()
        self.max_reach = sum(self.nominal_link_lengths)
        self.min_reach = abs(self.nominal_link_lengths[0] - sum(self.nominal_link_lengths[1:]))
        
        # Drawing specific parameters
        self.drawing_plane_y = drawing_plane_y  # Fixed Y for 2D drawing
        self.enable_domain_randomization = enable_domain_randomization
        
        # Environment parameters
        self.max_episode_steps = max_episode_steps
        self.success_threshold = success_threshold
        self.base_success_threshold = success_threshold  # Store original
        self.trajectory_points = []  # Will store drawing trajectory
        self.current_trajectory_idx = 0
        
        # Progressive difficulty parameters
        self.adaptive_threshold = True
        self.min_threshold = 0.05  # Start easier (5cm)
        self.max_threshold = success_threshold  # End target (2cm)
        
        # Domain randomization parameters
        self.dynamics_randomization = {
            'link_length_noise': 0.05,    # ±5% variation
            'joint_friction': 0.02,       # Joint friction coefficient
            'servo_lag': 0.1,             # Servo response delay
            'gravity_noise': 0.05,        # Gravity variation
            'observation_noise': 0.01     # Sensor noise
        }
        
        # Joint limits (radians) 
        self.joint_limits = np.array([
            [-np.pi, np.pi],        # Base rotation
            [-np.pi/3, np.pi/2],    # Shoulder (more restrictive for drawing)
            [-np.pi, np.pi],        # Elbow
            [-np.pi/2, np.pi/2]     # Wrist (limited for stability)
        ])
        
        # Action space: 2D movement commands + pen pressure (for future camera integration)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32  # [dx, dz, pen_pressure]
        )
        
        # Enhanced observation space for drawing task
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(
                low=-10.0, high=10.0, shape=(12,), dtype=np.float32  
                # [4 joint pos, 4 joint vel, 3 ee_pos, trajectory_progress]
            ),
            'achieved_goal': gym.spaces.Box(
                low=-10.0, high=10.0, shape=(3,), dtype=np.float32  # Current end-effector position
            ),
            'desired_goal': gym.spaces.Box(
                low=-10.0, high=10.0, shape=(3,), dtype=np.float32  # Next trajectory waypoint
            )
        })
        
        # State variables
        self.current_joint_pos = np.zeros(4)
        self.current_joint_vel = np.zeros(4) 
        self.prev_joint_pos = np.zeros(4)  # For smoothness penalty
        self.current_ee_pos = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.step_count = 0
        
        # Drawing workspace (vertical plane at fixed Y)
        self.workspace_bounds = {
            'x': [-0.3, 0.3],  # Left-right range
            'z': [self.base_height + 0.1, self.base_height + 0.5],  # Up-down range
            'y': self.drawing_plane_y  # Fixed depth
        }
        
        # Initialize with simple circle trajectory for testing
        self._generate_simple_trajectory()
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state with domain randomization"""
        super().reset(seed=seed)
        
        # Apply domain randomization
        if self.enable_domain_randomization:
            self._apply_domain_randomization()
        
        # Reset robot to random start position (near trajectory start)
        self.current_trajectory_idx = 0
        start_noise = np.random.uniform(-0.05, 0.05, 3) if self.enable_domain_randomization else np.zeros(3)
        
        # Initialize near first trajectory point
        if len(self.trajectory_points) > 0:
            start_pos = self.trajectory_points[0] + start_noise
            self.current_joint_pos = self._inverse_kinematics(start_pos)
        else:
            self.current_joint_pos = np.array([0.0, 0.2, -0.3, 0.1])  # Safe drawing position
        
        self.current_joint_vel = np.zeros(4)
        self.prev_joint_pos = self.current_joint_pos.copy()
        self.step_count = 0
        
        # Calculate initial end-effector position
        self.current_ee_pos = self._forward_kinematics(self.current_joint_pos)
        
        # Set target as first trajectory point
        self.target_pos = self._get_current_target()
        
        observation = self._get_observation()
        info = {
            'is_success': False,
            'distance_to_target': np.linalg.norm(self.current_ee_pos - self.target_pos),
            'trajectory_progress': self.current_trajectory_idx / max(len(self.trajectory_points)-1, 1)
        }
        
        return observation, info
    
    def step(self, action):
        """Execute action and return new state with dense rewards"""
        # Store previous position for smoothness penalty
        self.prev_joint_pos = self.current_joint_pos.copy()
        
        # Scale action: [dx, dz, pen_pressure] -> world coordinates  
        dx, dz, pen_pressure = action * 0.05  # Scale movement to 5cm max per step
        
        # Calculate target end-effector position (fixed Y for 2D drawing)
        target_ee_pos = self.current_ee_pos.copy()
        target_ee_pos[0] += dx  # X movement
        target_ee_pos[2] += dz  # Z movement  
        target_ee_pos[1] = self.drawing_plane_y  # Fixed Y
        
        # Clamp to workspace bounds
        target_ee_pos = self._clamp_to_workspace(target_ee_pos)
        
        # Inverse kinematics to get target joint angles
        target_joint_pos = self._inverse_kinematics(target_ee_pos)
        
        # Simple robot dynamics simulation with domain randomization
        self._simulate_robot_motion(target_joint_pos)
        
        # Update end-effector position
        self.current_ee_pos = self._forward_kinematics(self.current_joint_pos)
        
        # Calculate dense reward for drawing task
        reward = self._compute_drawing_reward_improved(action)
        
        # Check termination conditions
        self.step_count += 1
        distance_to_target = np.linalg.norm(self.current_ee_pos[:2] - self.target_pos[:2])  # Only X-Z distance
        is_success = distance_to_target < self.success_threshold
        
        # IMPROVED: More generous trajectory progression
        if is_success:
            if self.current_trajectory_idx < len(self.trajectory_points) - 1:
                self.current_trajectory_idx += 1
                self.target_pos = self._get_current_target()
                print(f"🎯 Waypoint {self.current_trajectory_idx}/{len(self.trajectory_points)-1} reached!")
            elif self.current_trajectory_idx == len(self.trajectory_points) - 1:
                # Reached final waypoint - mark as completed
                self.current_trajectory_idx += 1  # Move beyond final to indicate completion
                print(f"🏁 FINAL Waypoint {len(self.trajectory_points)}/8 reached! Square complete!")
        elif distance_to_target < self.success_threshold * 1.5:  # Near success
            # Give partial credit and maybe advance anyway
            if np.random.random() < 0.3:  # 30% chance to advance when close
                if self.current_trajectory_idx < len(self.trajectory_points) - 1:
                    self.current_trajectory_idx += 1
                    self.target_pos = self._get_current_target()
                    print(f"📍 Near waypoint - advancing to {self.current_trajectory_idx}/{len(self.trajectory_points)-1}")
            
        # Episode completion (FIXED: Allow final waypoint to be tested)
        # Complete when robot moves beyond the final waypoint (index > 7)
        trajectory_complete = self.current_trajectory_idx > len(self.trajectory_points) - 1
        terminated = trajectory_complete
        truncated = self.step_count >= self.max_episode_steps
        
        observation = self._get_observation()
        info = {
            'is_success': is_success,
            'distance_to_target': distance_to_target,
            'trajectory_progress': self.current_trajectory_idx / max(len(self.trajectory_points)-1, 1),
            'trajectory_complete': trajectory_complete
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
        # Sample within reachable workspace for robot
        theta = np.random.uniform(0, 2*np.pi)  # Full rotation around base
        
        # Use curriculum max distance but ensure it's within robot reach
        max_radius = min(self.workspace_radius, self.max_distance_from_origin)
        r_horizontal = np.random.uniform(0.1, max_radius * 0.8)  # Conservative horizontal reach
        
        # Height should be within robot's realistic vertical reach
        z_min = self.base_height + 0.05  # Just above base  
        z_max = self.base_height + 0.3   # Conservative for curriculum learning (0.1 + 0.3 = 0.4m max)
        z = np.random.uniform(z_min, z_max)
        
        x = r_horizontal * np.cos(theta)
        y = r_horizontal * np.sin(theta)
        
        target = np.array([x, y, z])
        target_distance_from_origin = np.linalg.norm(target)
        
        # print(f"Debug: max_distance_from_origin={self.max_distance_from_origin:.3f}, r_horizontal={r_horizontal:.3f}, z={z:.3f}")
        # print(f"Debug: target={target}, distance_from_origin={target_distance_from_origin:.3f}")
        
        return target
    
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
        """Get current observation with enhanced features for drawing"""
        # Add observation noise if domain randomization is enabled
        noise = 0
        if self.enable_domain_randomization:
            noise = np.random.normal(0, self.dynamics_randomization['observation_noise'], 12)
        
        # Enhanced observation: [joint_pos(4), joint_vel(4), ee_pos(3), trajectory_progress(1)]
        obs = np.concatenate([
            self.current_joint_pos,
            self.current_joint_vel, 
            self.current_ee_pos,
            [self.current_trajectory_idx / max(len(self.trajectory_points)-1, 1)]
        ]).astype(np.float32) + noise
        
        return {
            'observation': obs,
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
    
    # ==================== NEW METHODS FOR DRAWING TASK ====================
    
    def _generate_circle_trajectory(self, radius=0.15, center=None, num_points=50):
        """Generate circular trajectory for drawing"""
        if center is None:
            center = [0.0, self.drawing_plane_y, self.base_height + 0.25]
        
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        self.trajectory_points = []
        
        for angle in angles:
            x = center[0] + radius * np.cos(angle)
            y = center[1]  # Fixed Y for 2D drawing
            z = center[2] + radius * np.sin(angle)
            self.trajectory_points.append(np.array([x, y, z]))
        
        # Close the circle
        self.trajectory_points.append(self.trajectory_points[0])
    
    def _generate_square_trajectory(self, size=0.2, center=None, num_points_per_side=12):
        """Generate square trajectory for drawing"""
        if center is None:
            center = [0.0, self.drawing_plane_y, self.base_height + 0.25]
            
        half_size = size / 2
        self.trajectory_points = []
        
        # Define square corners
        corners = [
            [center[0] - half_size, center[1], center[2] - half_size],  # Bottom-left
            [center[0] + half_size, center[1], center[2] - half_size],  # Bottom-right
            [center[0] + half_size, center[1], center[2] + half_size],  # Top-right
            [center[0] - half_size, center[1], center[2] + half_size],  # Top-left
        ]
        
        # Generate points along each side
        for i in range(4):
            start = np.array(corners[i])
            end = np.array(corners[(i + 1) % 4])
            
            for j in range(num_points_per_side):
                t = j / num_points_per_side
                point = start + t * (end - start)
                self.trajectory_points.append(point)
        
        # Close the square
        self.trajectory_points.append(self.trajectory_points[0])
    

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
        print(f"   Center: ({center_x}, {center_z}), Size: {size}m")

    def _apply_domain_randomization(self):
        """Apply domain randomization for robustness"""
        # Randomize link lengths (±5%)
        noise_scale = self.dynamics_randomization['link_length_noise']
        length_multipliers = 1 + np.random.uniform(-noise_scale, noise_scale, 4)
        self.link_lengths = self.nominal_link_lengths * length_multipliers
        
        # Randomize drawing plane position slightly
        y_noise = np.random.uniform(-0.05, 0.05)
        self.drawing_plane_y = self.workspace_bounds['y'] + y_noise
    
    def _get_current_target(self):
        """Get current trajectory target point"""
        if len(self.trajectory_points) == 0:
            return np.array([0.0, self.drawing_plane_y, self.base_height + 0.25])
        
        idx = min(self.current_trajectory_idx, len(self.trajectory_points) - 1)
        return self.trajectory_points[idx].copy()
    
    def _clamp_to_workspace(self, pos):
        """Clamp position to workspace bounds"""
        pos[0] = np.clip(pos[0], self.workspace_bounds['x'][0], self.workspace_bounds['x'][1])
        pos[1] = self.drawing_plane_y  # Always fixed Y
        pos[2] = np.clip(pos[2], self.workspace_bounds['z'][0], self.workspace_bounds['z'][1])
        return pos
    
    def _compute_drawing_reward_improved(self, action):
        """
        SIMPLIFIED: Much easier reward function for learning
        """
        # 1. Large base reward
        base_reward = 2.0
        
        # 2. Primary distance reward - VERY GENEROUS
        distance_to_target = np.linalg.norm(self.current_ee_pos[:2] - self.target_pos[:2])
        # Linear reward instead of exponential - easier gradient
        max_reward_distance = 0.15  # Give reward up to 15cm away
        distance_reward = max(0, (max_reward_distance - distance_to_target) / max_reward_distance * 5.0)
        
        # 3. HUGE success bonus
        success_bonus = 0.0
        if distance_to_target < self.success_threshold:
            success_bonus = 20.0  # Much larger
            if self.current_trajectory_idx >= len(self.trajectory_points) - 1:
                success_bonus += 50.0  # Huge completion bonus
        
        # 4. Progressive bonus for just getting closer
        if distance_to_target < self.success_threshold * 2:  # Within 16cm
            success_bonus += 5.0
        if distance_to_target < self.success_threshold * 1.5:  # Within 12cm  
            success_bonus += 10.0
        
        # 5. MINIMAL penalties
        small_penalties = 0.01 * np.sum(np.abs(action))
        
        # 6. Movement reward - encourage any movement toward target
        if hasattr(self, 'prev_ee_pos'):
            prev_distance = np.linalg.norm(self.prev_ee_pos[:2] - self.target_pos[:2])
            if distance_to_target < prev_distance:  # Moving closer
                movement_reward = 1.0
            else:
                movement_reward = -0.1  # Small penalty for moving away
        else:
            movement_reward = 0
        self.prev_ee_pos = self.current_ee_pos.copy()
        
        total_reward = base_reward + distance_reward + success_bonus + movement_reward - small_penalties
        return total_reward
    
    def set_trajectory(self, trajectory_type="circle", **kwargs):
        """Set trajectory for drawing task"""
        if trajectory_type == "circle":
            self._generate_circle_trajectory(**kwargs)
        elif trajectory_type == "square":
            self._generate_square_trajectory(**kwargs)
        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")
        
        self.current_trajectory_idx = 0

# Register the environment
gym.register(
    id='Robot4DOFDrawing-v1',
    entry_point='robot_4dof_env:Robot4DOFDrawingEnv',
    max_episode_steps=30,
)
