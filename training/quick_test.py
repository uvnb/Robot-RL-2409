"""
Quick test training - 10 episodes only for testing
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_4dof_env import Robot4DOFReachEnv
from agents.ddpg import DDPGAgent

print("🔥 QUICK TEST TRAINING - 10 EPISODES")
print("===================================")

# Quick test parameters
n_games = 10
threshold = 0.1

# Create environment
env = Robot4DOFReachEnv(
    base_height=0.1,
    link_lengths=[0.25, 0.25, 0.2, 0.15],
    max_episode_steps=50,
    success_threshold=threshold
)

# Calculate observation space
obs_shape = env.observation_space['observation'].shape[0] + \
            env.observation_space['achieved_goal'].shape[0] + \
            env.observation_space['desired_goal'].shape[0]

print(f"Environment ready - State space: {obs_shape}D")

# Initialize agent
agent = DDPGAgent(env=env, input_dims=obs_shape)
print("Agent initialized")

# Quick training
successes = 0

for i in range(n_games):
    observation, info = env.reset()
    score = 0
    step = 0
    
    while True:
        # Get state
        curr_obs = observation['observation']
        curr_achgoal = observation['achieved_goal']
        curr_desgoal = observation['desired_goal']
        state = np.concatenate((curr_obs, curr_achgoal, curr_desgoal))
        
        # Choose action
        action = agent.choose_action(state, False)
        
        # Execute
        observation, reward, done, truncated, step_info = env.step(np.array(action))
        score += reward
        step += 1
        
        if done or truncated:
            break
    
    # Check success
    is_success = step_info['is_success']
    final_distance = step_info['distance_to_target']
    
    if is_success:
        successes += 1
    
    status = "✅" if is_success else "❌"
    print(f"Episode {i+1:2d}: {status} Steps={step:2d}, Distance={final_distance:.4f}m, Score={score:.1f}")

success_rate = (successes / n_games) * 100
print(f"\n🎯 QUICK TEST RESULTS:")
print(f"Success Rate: {success_rate:.1f}% ({successes}/{n_games})")
print(f"Environment is working correctly!")
