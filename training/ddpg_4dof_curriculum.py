"""
Optimized DDPG+HER Training for 4-DOF Robot Drawing Task
Features: Enhanced reward, domain randomization, trajectory following
Fine-tuned for stability and performance on drawing tasks
"""

import numpy as np
import gymnasium as gym
import sys
import os
import random
import torch

# Add parent directory to path to import custom environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_4dof_env import Robot4DOFDrawingEnv
from agents.ddpg import DDPGAgent
from utils.HER import her_augmentation
import matplotlib.pyplot as plt


class DrawingCurriculumManager:
    """Manages curriculum learning for drawing tasks with increasing complexity"""
    
    def __init__(self):
        self.stage = 0
        self.stages = [
            {
                "name": "🎯 Stage 1: Simple Circles", 
                "trajectory_type": "circle",
                "trajectory_params": {"radius": 0.10, "num_points": 20},
                "success_threshold": 0.025,
                "episodes": 50,
                "domain_randomization": False
            },
            {
                "name": "⭕ Stage 2: Medium Circles + Noise", 
                "trajectory_type": "circle",
                "trajectory_params": {"radius": 0.15, "num_points": 30},
                "success_threshold": 0.020,
                "episodes": 60,
                "domain_randomization": True
            },
            {
                "name": "⬜ Stage 3: Squares", 
                "trajectory_type": "square",
                "trajectory_params": {"size": 0.20, "num_points_per_side": 10},
                "success_threshold": 0.020,
                "episodes": 70,
                "domain_randomization": True
            },
            {
                "name": "🔥 Stage 4: Complex Shapes + Full Randomization", 
                "trajectory_type": "circle",
                "trajectory_params": {"radius": 0.18, "num_points": 40},
                "success_threshold": 0.015,
                "episodes": 60,
                "domain_randomization": True
            }
        ]
        self.episodes_in_stage = 0
        self.stage_success_rates = []
        self.stage_rewards = []
        
    def get_current_params(self):
        """Get current curriculum parameters"""
        if self.stage >= len(self.stages):
            return self.stages[-1]
        return self.stages[self.stage]
    
    def should_advance(self, recent_success_rate, recent_avg_reward, min_success_rate=0.4):
        """Enhanced advancement criteria for drawing tasks"""
        current_stage = self.get_current_params()
        
        print(f"📊 Stage progress: {self.episodes_in_stage}/{current_stage['episodes']} episodes")
        print(f"   Success rate: {recent_success_rate:.2f}, Avg reward: {recent_avg_reward:.2f}")
        
        # Advance if performance is good and minimum episodes completed
        min_episodes = int(current_stage["episodes"] * 0.7)  # Complete 70% of episodes
        performance_good = recent_success_rate >= min_success_rate and recent_avg_reward > 0.5
        
        if performance_good and self.episodes_in_stage >= min_episodes:
            print(f"✅ Performance criteria met! Advancing to next stage...")
            return True
            
        # Force advance if completed all episodes for this stage  
        if self.episodes_in_stage >= current_stage["episodes"]:
            print(f"⏰ Stage time limit reached. Advancing anyway...")
            return True
            
        return False
            
        return False
    
    def advance_stage(self, success_rate, avg_reward):
        """Advance to next curriculum stage"""
        self.stage_success_rates.append(success_rate)
        self.stage_rewards.append(avg_reward)
        
        if self.stage < len(self.stages) - 1:
            self.stage += 1
            self.episodes_in_stage = 0
            print(f"\n🚀 ADVANCING TO {self.stages[self.stage]['name']}")
            print(f"   New success threshold: {self.stages[self.stage]['success_threshold']}")
            print(f"   New trajectory: {self.stages[self.stage]['trajectory_type']}")
            print(f"   Domain randomization: {self.stages[self.stage]['domain_randomization']}")
            return True
        else:
            print(f"\n🎉 CURRICULUM COMPLETED! All stages finished.")
            return False
            print(f"Completed: {current_stage['name']} (Success: {recent_success_rate*100:.1f}%)")
            print(f"Starting: {next_stage['name']}")
            print(f"New parameters - Max distance: {next_stage['max_distance']}m, Threshold: {next_stage['threshold']}m")
            return True
        else:
            print(f"\n🏆 CURRICULUM COMPLETED!")
            print(f"All stages finished. Continuing with final parameters.")
            return False
    
    def update_episode(self):
        """Update episode counter for current stage"""
        self.episodes_in_stage += 1


if __name__ == "__main__":

    print("=== 4-DOF ROBOT CURRICULUM TRAINING WITH DDPG + HER ===")
    
    # Initialize curriculum manager
    curriculum = CurriculumManager()
    
    # Training parameters
    max_total_episodes = 150
    opt_steps = 64
    best_score = 0
    
    # Tracking variables
    score_history = []
    avg_score_history = []
    actor_loss_history = []
    critic_loss_history = []
    success_history = []
    distance_to_goal_history = []
    curriculum_stages = []

    # Success tracking
    num_success = 0
    noise_decay_rate = 0.995
    
    # Early stopping parameters
    early_stop_threshold = 0.7  # Higher threshold for curriculum learning
    stability_window = 20

    print(f"Training Parameters:")
    print(f"- Max episodes: {max_total_episodes}")
    print(f"- Optimization steps per episode: {opt_steps}")
    print(f"- Curriculum stages: {len(curriculum.stages)}")
    
    # Print curriculum stages
    for i, stage in enumerate(curriculum.stages):
        print(f"  Stage {i+1}: {stage['name']}")
        print(f"    Max distance: {stage['max_distance']}m, Threshold: {stage['threshold']}m, Episodes: {stage['episodes']}")

    # Create 4-DOF robot environment with initial curriculum parameters
    initial_params = curriculum.get_current_params()
    env = Robot4DOFReachEnv(
        success_threshold=initial_params["threshold"]
    )
    # Set curriculum parameters properly
    env.max_distance_from_origin = initial_params["max_distance"]
    env.success_threshold = initial_params["threshold"]  # Make sure both are set
    
    # Initialize DDPG agent
    # State dimension: observation(8) + achieved_goal(3) + desired_goal(3) = 14
    state_dim = 14
    agent = DDPGAgent(env=env, input_dims=state_dim)

    print(f"\nEnvironment Info:")
    print(f"- Observation space: {env.observation_space}")
    print(f"- Action space: {env.action_space.shape}")
    print(f"- State dimension: {state_dim}")

    print(f"\nStarting curriculum stage: {initial_params['name']}")
    print(f"Initial parameters - Max distance: {initial_params['max_distance']}m, Threshold: {initial_params['threshold']}m")

    # Training loop
    for i in range(max_total_episodes):
        
        # Update environment parameters every episode to ensure consistency
        current_params = curriculum.get_current_params()
        env.max_distance_from_origin = current_params["max_distance"]
        env.success_threshold = current_params["threshold"]

        # Episode variables
        obs_array = []
        actions_array = []
        new_obs_array = []
        score = 0
        step = 0
        actor_loss_episode = None
        critic_loss_episode = None
        
        observation, info = env.reset()

        while True:
            # Get state
            curr_obs = observation['observation']
            curr_achgoal = observation['achieved_goal']
            curr_desgoal = observation['desired_goal']
            state = np.concatenate((curr_obs, curr_achgoal, curr_desgoal))

            # Choose action
            action = agent.choose_action(state, False)

            # Execute action
            new_observation, reward, done, truncated, step_info = env.step(np.array(action))
            
            # Get new state
            next_obs = new_observation['observation']
            next_achgoal = new_observation['achieved_goal'] 
            next_desgoal = new_observation['desired_goal']
            new_state = np.concatenate((next_obs, next_achgoal, next_desgoal))

            # Store experience
            agent.remember(state, action, reward, new_state, done)
        
            # Store for HER
            obs_array.append(observation)
            actions_array.append(action)
            new_obs_array.append(new_observation)

            observation = new_observation
            score += reward
            step += 1

            if done or truncated:
                break

        # Check success
        final_distance = step_info['distance_to_target']
        is_success = step_info['is_success']
        
        if is_success:
            num_success += 1

        success_history.append(int(is_success))
        distance_to_goal_history.append(final_distance)
        curriculum_stages.append(curriculum.stage)

        # HER augmentation
        her_augmentation(agent, obs_array, actions_array, new_obs_array)

        # Apply noise decay
        if i > 60:
            agent.noise_factor = max(agent.noise_factor * noise_decay_rate, 0.05)

        # Train agent
        for _ in range(opt_steps):
            actor_loss, critic_loss = agent.learn()
            if actor_loss is not None:
                actor_loss_episode = actor_loss
            if critic_loss is not None:
                critic_loss_episode = critic_loss

        # Update curriculum
        curriculum.update_episode()
        
        # Check curriculum advancement
        if curriculum.episodes_in_stage >= 20:  # Check after minimum episodes
            recent_success_rate = np.mean(success_history[-min(20, len(success_history)):])
            if curriculum.should_advance(recent_success_rate):
                curriculum.advance_stage(recent_success_rate)

        # Logging
        current_params = curriculum.get_current_params()
        stage_info = f"[{current_params['name']}]"
        
        if actor_loss_episode is not None and critic_loss_episode is not None:
            print(f"Episode {i}: {stage_info} Steps={step}, Score={score:.1f}, Distance={final_distance:.4f}m, Success={'✅' if is_success else '❌'}")
            print(f"          Actor Loss={actor_loss_episode:.4f}, Critic Loss={critic_loss_episode:.4f}")
        else:
            print(f"Episode {i}: {stage_info} Steps={step}, Score={score:.1f}, Distance={final_distance:.4f}m, Success={'✅' if is_success else '❌'}")
            
        # Track performance
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)
        actor_loss_history.append(actor_loss_episode)
        critic_loss_history.append(critic_loss_episode)

        if avg_score > best_score:
            best_score = avg_score

        # Check for early stopping
        if i >= stability_window:
            recent_success_rate = np.mean(success_history[-stability_window:])
            if recent_success_rate >= early_stop_threshold:
                print(f"\n🎯 EARLY STOPPING TRIGGERED!")
                print(f"Success rate ({recent_success_rate*100:.1f}%) exceeded threshold ({early_stop_threshold*100:.1f}%)")
                print(f"over the last {stability_window} episodes.")
                print(f"Training stopped at episode {i+1}")
                break

        # Progress update every 25 episodes
        if (i + 1) % 25 == 0:
            current_success_rate = (num_success / (i + 1)) * 100
            stage_episodes = curriculum.episodes_in_stage
            stage_name = current_params['name']
            
            print(f"\n=== PROGRESS UPDATE (Episode {i+1}) ===")
            print(f"Current Stage: {stage_name} (Episode {stage_episodes})")
            print(f"Average Score: {avg_score:.2f}")
            print(f"Success Rate: {current_success_rate:.1f}%")
            print(f"Best Score: {best_score:.2f}")
            if i >= stability_window:
                recent_success_rate = np.mean(success_history[-stability_window:]) * 100
                print(f"Recent Success Rate (last {stability_window}): {recent_success_rate:.1f}%")
            print("=" * 50)

    # Final statistics
    final_success_rate = num_success / (i + 1)
    total_episodes = i + 1
    print(f"\n=== CURRICULUM TRAINING COMPLETED ===")
    print(f"Episodes Completed: {total_episodes}/{max_total_episodes}")
    print(f"Final Success Rate: {final_success_rate*100:.2f}%")
    print(f"Total Successes: {num_success}/{total_episodes}")
    print(f"Best Average Score: {best_score:.2f}")
    
    if total_episodes < max_total_episodes:
        print(f"✅ Training stopped early due to reaching success threshold!")
    else:
        print(f"Training completed all planned episodes.")
    
    # Print curriculum progression
    print(f"\n📚 Curriculum Progression:")
    for stage_info in curriculum.stage_success_rates:
        print(f"  {stage_info['name']}: {stage_info['episodes']} episodes, {stage_info['success_rate']*100:.1f}% success")

    # Save results
    np.savez('results_4dof_curriculum.npz', 
             score_history=score_history, 
             avg_score_history=avg_score_history,
             success_history=success_history,
             distance_to_goal_history=distance_to_goal_history,
             actor_loss_history=actor_loss_history,
             critic_loss_history=critic_loss_history,
             curriculum_stages=curriculum_stages)

    # Create enhanced visualizations
    print("\nGenerating curriculum visualization plots...")

    # Success rate with moving average and curriculum stages
    window = 20
    if len(success_history) > window:
        success_rate_avg = np.convolve(success_history, np.ones(window)/window, mode='valid')
    else:
        success_rate_avg = []

    # Plot 1: Training Progress with Curriculum Stages
    plt.figure(figsize=(15,8))
    
    plt.subplot(2,2,1)
    plt.plot(score_history, label='Score per Episode', alpha=0.6)
    plt.plot(avg_score_history, label='Average Score (last 100)', linewidth=2)
    
    # Add vertical lines for curriculum stage changes
    for stage_info in curriculum.stage_success_rates:
        episode_idx = len([x for x in curriculum_stages[:len(score_history)] if x < stage_info['stage']])
        if episode_idx < len(score_history):
            plt.axvline(x=episode_idx, color='red', linestyle='--', alpha=0.7, 
                       label=f"Stage {stage_info['stage']+1}" if stage_info['stage'] < 2 else "")
    
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('4-DOF Robot Curriculum Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2,2,2)
    plt.plot(success_history, label='Success (1=success, 0=fail)', alpha=0.3)
    if len(success_rate_avg) > 0:
        plt.plot(np.arange(window-1, len(success_history)), success_rate_avg, 
                label=f'Success Rate (avg {window})', color='red', linewidth=2)
    
    # Add curriculum stage changes
    for stage_info in curriculum.stage_success_rates:
        episode_idx = len([x for x in curriculum_stages[:len(success_history)] if x < stage_info['stage']])
        if episode_idx < len(success_history):
            plt.axvline(x=episode_idx, color='red', linestyle='--', alpha=0.7)
    
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Success Rate with Curriculum Stages')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2,2,3)
    plt.plot(distance_to_goal_history, label='Distance to Goal per Episode')
    
    # Show varying thresholds based on curriculum
    thresholds = []
    for stage_idx in curriculum_stages[:len(distance_to_goal_history)]:
        if stage_idx < len(curriculum.stages):
            thresholds.append(curriculum.stages[stage_idx]["threshold"])
        else:
            thresholds.append(curriculum.stages[-1]["threshold"])
    
    plt.plot(thresholds, color='red', linestyle='--', label='Success Threshold (varying)')
    plt.xlabel('Episode')
    plt.ylabel('Distance to Goal (m)')
    plt.title('Distance to Goal with Varying Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2,2,4)
    if actor_loss_history and critic_loss_history:
        plt.plot([x for x in actor_loss_history if x is not None], label='Actor Loss')
        plt.plot([x for x in critic_loss_history if x is not None], label='Critic Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('curriculum_training_4dof.png', dpi=300)
    plt.close()

    print("✅ Curriculum visualization plot saved:")
    print("  - curriculum_training_4dof.png")

    # Save trained model
    agent.save_models()
    print("✅ Trained model saved to ckp/ddpg/")

    print(f"\n🎉 4-DOF Robot Curriculum Training Complete!")
    print(f"Final Performance ({total_episodes} episodes):")
    print(f"  Success Rate: {final_success_rate*100:.1f}%")
    print(f"  Average Score: {avg_score:.2f}")
    print(f"  Average Distance (last 20): {np.mean(distance_to_goal_history[-min(20, len(distance_to_goal_history)):]):.4f}m")
    if total_episodes >= stability_window:
        recent_success_rate = np.mean(success_history[-stability_window:]) * 100
        print(f"  Recent Success Rate (last {stability_window}): {recent_success_rate:.1f}%")
