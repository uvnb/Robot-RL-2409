"""
Training script for 4-DOF Robot with DDPG + HER
Based on existing ddpg_her.py but adapted for 4-DOF robot
"""

import numpy as np
import gymnasium as gym
import sys
import os

# Add parent directory to path to import custom environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_4dof_env import Robot4DOFReachEnv
from agents.ddpg import DDPGAgent
from utils.HER import her_augmentation
import matplotlib.pyplot as plt


if __name__ == "__main__":

    print("🚀 IMPROVED 4-DOF ROBOT TRAINING WITH DDPG + HER 🚀")
    
    # IMPROVED Training parameters
    n_games = 150  # More episodes for better learning
    opt_steps = 80  # More optimization steps
    best_score = 0
    
    # Tracking variables
    score_history = []
    avg_score_history = []
    actor_loss_history = []
    critic_loss_history = []
    success_history = []
    distance_to_goal_history = []

    # Success tracking
    num_success = 0
    threshold = 0.08  # Stricter 8cm threshold for better precision
    
    # Early stopping parameters
    early_stop_threshold = 0.6  # Higher success rate target (60%)
    stability_window = 25  # Larger stability window

    print(f"🎯 IMPROVED TRAINING PARAMETERS:")
    print(f"- Episodes: {n_games}")
    print(f"- Optimization steps per episode: {opt_steps}")
    print(f"- Success threshold: {threshold}m (stricter precision)")
    print(f"- Early stopping: {early_stop_threshold*100}% success rate")
    print(f"- Stability window: {stability_window} episodes")

    # Create 4-DOF robot environment
    env = Robot4DOFReachEnv(
        base_height=0.1,
        link_lengths=[0.25, 0.25, 0.2, 0.15],  # Adjust to your robot
        max_episode_steps=50,
        success_threshold=threshold
    )
    
    print(f"\nEnvironment created:")
    print(f"- Action space: {env.action_space.shape}")
    print(f"- Max reach: {env.max_reach:.3f}m")
    print(f"- Workspace radius: {env.workspace_radius:.3f}m")

    # Calculate observation space size for DDPG agent
    obs_shape = env.observation_space['observation'].shape[0] + \
                env.observation_space['achieved_goal'].shape[0] + \
                env.observation_space['desired_goal'].shape[0]
    
    print(f"- Total state space: {obs_shape}")

    # Initialize DDPG agent
    agent = DDPGAgent(env=env, input_dims=obs_shape)
    print(f"DDPG Agent initialized with {obs_shape}D input")

    # Training loop
    actor_loss_episode = None
    critic_loss_episode = None
    
    # Regularization parameters
    initial_noise_factor = 0.165  # From DDPG agent default
    noise_decay_rate = 0.997  # Slower decay for better exploration
    
    print(f"\n{'='*60}")
    print("🏁 STARTING IMPROVED TRAINING...")
    print(f"{'='*60}")
    
    for i in range(n_games):
        done = False
        truncated = False
        score = 0
        step = 0

        # Arrays for HER
        obs_array = []
        actions_array = []
        new_obs_array = []

        # Reset environment
        observation, info = env.reset()

        while not (done or truncated):
            # Get current state
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

        # Check success
        final_distance = step_info['distance_to_target']
        is_success = step_info['is_success']
        
        if is_success:
            num_success += 1

        success_history.append(int(is_success))
        distance_to_goal_history.append(final_distance)

        # HER augmentation
        her_augmentation(agent, obs_array, actions_array, new_obs_array)

        # Apply noise decay to prevent overfitting
        if i > 80:  # Start decay after 80 episodes (later start)
            agent.noise_factor = max(agent.noise_factor * noise_decay_rate, 0.03)  # Lower min noise

        # Train agent
        for _ in range(opt_steps):
            actor_loss, critic_loss = agent.learn()
            if actor_loss is not None:
                actor_loss_episode = actor_loss
            if critic_loss is not None:
                critic_loss_episode = critic_loss

        # Enhanced logging with performance indicators
        status = "✅" if is_success else "❌"
        distance_color = "🟢" if final_distance <= threshold else "🟡" if final_distance <= 0.12 else "🔴"
        
        if actor_loss_episode is not None and critic_loss_episode is not None:
            print(f"Ep {i:3d}: {status} Steps={step:2d}, Score={score:5.1f}, Dist={final_distance:.4f}m {distance_color}")
            print(f"       Actor={actor_loss_episode:7.4f}, Critic={critic_loss_episode:7.4f}")
        else:
            print(f"Ep {i:3d}: {status} Steps={step:2d}, Score={score:5.1f}, Dist={final_distance:.4f}m {distance_color}")
            
        # Track performance
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)
        actor_loss_history.append(actor_loss_episode)
        critic_loss_history.append(critic_loss_episode)

        if avg_score > best_score:
            best_score = avg_score

        # Check for early stopping
        if i >= stability_window:  # Only check after enough episodes
            recent_success_rate = np.mean(success_history[-stability_window:])
            if recent_success_rate >= early_stop_threshold:
                print(f"\n🎯 EARLY STOPPING TRIGGERED!")
                print(f"Success rate ({recent_success_rate*100:.1f}%) exceeded threshold ({early_stop_threshold*100:.1f}%)")
                print(f"over the last {stability_window} episodes.")
                print(f"Training stopped at episode {i+1}")
                break

        # Progress update every 30 episodes
        if (i + 1) % 30 == 0:
            current_success_rate = (num_success / (i + 1)) * 100
            print(f"\n{'='*60}")
            print(f"📊 PROGRESS UPDATE (Episode {i+1})")
            print(f"{'='*60}")
            print(f"🎯 Success Rate: {current_success_rate:.1f}%")
            print(f"📈 Average Score: {avg_score:.2f}")
            print(f"🏆 Best Score: {best_score:.2f}")
            print(f"📏 Avg Distance (last 30): {np.mean(distance_to_goal_history[-30:]):.4f}m")
            if i >= stability_window:
                recent_success_rate = np.mean(success_history[-stability_window:]) * 100
                print(f"🔥 Recent Success Rate (last {stability_window}): {recent_success_rate:.1f}%")
            print(f"🔊 Noise Factor: {agent.noise_factor:.3f}")
            print(f"{'='*60}")

    # Final statistics
    final_success_rate = num_success / (i + 1)  # Use actual episodes completed
    total_episodes = i + 1
    print(f"\n{'='*60}")
    print("🏁 IMPROVED TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"📊 Episodes Completed: {total_episodes}/{n_games}")
    print(f"🎯 Final Success Rate: {final_success_rate*100:.2f}%")
    print(f"✅ Total Successes: {num_success}/{total_episodes}")
    print(f"🏆 Best Average Score: {best_score:.2f}")
    
    if total_episodes < n_games:
        print(f"⚡ Training stopped early due to reaching success threshold!")
    else:
        print(f"✔️  Training completed all planned episodes.")

    # Enhanced statistics
    if len(distance_to_goal_history) > 10:
        recent_distances = distance_to_goal_history[-30:]
        print(f"\n📏 PRECISION ANALYSIS:")
        print(f"   Average distance (last 30): {np.mean(recent_distances):.4f}m")
        print(f"   Best distance achieved: {np.min(distance_to_goal_history):.4f}m")
        precision_hits = sum(1 for d in recent_distances if d <= threshold)
        print(f"   Precision hits (last 30): {precision_hits}/30 ({precision_hits*100/30:.1f}%)")

    # Save results
    np.savez('results_4dof.npz', 
             score_history=score_history, 
             avg_score_history=avg_score_history,
             success_history=success_history,
             distance_to_goal_history=distance_to_goal_history,
             actor_loss_history=actor_loss_history,
             critic_loss_history=critic_loss_history)

    # Create visualizations
    print("\nGenerating visualization plots...")

    # Success rate with moving average
    window = 20
    if len(success_history) > window:
        success_rate_avg = np.convolve(success_history, np.ones(window)/window, mode='valid')
    else:
        success_rate_avg = []

    # Plot 1: Training Progress & Loss
    plt.figure(figsize=(12,6))
    plt.plot(score_history, label='Score per Episode', alpha=0.6)
    plt.plot(avg_score_history, label='Average Score (last 100)', linewidth=2)
    plt.plot(actor_loss_history, label='Actor Loss per Episode')
    plt.plot(critic_loss_history, label='Critic Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('4-DOF Robot Training Progress & Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_progress_4dof.png', dpi=300)
    plt.close()

    # Plot 2: Success Rate
    plt.figure(figsize=(12,6))
    plt.plot(success_history, label='Success (1=success, 0=fail)', alpha=0.3)
    if len(success_rate_avg) > 0:
        plt.plot(np.arange(window-1, len(success_history)), success_rate_avg, 
                label=f'Success Rate (avg {window})', color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Success')
    plt.title('4-DOF Robot Success Rate per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('success_rate_4dof.png', dpi=300)
    plt.close()

    # Plot 3: Distance to Goal
    plt.figure(figsize=(12,6))
    plt.plot(distance_to_goal_history, label='Distance to Goal per Episode')
    plt.axhline(y=threshold, color='red', linestyle='--', 
               label=f'Success Threshold = {threshold}m')
    plt.xlabel('Episode')
    plt.ylabel('Distance to Goal (m)')
    plt.title('4-DOF Robot Distance to Goal per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('distance_to_goal_4dof.png', dpi=300)
    plt.close()

    print("✅ Visualization plots saved:")
    print("  - training_progress_4dof.png")
    print("  - success_rate_4dof.png") 
    print("  - distance_to_goal_4dof.png")

    # Save trained model
    agent.save_models()
    print("✅ Trained model saved to ckp/ddpg/")

    print(f"\n🎉 IMPROVED 4-DOF Robot Training Complete!")
    print(f"🏆 FINAL PERFORMANCE ({total_episodes} episodes):")
    print(f"  🎯 Success Rate: {final_success_rate*100:.1f}%")
    print(f"  📈 Average Score: {avg_score:.2f}")
    print(f"  📏 Average Distance (last 20): {np.mean(distance_to_goal_history[-min(20, len(distance_to_goal_history)):]):.4f}m")
    if total_episodes >= stability_window:
        recent_success_rate = np.mean(success_history[-stability_window:]) * 100
        print(f"  🔥 Recent Success Rate (last {stability_window}): {recent_success_rate:.1f}%")

    print(f"\n💡 PERFORMANCE ASSESSMENT:")
    if final_success_rate >= 0.7:
        print("  🏆 EXCELLENT - Ready for real robot deployment!")
        print("  🔧 Consider testing with robot_4dof_adapter.py")
    elif final_success_rate >= 0.5:
        print("  ✅ GOOD - Performance is solid, consider fine-tuning")
        print("  🎓 Try curriculum learning for even better results")
    else:
        print("  ⚠️  MODERATE - Consider training longer or adjusting parameters")
        print("  🎯 Try increasing threshold to 0.1m for easier targets")

# Cần implement IK solver chính xác hơn cho robot 4-DOF cụ thể của bạn

