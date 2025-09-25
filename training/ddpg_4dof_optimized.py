"""
OPTIMIZED Training script for 4-DOF Robot with DDPG + HER
- More episodes for better convergence
- Improved hyperparameters
- Better success tracking
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

    print("🚀 OPTIMIZED 4-DOF ROBOT TRAINING WITH DDPG + HER")
    print("=" * 60)
    
    # OPTIMIZED Training parameters
    n_games = 200  # More episodes for better learning
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
    threshold = 0.08  # Stricter 8cm threshold
    noise_decay_rate = 0.997  # Slower decay for better exploration
    
    # Early stopping parameters
    early_stop_threshold = 0.6  # Higher success rate target
    stability_window = 30  # Larger window for stability

    print(f"🎯 TRAINING PARAMETERS:")
    print(f"- Episodes: {n_games}")
    print(f"- Optimization steps: {opt_steps}")
    print(f"- Success threshold: {threshold}m")
    print(f"- Early stopping: {early_stop_threshold*100}% success rate")

    # Create optimized 4-DOF robot environment
    env = Robot4DOFReachEnv(
        base_height=0.1,
        link_lengths=[0.25, 0.25, 0.2, 0.15],  # 4-DOF link lengths
        max_episode_steps=50,
        success_threshold=threshold
    )
    
    # Calculate observation space size for DDPG agent
    obs_shape = env.observation_space['observation'].shape[0] + \
                env.observation_space['achieved_goal'].shape[0] + \
                env.observation_space['desired_goal'].shape[0]
    
    # Initialize DDPG agent with better parameters
    agent = DDPGAgent(
        env=env,
        input_dims=obs_shape,
        lr_actor=0.0005,  # Slightly lower learning rate for stability
        lr_critic=0.001, 
        mem_size=100000,  # Larger memory buffer
        batch_size=128,   # Larger batch size
        tau=0.001         # Slower target network updates
    )

    print(f"\n🤖 ENVIRONMENT INFO:")
    print(f"- Action space: {env.action_space.shape}")
    print(f"- Max reach: {env.max_reach:.3f}m")
    print(f"- Workspace radius: {env.workspace_radius:.3f}m")
    print(f"- Total state space: {obs_shape}")

    print(f"\n🧠 AGENT INFO:")
    print(f"- Actor LR: 0.0005")
    print(f"- Critic LR: 0.001") 
    print(f"- Memory size: 100,000")
    print(f"- Batch size: 128")

    print(f"\n{'='*60}")
    print("🏁 STARTING TRAINING...")
    print(f"{'='*60}")

    # Training loop
    for i in range(n_games):
        
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

        # HER augmentation
        her_augmentation(agent, obs_array, actions_array, new_obs_array)

        # Apply noise decay (start later for better exploration)
        if i > 80:  # Start decay after 80 episodes
            agent.noise_factor = max(agent.noise_factor * noise_decay_rate, 0.02)  # Lower min noise

        # Train agent with more steps
        for _ in range(opt_steps):
            actor_loss, critic_loss = agent.learn()
            if actor_loss is not None:
                actor_loss_episode = actor_loss
            if critic_loss is not None:
                critic_loss_episode = critic_loss

        # Enhanced logging with performance indicators
        status = "✅" if is_success else "❌"
        distance_color = "🟢" if final_distance <= threshold else "🟡" if final_distance <= 0.15 else "🔴"
        
        if actor_loss_episode is not None and critic_loss_episode is not None:
            print(f"Ep {i:3d}: {status} Steps={step:2d}, Score={score:5.1f}, Dist={final_distance:.4f}m {distance_color}")
            print(f"       Actor={actor_loss_episode:7.4f}, Critic={critic_loss_episode:7.4f}, Noise={agent.noise_factor:.3f}")
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
            print(f"\n{'='*60}")
            print(f"📊 PROGRESS UPDATE (Episode {i+1})")
            print(f"{'='*60}")
            print(f"🎯 Success Rate: {current_success_rate:.1f}%")
            print(f"📈 Average Score: {avg_score:.2f}")
            print(f"🏆 Best Score: {best_score:.2f}")
            print(f"📏 Avg Distance (last 25): {np.mean(distance_to_goal_history[-25:]):.4f}m")
            if i >= stability_window:
                recent_success_rate = np.mean(success_history[-stability_window:]) * 100
                print(f"🔥 Recent Success Rate (last {stability_window}): {recent_success_rate:.1f}%")
            print(f"🔊 Current Noise Factor: {agent.noise_factor:.3f}")
            print(f"{'='*60}")

    # Final statistics
    final_success_rate = num_success / (i + 1)
    total_episodes = i + 1
    print(f"\n{'='*60}")
    print("🏁 TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"📊 Episodes Completed: {total_episodes}/{n_games}")
    print(f"🎯 Final Success Rate: {final_success_rate*100:.2f}%")
    print(f"✅ Total Successes: {num_success}/{total_episodes}")
    print(f"🏆 Best Average Score: {best_score:.2f}")
    
    if total_episodes < n_games:
        print(f"⚡ Training stopped early due to reaching success threshold!")
    else:
        print(f"✔️  Training completed all planned episodes.")

    # Advanced statistics
    if len(distance_to_goal_history) > 10:
        recent_distances = distance_to_goal_history[-50:]
        print(f"\n📏 DISTANCE ANALYSIS:")
        print(f"   Average distance (last 50): {np.mean(recent_distances):.4f}m")
        print(f"   Best distance achieved: {np.min(distance_to_goal_history):.4f}m")
        precision_count = sum(1 for d in recent_distances if d <= threshold)
        print(f"   Precision hits (last 50): {precision_count}/50 ({precision_count*2}%)")

    # Save results
    np.savez('results_4dof_optimized.npz', 
             score_history=score_history, 
             avg_score_history=avg_score_history,
             success_history=success_history,
             distance_to_goal_history=distance_to_goal_history,
             actor_loss_history=actor_loss_history,
             critic_loss_history=critic_loss_history)

    # Create enhanced visualizations
    print(f"\n📊 Generating visualization plots...")

    # Success rate with moving average
    window = 30
    if len(success_history) > window:
        success_rate_avg = np.convolve(success_history, np.ones(window)/window, mode='valid')
    else:
        success_rate_avg = []

    # Plot 1: Training Progress & Loss
    plt.figure(figsize=(15,10))
    
    # Score subplot
    plt.subplot(2,2,1)
    plt.plot(score_history, label='Score per Episode', alpha=0.4, linewidth=1)
    plt.plot(avg_score_history, label='Average Score (last 100)', linewidth=2, color='red')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Score Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss subplot
    plt.subplot(2,2,2)
    valid_actor_losses = [x for x in actor_loss_history if x is not None]
    valid_critic_losses = [x for x in critic_loss_history if x is not None]
    if valid_actor_losses:
        plt.plot(valid_actor_losses, label='Actor Loss', alpha=0.7)
    if valid_critic_losses:
        plt.plot(valid_critic_losses, label='Critic Loss', alpha=0.7)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Success rate subplot
    plt.subplot(2,2,3)
    plt.plot(success_history, label='Success (1=success, 0=fail)', alpha=0.3)
    if len(success_rate_avg) > 0:
        plt.plot(np.arange(window-1, len(success_history)), success_rate_avg, 
                label=f'Success Rate (avg {window})', color='green', linewidth=2)
    plt.axhline(y=early_stop_threshold, color='red', linestyle='--', 
               label=f'Early Stop Threshold ({early_stop_threshold*100:.0f}%)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Success Rate Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Distance subplot
    plt.subplot(2,2,4)
    plt.plot(distance_to_goal_history, label='Distance to Goal', alpha=0.6)
    plt.axhline(y=threshold, color='red', linestyle='--', 
               label=f'Success Threshold = {threshold}m')
    
    # Add moving average for distance
    if len(distance_to_goal_history) > 20:
        dist_avg = np.convolve(distance_to_goal_history, np.ones(20)/20, mode='valid')
        plt.plot(np.arange(19, len(distance_to_goal_history)), dist_avg, 
                color='orange', linewidth=2, label='Distance (avg 20)')
    
    plt.xlabel('Episode')
    plt.ylabel('Distance to Goal (m)')
    plt.title('Distance to Goal Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_progress_4dof_optimized.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Enhanced visualization plot saved:")
    print("   📊 training_progress_4dof_optimized.png")

    # Save trained model
    agent.save_models()
    print("✅ Trained model saved to ckp/ddpg/")

    print(f"\n🎉 OPTIMIZED 4-DOF ROBOT TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"🏆 FINAL PERFORMANCE ({total_episodes} episodes):")
    print(f"   🎯 Success Rate: {final_success_rate*100:.1f}%")
    print(f"   📈 Average Score: {avg_score:.2f}")
    if len(distance_to_goal_history) >= 20:
        recent_dist = np.mean(distance_to_goal_history[-20:])
        print(f"   📏 Average Distance (last 20): {recent_dist:.4f}m")
        print(f"   🎯 Improvement from threshold: {((threshold-recent_dist)/threshold*100):+.1f}%")
    
    if total_episodes >= stability_window:
        recent_success_rate = np.mean(success_history[-stability_window:]) * 100
        print(f"   🔥 Recent Success Rate (last {stability_window}): {recent_success_rate:.1f}%")

    print(f"\n💡 NEXT STEPS:")
    if final_success_rate >= 0.7:
        print("   🚀 Excellent performance! Ready for real hardware deployment")
        print("   🔧 Consider testing with robot_4dof_adapter.py")
    elif final_success_rate >= 0.5:
        print("   ✅ Good performance! Consider fine-tuning or more episodes")
        print("   🎓 Try curriculum learning for even better results")
    else:
        print("   ⚠️  Consider adjusting hyperparameters or training longer")
        print("   🎯 Try reducing success threshold to 0.12m for easier targets")

    print(f"{'='*60}")
