"""
Test and evaluate trained 4-DOF Robot model
- Load trained DDPG model
- Test performance on multiple targets
- Generate performance statistics and visualizations
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robot_4dof_env import Robot4DOFReachEnv
from agents.ddpg import DDPGAgent


def test_model_performance(agent, env, num_tests=50, render=False, verbose=False):
    """Test trained model performance"""
    
    successes = 0
    distances = []
    episode_lengths = []
    scores = []
    test_results = []
    
    print(f"🧪 Testing model performance with {num_tests} episodes...")
    
    for test_episode in range(num_tests):
        observation, info = env.reset()
        episode_length = 0
        total_reward = 0
        
        # Extract target position for tracking
        target_pos = observation['desired_goal']
        
        while True:
            # Get state
            curr_obs = observation['observation']
            curr_achgoal = observation['achieved_goal']
            curr_desgoal = observation['desired_goal']
            state = np.concatenate((curr_obs, curr_achgoal, curr_desgoal))
            
            # Choose action (deterministic for testing)
            action = agent.choose_action(state, add_noise=False)
            
            # Execute action
            observation, reward, done, truncated, step_info = env.step(action)
            
            total_reward += reward
            episode_length += 1
            
            if done or truncated:
                break
        
        # Record results
        final_distance = step_info['distance_to_target']
        is_success = step_info['is_success']
        final_position = observation['achieved_goal']
        
        if is_success:
            successes += 1
        
        distances.append(final_distance)
        episode_lengths.append(episode_length)
        scores.append(total_reward)
        
        test_results.append({
            'episode': test_episode,
            'success': is_success,
            'distance': final_distance,
            'length': episode_length,
            'score': total_reward,
            'target': target_pos.copy(),
            'final_pos': final_position.copy()
        })
        
        if verbose or (test_episode + 1) % 10 == 0:
            status = "✅" if is_success else "❌"
            print(f"Test {test_episode+1:2d}: {status} Distance={final_distance:.4f}m, "
                  f"Steps={episode_length}, Score={total_reward:.1f}")
    
    # Calculate statistics
    success_rate = successes / num_tests * 100
    avg_distance = np.mean(distances)
    std_distance = np.std(distances)
    avg_episode_length = np.mean(episode_lengths)
    avg_score = np.mean(scores)
    
    results = {
        'success_rate': success_rate,
        'successes': successes,
        'total_tests': num_tests,
        'avg_distance': avg_distance,
        'std_distance': std_distance,
        'min_distance': np.min(distances),
        'max_distance': np.max(distances),
        'avg_episode_length': avg_episode_length,
        'avg_score': avg_score,
        'distances': distances,
        'episode_lengths': episode_lengths,
        'scores': scores,
        'test_results': test_results
    }
    
    return results


def visualize_test_results(results, save_prefix='test_results'):
    """Create visualizations of test results"""
    
    print("📊 Creating test result visualizations...")
    
    # Create comprehensive test results plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Success Rate Pie Chart
    ax1 = axes[0, 0]
    successes = results['successes']
    failures = results['total_tests'] - successes
    sizes = [successes, failures]
    labels = ['Successes', 'Failures']
    colors = ['lightgreen', 'lightcoral']
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 10})
    ax1.set_title(f'Test Results\n({results["total_tests"]} episodes)')
    
    # Plot 2: Distance Distribution
    ax2 = axes[0, 1]
    distances = results['distances']
    ax2.hist(distances, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(results['avg_distance'], color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {results["avg_distance"]:.4f}m')
    ax2.axvline(0.1, color='green', linestyle='--', linewidth=2, label='Success Threshold: 0.1m')
    ax2.set_xlabel('Final Distance to Target (m)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distance Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Episode Length Distribution
    ax3 = axes[0, 2]
    episode_lengths = results['episode_lengths']
    ax3.hist(episode_lengths, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
    ax3.axvline(results['avg_episode_length'], color='red', linestyle='--', linewidth=2,
                label=f'Mean: {results["avg_episode_length"]:.1f} steps')
    ax3.set_xlabel('Episode Length (steps)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Episode Length Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distance vs Episode Length
    ax4 = axes[1, 0]
    colors = ['green' if result['success'] else 'red' for result in results['test_results']]
    ax4.scatter(episode_lengths, distances, c=colors, alpha=0.6)
    ax4.set_xlabel('Episode Length (steps)')
    ax4.set_ylabel('Final Distance (m)')
    ax4.set_title('Distance vs Episode Length')
    ax4.axhline(0.1, color='gray', linestyle='--', alpha=0.5, label='Success Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Performance over Test Episodes
    ax5 = axes[1, 1]
    episodes = range(1, len(distances) + 1)
    success_markers = ['o' if result['success'] else 'x' for result in results['test_results']]
    
    for i, (ep, dist, marker) in enumerate(zip(episodes, distances, success_markers)):
        color = 'green' if marker == 'o' else 'red'
        ax5.scatter(ep, dist, marker=marker, c=color, s=50, alpha=0.7)
    
    ax5.axhline(0.1, color='gray', linestyle='--', alpha=0.5, label='Success Threshold')
    ax5.set_xlabel('Test Episode')
    ax5.set_ylabel('Final Distance (m)')
    ax5.set_title('Performance Consistency')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Spatial Distribution of Targets and Final Positions
    ax6 = axes[1, 2]
    
    # Plot targets and final positions
    for result in results['test_results']:
        target = result['target']
        final_pos = result['final_pos']
        
        # Only plot X-Y coordinates (assuming first 2 dimensions)
        if result['success']:
            ax6.scatter(target[0], target[1], marker='*', c='green', s=100, alpha=0.7, label='Success Target' if result['episode'] == 0 else "")
            ax6.scatter(final_pos[0], final_pos[1], marker='o', c='lightgreen', s=50, alpha=0.7, label='Success Final' if result['episode'] == 0 else "")
        else:
            ax6.scatter(target[0], target[1], marker='*', c='red', s=100, alpha=0.7, label='Fail Target' if result['episode'] == 0 else "")
            ax6.scatter(final_pos[0], final_pos[1], marker='o', c='lightcoral', s=50, alpha=0.7, label='Fail Final' if result['episode'] == 0 else "")
        
        # Draw line connecting target to final position
        ax6.plot([target[0], final_pos[0]], [target[1], final_pos[1]], 
                'gray', alpha=0.3, linewidth=0.5)
    
    ax6.set_xlabel('X Position')
    ax6.set_ylabel('Y Position')
    ax6.set_title('Spatial Distribution (X-Y Plane)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a detailed workspace visualization
    plt.figure(figsize=(12, 10))
    
    # Plot reachable workspace (approximate circle)
    workspace_radius = 0.85  # Sum of link lengths: 0.25+0.25+0.2+0.15 = 0.85
    circle = Circle((0, 0), workspace_radius, fill=False, linestyle='--', 
                   color='gray', alpha=0.5, label='Max Workspace')
    plt.gca().add_patch(circle)
    
    # Plot test targets and results
    successful_targets = []
    failed_targets = []
    successful_finals = []
    failed_finals = []
    
    for result in results['test_results']:
        target = result['target']
        final_pos = result['final_pos']
        
        if result['success']:
            successful_targets.append(target[:2])
            successful_finals.append(final_pos[:2])
        else:
            failed_targets.append(target[:2])
            failed_finals.append(final_pos[:2])
    
    if successful_targets:
        successful_targets = np.array(successful_targets)
        successful_finals = np.array(successful_finals)
        plt.scatter(successful_targets[:, 0], successful_targets[:, 1], 
                   marker='*', c='green', s=150, alpha=0.8, label=f'Successful Targets ({len(successful_targets)})')
        plt.scatter(successful_finals[:, 0], successful_finals[:, 1], 
                   marker='o', c='lightgreen', s=80, alpha=0.6, label='Successful Finals')
    
    if failed_targets:
        failed_targets = np.array(failed_targets)
        failed_finals = np.array(failed_finals)
        plt.scatter(failed_targets[:, 0], failed_targets[:, 1], 
                   marker='*', c='red', s=150, alpha=0.8, label=f'Failed Targets ({len(failed_targets)})')
        plt.scatter(failed_finals[:, 0], failed_finals[:, 1], 
                   marker='o', c='lightcoral', s=80, alpha=0.6, label='Failed Finals')
    
    # Draw lines connecting targets to finals for failed attempts
    for result in results['test_results']:
        if not result['success']:
            target = result['target']
            final_pos = result['final_pos']
            plt.plot([target[0], final_pos[0]], [target[1], final_pos[1]], 
                    'red', alpha=0.2, linewidth=1)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(f'4-DOF Robot Test Results - Workspace Analysis\nSuccess Rate: {results["success_rate"]:.1f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig(f'{save_prefix}_workspace.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Test visualizations saved:")
    print(f"  - {save_prefix}_comprehensive.png")
    print(f"  - {save_prefix}_workspace.png")


def print_detailed_statistics(results):
    """Print detailed test statistics"""
    
    print("\n" + "="*60)
    print("📊 DETAILED TEST RESULTS")
    print("="*60)
    
    print(f"\n🎯 SUCCESS METRICS:")
    print(f"  Success Rate: {results['success_rate']:.1f}%")
    print(f"  Successes: {results['successes']}/{results['total_tests']}")
    
    print(f"\n📏 DISTANCE METRICS:")
    print(f"  Average Distance: {results['avg_distance']:.4f}m")
    print(f"  Standard Deviation: {results['std_distance']:.4f}m")
    print(f"  Min Distance: {results['min_distance']:.4f}m")
    print(f"  Max Distance: {results['max_distance']:.4f}m")
    print(f"  Success Threshold: 0.1000m")
    
    print(f"\n⏱️ EPISODE METRICS:")
    print(f"  Average Episode Length: {results['avg_episode_length']:.1f} steps")
    print(f"  Average Score: {results['avg_score']:.2f}")
    
    # Performance analysis
    distances = np.array(results['distances'])
    within_threshold = np.sum(distances <= 0.1)
    close_misses = np.sum((distances > 0.1) & (distances <= 0.15))  # 10-15cm
    far_misses = np.sum(distances > 0.15)
    
    print(f"\n🎯 ACCURACY BREAKDOWN:")
    print(f"  Within Threshold (≤10cm): {within_threshold} ({within_threshold/len(distances)*100:.1f}%)")
    print(f"  Close Misses (10-15cm): {close_misses} ({close_misses/len(distances)*100:.1f}%)")
    print(f"  Far Misses (>15cm): {far_misses} ({far_misses/len(distances)*100:.1f}%)")
    
    # Quartile analysis
    q25, q50, q75 = np.percentile(distances, [25, 50, 75])
    print(f"\n📊 DISTANCE QUARTILES:")
    print(f"  25th Percentile: {q25:.4f}m")
    print(f"  50th Percentile (Median): {q50:.4f}m")
    print(f"  75th Percentile: {q75:.4f}m")
    
    # Identify problematic areas
    failed_tests = [r for r in results['test_results'] if not r['success']]
    if failed_tests:
        failed_distances = [r['distance'] for r in failed_tests]
        avg_failed_distance = np.mean(failed_distances)
        
        print(f"\n❌ FAILURE ANALYSIS:")
        print(f"  Failed Tests: {len(failed_tests)}")
        print(f"  Average Failed Distance: {avg_failed_distance:.4f}m")
        print(f"  Improvement Needed: {avg_failed_distance - 0.1:.4f}m")
    
    print(f"\n💡 PERFORMANCE ASSESSMENT:")
    if results['success_rate'] >= 80:
        print("  🏆 EXCELLENT - Ready for real robot deployment")
    elif results['success_rate'] >= 60:
        print("  ✅ GOOD - Consider additional fine-tuning")
    elif results['success_rate'] >= 40:
        print("  ⚠️  MODERATE - More training recommended")
    else:
        print("  ❌ POOR - Significant improvement needed")


if __name__ == "__main__":
    
    print("🤖 4-DOF Robot Model Testing")
    print("="*40)
    
    # Create environment
    env = Robot4DOFReachEnv(render_mode=None)
    
    # Initialize agent
    agent = DDPGAgent(lr_actor=0.001, lr_critic=0.001, 
                      input_dims=env.observation_space.shape[0] + 6,
                      n_actions=env.action_space.shape[0], mem_size=1000)
    
    # Load trained model
    try:
        agent.load_models()
        print("✅ Trained model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load trained model: {e}")
        print("Please train the model first using:")
        print("  python training/ddpg_4dof_training.py")
        exit(1)
    
    print(f"\nEnvironment Configuration:")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    print(f"  Success threshold: {env.success_threshold}m")
    print(f"  Max workspace radius: ~{sum(env.link_lengths)}m")
    
    # Test model performance
    print(f"\n{'='*60}")
    
    # Run comprehensive testing
    results = test_model_performance(agent, env, num_tests=100, verbose=False)
    
    # Print detailed statistics
    print_detailed_statistics(results)
    
    # Create visualizations
    visualize_test_results(results)
    
    # Save test results
    np.savez('test_results_4dof.npz', **results)
    print(f"\n✅ Test results saved to test_results_4dof.npz")
    
    print(f"\n🎉 Model testing complete!")
    
    # Recommendations
    print(f"\n💭 RECOMMENDATIONS:")
    if results['success_rate'] >= 70:
        print("  🚀 Model is ready for real hardware testing")
        print("  📋 Use robot_4dof_adapter.py for hardware interface")
    else:
        print("  🔧 Consider additional training or parameter tuning")
        print("  🎓 Try curriculum learning: python training/ddpg_4dof_curriculum.py")
    
    if results['avg_distance'] > 0.08:
        print("  🎯 Consider reducing success threshold for fine-tuning")
    
    print(f"\n📝 Next Steps:")
    print("  1. Analyze failure patterns in test_results_workspace.png")
    print("  2. Review distance distribution for improvement areas")
    print("  3. Test with varying target difficulties")
    print("  4. Deploy to real hardware when ready")
