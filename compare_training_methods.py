"""
Compare different training methods for 4-DOF Robot
- Standard DDPG + HER training
- Curriculum learning training
- Performance analysis and comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def load_results(filename):
    """Load training results from npz file"""
    try:
        results = np.load(filename)
        return {key: results[key] for key in results.files}
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return None


def calculate_metrics(success_history, distance_history):
    """Calculate performance metrics"""
    success_rate = np.mean(success_history) * 100
    avg_distance = np.mean(distance_history)
    convergence_episode = None
    
    # Find convergence point (when success rate stays > 40% for 20+ episodes)
    window = 20
    threshold = 0.4
    
    if len(success_history) >= window:
        for i in range(window, len(success_history)):
            recent_success = np.mean(success_history[i-window:i])
            if recent_success >= threshold:
                convergence_episode = i
                break
    
    return {
        'success_rate': success_rate,
        'avg_distance': avg_distance,
        'convergence_episode': convergence_episode,
        'episodes_trained': len(success_history)
    }


def plot_comparison(standard_results, curriculum_results):
    """Create comparison plots"""
    
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Success Rate Comparison
    plt.subplot(2, 3, 1)
    
    if standard_results:
        success_std = standard_results['success_history']
        window = 20
        if len(success_std) > window:
            success_rate_std = np.convolve(success_std, np.ones(window)/window, mode='valid')
            plt.plot(np.arange(window-1, len(success_std)), success_rate_std, 
                    label='Standard DDPG+HER', linewidth=2, color='blue')
    
    if curriculum_results:
        success_cur = curriculum_results['success_history']
        window = 20
        if len(success_cur) > window:
            success_rate_cur = np.convolve(success_cur, np.ones(window)/window, mode='valid')
            plt.plot(np.arange(window-1, len(success_cur)), success_rate_cur, 
                    label='Curriculum Learning', linewidth=2, color='red')
    
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Success Rate Comparison (20-episode moving average)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Plot 2: Distance to Goal Comparison
    plt.subplot(2, 3, 2)
    
    if standard_results:
        dist_std = standard_results['distance_to_goal_history']
        # Smooth distance with moving average
        window = 10
        if len(dist_std) > window:
            dist_smooth_std = np.convolve(dist_std, np.ones(window)/window, mode='valid')
            plt.plot(np.arange(window-1, len(dist_std)), dist_smooth_std, 
                    label='Standard DDPG+HER', linewidth=2, alpha=0.8, color='blue')
    
    if curriculum_results:
        dist_cur = curriculum_results['distance_to_goal_history']
        window = 10
        if len(dist_cur) > window:
            dist_smooth_cur = np.convolve(dist_cur, np.ones(window)/window, mode='valid')
            plt.plot(np.arange(window-1, len(dist_cur)), dist_smooth_cur, 
                    label='Curriculum Learning', linewidth=2, alpha=0.8, color='red')
    
    plt.xlabel('Episode')
    plt.ylabel('Distance to Goal (m)')
    plt.title('Distance to Goal Comparison (10-episode moving average)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Score Comparison
    plt.subplot(2, 3, 3)
    
    if standard_results:
        score_std = standard_results['avg_score_history']
        plt.plot(score_std, label='Standard DDPG+HER', linewidth=2, color='blue')
    
    if curriculum_results:
        score_cur = curriculum_results['avg_score_history']
        plt.plot(score_cur, label='Curriculum Learning', linewidth=2, color='red')
    
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.title('Average Score Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Learning Efficiency (Success Rate vs Episodes)
    plt.subplot(2, 3, 4)
    
    episodes_to_success = []
    success_thresholds = [0.2, 0.4, 0.6, 0.8]
    
    methods = []
    if standard_results:
        methods.append(('Standard', standard_results['success_history'], 'blue'))
    if curriculum_results:
        methods.append(('Curriculum', curriculum_results['success_history'], 'red'))
    
    window = 20
    for threshold in success_thresholds:
        threshold_episodes = []
        
        for method_name, success_history, color in methods:
            episodes_to_threshold = None
            if len(success_history) >= window:
                for i in range(window, len(success_history)):
                    recent_success = np.mean(success_history[i-window:i])
                    if recent_success >= threshold:
                        episodes_to_threshold = i
                        break
            threshold_episodes.append(episodes_to_threshold if episodes_to_threshold else len(success_history))
        
        if len(threshold_episodes) == 2:
            x_pos = np.arange(len(threshold_episodes))
            colors = ['blue', 'red']
            plt.bar(x_pos + threshold * 0.2 - 0.3, threshold_episodes, width=0.15, 
                   label=f'{threshold*100}% success', alpha=0.7)
    
    plt.xlabel('Method')
    plt.ylabel('Episodes to Reach Threshold')
    plt.title('Learning Efficiency Comparison')
    plt.xticks([0, 1], ['Standard', 'Curriculum'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Final Performance Comparison
    plt.subplot(2, 3, 5)
    
    metrics = {}
    if standard_results:
        metrics['Standard'] = calculate_metrics(
            standard_results['success_history'], 
            standard_results['distance_to_goal_history']
        )
    
    if curriculum_results:
        metrics['Curriculum'] = calculate_metrics(
            curriculum_results['success_history'], 
            curriculum_results['distance_to_goal_history']
        )
    
    method_names = list(metrics.keys())
    success_rates = [metrics[method]['success_rate'] for method in method_names]
    colors = ['blue', 'red'][:len(method_names)]
    
    bars = plt.bar(method_names, success_rates, color=colors, alpha=0.7)
    plt.ylabel('Final Success Rate (%)')
    plt.title('Final Success Rate Comparison')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, success_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom')
    
    # Plot 6: Training Stability (Success Rate Variance)
    plt.subplot(2, 3, 6)
    
    stability_metrics = {}
    window_size = 20
    
    if methods:  # Only plot if we have methods
        for method_name, success_history, color in methods:
            if len(success_history) >= window_size * 2:
                # Calculate rolling variance of success rate
                rolling_variance = []
                for i in range(window_size, len(success_history) - window_size):
                    window_success = success_history[i-window_size:i+window_size]
                    rolling_variance.append(np.var(window_success))
                
                plt.plot(np.arange(window_size, len(success_history) - window_size), 
                        rolling_variance, label=f'{method_name}', linewidth=2, color=color)
                
                stability_metrics[method_name] = np.mean(rolling_variance)
    
    plt.xlabel('Episode')
    plt.ylabel('Success Rate Variance')
    plt.title('Training Stability Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics


def print_comparison_report(metrics):
    """Print detailed comparison report"""
    print("\n" + "="*60)
    print("🔍 TRAINING METHODS COMPARISON REPORT")
    print("="*60)
    
    if not metrics:
        print("No training results found for comparison.")
        return
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print("-" * 40)
    
    for method, metric in metrics.items():
        print(f"\n{method.upper()} TRAINING:")
        print(f"  Episodes Trained: {metric['episodes_trained']}")
        print(f"  Final Success Rate: {metric['success_rate']:.1f}%")
        print(f"  Average Distance: {metric['avg_distance']:.4f}m")
        if metric['convergence_episode']:
            print(f"  Convergence Episode: {metric['convergence_episode']}")
        else:
            print(f"  Convergence Episode: Not reached")
    
    if len(metrics) == 2:
        method_names_comp = list(metrics.keys())
        std_metrics = metrics[method_names_comp[0]]
        cur_metrics = metrics[method_names_comp[1]]
        
        print(f"\n🏆 COMPARISON SUMMARY:")
        print("-" * 40)
        
        # Success rate comparison
        success_diff = cur_metrics['success_rate'] - std_metrics['success_rate']
        if success_diff > 0:
            print(f"✅ Curriculum learning achieved {success_diff:.1f}% higher success rate")
        elif success_diff < 0:
            print(f"✅ Standard training achieved {abs(success_diff):.1f}% higher success rate")
        else:
            print(f"🤝 Both methods achieved similar success rates")
        
        # Distance comparison
        distance_diff = std_metrics['avg_distance'] - cur_metrics['avg_distance']
        if distance_diff > 0:
            print(f"🎯 Curriculum learning achieved {distance_diff:.4f}m better accuracy")
        elif distance_diff < 0:
            print(f"🎯 Standard training achieved {abs(distance_diff):.4f}m better accuracy")
        else:
            print(f"🤝 Both methods achieved similar accuracy")
        
        # Convergence comparison
        if (std_metrics['convergence_episode'] and cur_metrics['convergence_episode']):
            conv_diff = std_metrics['convergence_episode'] - cur_metrics['convergence_episode']
            if conv_diff > 0:
                print(f"⚡ Curriculum learning converged {conv_diff} episodes faster")
            elif conv_diff < 0:
                print(f"⚡ Standard training converged {abs(conv_diff)} episodes faster")
            else:
                print(f"🤝 Both methods converged at similar rates")
        
        # Training efficiency
        std_efficiency = std_metrics['success_rate'] / std_metrics['episodes_trained'] * 100
        cur_efficiency = cur_metrics['success_rate'] / cur_metrics['episodes_trained'] * 100
        
        print(f"\n📈 TRAINING EFFICIENCY:")
        print(f"  Standard: {std_efficiency:.2f}% success per episode")
        print(f"  Curriculum: {cur_efficiency:.2f}% success per episode")
        
        if cur_efficiency > std_efficiency:
            print(f"  🏅 Curriculum learning is more efficient")
        elif std_efficiency > cur_efficiency:
            print(f"  🏅 Standard training is more efficient")
        else:
            print(f"  🤝 Both methods have similar efficiency")


if __name__ == "__main__":
    
    print("🔍 Analyzing training methods for 4-DOF Robot...")
    
    # Load results from both training methods
    standard_results = load_results('results_4dof.npz')
    curriculum_results = load_results('results_4dof_curriculum.npz')
    
    if not standard_results and not curriculum_results:
        print("❌ No training results found. Please run training first.")
        print("\nTo run training:")
        print("  Standard: python training/ddpg_4dof_training.py")
        print("  Curriculum: python training/ddpg_4dof_curriculum.py")
        exit(1)
    
    # Calculate metrics and create comparison plots
    metrics = plot_comparison(standard_results, curriculum_results)
    
    # Print detailed comparison report
    print_comparison_report(metrics)
    
    print(f"\n✅ Comparison analysis complete!")
    print(f"📊 Comparison plot saved: training_methods_comparison.png")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 30)
    
    if len(metrics) == 2:
        method_names_rec = list(metrics.keys())
        std_metrics = metrics[method_names_rec[0]]
        cur_metrics = metrics[method_names_rec[1]]
        
        if cur_metrics['success_rate'] > std_metrics['success_rate'] + 5:
            print("🎓 Use Curriculum Learning for better performance")
        elif std_metrics['success_rate'] > cur_metrics['success_rate'] + 5:
            print("⚡ Use Standard Training for better performance")
        else:
            print("🤔 Both methods perform similarly - choose based on:")
            print("   - Curriculum: Better for complex tasks, gradual learning")
            print("   - Standard: Faster training, simpler implementation")
        
        if (cur_metrics['convergence_episode'] and std_metrics['convergence_episode'] and
            cur_metrics['convergence_episode'] < std_metrics['convergence_episode'] - 20):
            print("⏰ Curriculum Learning converges faster - good for quick prototyping")
        
        print(f"\n🔧 NEXT STEPS:")
        print("  1. Fine-tune hyperparameters for the better performing method")
        print("  2. Test with real hardware using robot_4dof_adapter.py")
        print("  3. Consider hybrid approach combining both methods")
        print("  4. Implement additional safety constraints for real robot")
    else:
        print("Run both training methods for detailed comparison")
