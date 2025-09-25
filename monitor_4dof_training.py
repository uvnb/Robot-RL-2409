#!/usr/bin/env python3
"""
Monitor training progress for 4-DOF robot
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os

def monitor_4dof_training():
    """Monitor the 4DOF training progress"""
    
    print("=== 4-DOF TRAINING MONITOR ===")
    print("Monitoring training progress...")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Check if results file exists
            if os.path.exists('results_4dof.npz'):
                try:
                    # Load current results
                    data = np.load('results_4dof.npz')
                    score_history = data['score_history']
                    success_history = data['success_history']
                    distance_history = data['distance_to_goal_history']
                    
                    current_episode = len(score_history)
                    
                    if current_episode > 0:
                        # Calculate statistics
                        recent_scores = score_history[-10:] if len(score_history) >= 10 else score_history
                        avg_recent_score = np.mean(recent_scores)
                        
                        recent_distances = distance_history[-10:] if len(distance_history) >= 10 else distance_history
                        avg_recent_distance = np.mean(recent_distances)
                        
                        success_count = np.sum(success_history)
                        success_rate = success_count / current_episode * 100
                        
                        # Print progress
                        print(f"Episode {current_episode}:")
                        print(f"  Recent avg score: {avg_recent_score:.2f}")
                        print(f"  Recent avg distance: {avg_recent_distance:.4f}m")
                        print(f"  Success rate: {success_rate:.1f}% ({success_count}/{current_episode})")
                        print(f"  Last score: {score_history[-1]:.2f}")
                        print("-" * 50)
                        
                except Exception as e:
                    print(f"Error reading results: {e}")
            else:
                print("Waiting for training to start...")
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        
        # Generate final plot if data exists
        if os.path.exists('results_4dof.npz'):
            print("Generating current progress plot...")
            
            data = np.load('results_4dof.npz')
            score_history = data['score_history']
            success_history = data['success_history']
            
            if len(score_history) > 0:
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 2, 1)
                plt.plot(score_history, alpha=0.7)
                if len(score_history) > 10:
                    # Moving average
                    window = min(20, len(score_history) // 5)
                    moving_avg = np.convolve(score_history, np.ones(window)/window, mode='valid')
                    plt.plot(range(window-1, len(score_history)), moving_avg, 'r-', linewidth=2)
                plt.title('4DOF Training Progress')
                plt.xlabel('Episode')
                plt.ylabel('Score')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                plt.plot(success_history, alpha=0.5)
                if len(success_history) > 10:
                    window = min(20, len(success_history) // 5)
                    success_rate = np.convolve(success_history, np.ones(window)/window, mode='valid')
                    plt.plot(range(window-1, len(success_history)), success_rate, 'g-', linewidth=2)
                plt.title('4DOF Success Rate')
                plt.xlabel('Episode')
                plt.ylabel('Success Rate')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('current_4dof_progress.png', dpi=150)
                print("Current progress saved as 'current_4dof_progress.png'")

if __name__ == "__main__":
    monitor_4dof_training()
