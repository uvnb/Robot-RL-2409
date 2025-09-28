#!/usr/bin/env python3
"""
Advanced Curriculum Learning for 4-DOF Robot Drawing
Progressive difficulty with adaptive thresholds and domain adaptation
"""

import numpy as np
import torch
import importlib.util
from agents.ddpg import DDPGAgent
import matplotlib.pyplot as plt
import os
from datetime import datetime

class CurriculumTrainer:
    """Advanced curriculum learning trainer for robot drawing"""
    
    def __init__(self):
        self.load_environment()
        self.setup_curriculum()
        
    def load_environment(self):
        """Load learning environment module"""
        spec = importlib.util.spec_from_file_location("learning_env", "robot_4dof_env_learning.py")
        self.env_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.env_module)
        
    def setup_curriculum(self):
        """Setup curriculum learning stages"""
        self.curriculum_stages = [
            {
                'name': 'Foundation',
                'threshold': 0.15,      # 15cm - easier than before
                'points': 4,            # Ultra simple (4 corner square)
                'steps': 20,
                'episodes': 25,
                'success_target': 90,   # Need 90% success to advance
                'description': 'Learn basic movement and positioning'
            },
            {
                'name': 'Basic_Control', 
                'threshold': 0.12,      # 12cm
                'points': 6,            # Slightly more complex
                'steps': 25,
                'episodes': 30,
                'success_target': 85,
                'description': 'Improve precision and trajectory following'
            },
            {
                'name': 'Trajectory_Mastery',
                'threshold': 0.08,      # 8cm
                'points': 8,            # Standard square
                'steps': 30, 
                'episodes': 40,
                'success_target': 80,
                'description': 'Master standard 8-point trajectory'
            },
            {
                'name': 'Precision_Training',
                'threshold': 0.05,      # 5cm
                'points': 12,           # More waypoints
                'steps': 40,
                'episodes': 50,
                'success_target': 75,
                'description': 'High precision with complex paths'
            },
            {
                'name': 'Expert_Level',
                'threshold': 0.03,      # 3cm - very precise
                'points': 16,           # Complex trajectory
                'steps': 50,
                'episodes': 60,
                'success_target': 70,
                'description': 'Expert-level precision drawing'
            },
            {
                'name': 'Master_Challenge',
                'threshold': 0.02,      # 2cm - original target
                'points': 20,           # Maximum complexity
                'steps': 60,
                'episodes': 80,
                'success_target': 60,   # More forgiving at highest level
                'description': 'Ultimate drawing precision challenge'
            }
        ]
        
    def create_stage_environment(self, stage):
        """Create environment for specific curriculum stage"""
        env = self.env_module.Robot4DOFDrawingEnv(
            max_episode_steps=stage['steps'],
            success_threshold=stage['threshold'],
            enable_domain_randomization=False  # Start without randomization
        )
        
        # Generate appropriate trajectory complexity
        if stage['points'] == 4:
            # Ultra-simple 4-corner square
            center = np.array([0.3, 0.4, 0.35])
            size = 0.1  # Smaller for easier learning
            points = [
                center + np.array([-size/2, 0, -size/2]),  # Bottom-left
                center + np.array([size/2, 0, -size/2]),   # Bottom-right  
                center + np.array([size/2, 0, size/2]),    # Top-right
                center + np.array([-size/2, 0, size/2])    # Top-left
            ]
            env.trajectory_points = points
            
        elif stage['points'] <= 8:
            env._generate_simple_trajectory()
        else:
            # More complex trajectory
            env._generate_square_trajectory(size=0.15, num_points_per_side=stage['points']//4)
            
        return env
        
    def train_stage(self, agent, stage, stage_idx):
        """Train agent on specific curriculum stage"""
        print(f"\n🎯 STAGE {stage_idx+1}: {stage['name']}")
        print(f"   Threshold: {stage['threshold']*100:.0f}cm, Points: {stage['points']}, Episodes: {stage['episodes']}")
        print(f"   Goal: {stage['description']}")
        print(f"   Success Target: {stage['success_target']}%\n")
        
        env = self.create_stage_environment(stage)
        
        # Stage training metrics
        episode_rewards = []
        success_rates = []
        progress_rates = []
        
        # Adaptive parameters based on stage
        warmup = min(5, stage['episodes'] // 5)  # 20% warmup
        batch_size = 128
        
        consecutive_success_episodes = 0
        best_success_rate = 0
        
        for episode in range(stage['episodes']):
            obs, info = env.reset()
            episode_reward = 0
            episode_success = False
            max_progress = 0
            
            for step in range(env.max_episode_steps):
                obs_array = obs['observation'] if isinstance(obs, dict) else obs
                
                # Action selection with stage-appropriate exploration
                if episode < warmup:
                    action = env.action_space.sample()  # Random exploration
                else:
                    action = agent.choose_action(obs_array)
                    # Add adaptive noise (more at early stages)
                    noise_factor = max(0.05, 0.3 * (1 - episode/stage['episodes']))
                    action += np.random.normal(0, noise_factor, action.shape)
                    action = np.clip(action, -1, 1)
                    
                next_obs, reward, done, truncated, next_info = env.step(action)
                next_obs_array = next_obs['observation'] if isinstance(next_obs, dict) else next_obs
                
                # Store experience
                agent.remember(obs_array, action, reward, next_obs_array, done or truncated)
                
                episode_reward += reward
                max_progress = max(max_progress, next_info['trajectory_progress'])
                
                if next_info['is_success']:
                    episode_success = True
                    
                # Train after warmup
                if episode >= warmup and agent.memory.counter > batch_size:
                    agent.learn()
                    
                obs = next_obs
                
                if done or truncated:
                    break
                    
            # Track metrics
            episode_rewards.append(episode_reward)
            success_rates.append(1.0 if episode_success else 0.0)
            progress_rates.append(max_progress)
            
            # Calculate rolling success rate
            window = min(10, len(success_rates))
            recent_success = np.mean(success_rates[-window:]) * 100
            recent_progress = np.mean(progress_rates[-window:]) * 100
            recent_reward = np.mean(episode_rewards[-window:])
            
            # Track consecutive successes
            if episode_success:
                consecutive_success_episodes += 1
            else:
                consecutive_success_episodes = 0
                
            # Progress reporting
            if (episode + 1) % 10 == 0 or episode < 5:
                print(f"   Episode {episode+1:2d}: Success={recent_success:5.1f}%, "
                      f"Progress={recent_progress:5.1f}%, Reward={recent_reward:6.1f}")
                      
            # Early advancement check
            if (episode >= 15 and  # Minimum episodes
                recent_success >= stage['success_target'] and
                consecutive_success_episodes >= 5):  # Stable performance
                print(f"   🎉 STAGE MASTERED! Advanced early at episode {episode+1}")
                break
                
        # Stage completion analysis
        final_success_rate = np.mean(success_rates[-10:]) * 100 if len(success_rates) >= 10 else np.mean(success_rates) * 100
        final_progress_rate = np.mean(progress_rates[-10:]) * 100 if len(progress_rates) >= 10 else np.mean(progress_rates) * 100
        
        stage_passed = final_success_rate >= stage['success_target'] * 0.8  # 80% of target
        
        print(f"   📊 STAGE RESULTS:")
        print(f"      Final Success Rate: {final_success_rate:.1f}% (Target: {stage['success_target']}%)")
        print(f"      Final Progress Rate: {final_progress_rate:.1f}%")
        print(f"      Status: {'✅ PASSED' if stage_passed else '❌ NEEDS MORE TRAINING'}")
        
        return {
            'passed': stage_passed,
            'success_rate': final_success_rate,
            'progress_rate': final_progress_rate,
            'episode_rewards': episode_rewards,
            'success_rates': success_rates,
            'progress_rates': progress_rates
        }
        
    def run_curriculum(self):
        """Run complete curriculum training"""
        print("🚀 ADVANCED CURRICULUM LEARNING - 4-DOF Robot Drawing")
        print(f"   {len(self.curriculum_stages)} progressive stages")
        print(f"   Adaptive difficulty with success-based advancement\n")
        
        # Initialize agent with first stage
        initial_env = self.create_stage_environment(self.curriculum_stages[0])
        agent = DDPGAgent(
            env=initial_env,
            input_dims=initial_env.observation_space['observation'].shape[0],
            alpha=1e-3,     # Conservative learning rate for curriculum
            beta=1e-3,
            tau=0.005,
            gamma=0.99,
            batch_size=128,
            noise_factor=0.15  # Moderate exploration
        )
        
        # Curriculum training results
        curriculum_results = []
        stages_passed = 0
        
        for stage_idx, stage in enumerate(self.curriculum_stages):
            stage_result = self.train_stage(agent, stage, stage_idx)
            curriculum_results.append(stage_result)
            
            if stage_result['passed']:
                stages_passed += 1
                
                # Save checkpoint after each successful stage
                if not os.path.exists('ckp'):
                    os.makedirs('ckp')
                agent.save_models()
                print(f"   💾 Checkpoint saved after {stage['name']}")
                
            else:
                print(f"   ⚠️  Stage {stage['name']} not fully mastered - continuing anyway")
                
            # Optional: Stop if multiple consecutive failures
            if (stage_idx >= 2 and 
                not curriculum_results[-1]['passed'] and
                not curriculum_results[-2]['passed']):
                print(f"\n⏹️  Stopping curriculum - consecutive stage failures")
                break
                
        # Final curriculum assessment
        print(f"\n📈 CURRICULUM COMPLETION SUMMARY:")
        print(f"   Stages Completed: {len(curriculum_results)}/{len(self.curriculum_stages)}")
        print(f"   Stages Passed: {stages_passed}/{len(curriculum_results)}")
        print(f"   Success Rate: {stages_passed/len(curriculum_results)*100:.1f}%")
        
        # Plot curriculum results
        self.plot_curriculum_results(curriculum_results)
        
        # Final capability test
        print(f"\n🎯 FINAL CAPABILITY TEST...")
        self.test_final_capability(agent)
        
        return agent, curriculum_results

    def plot_curriculum_results(self, results):
        """Plot curriculum learning progress"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        stages = [f"Stage {i+1}" for i in range(len(results))]
        success_rates = [r['success_rate'] for r in results]
        progress_rates = [r['progress_rate'] for r in results] 
        
        # Success rates by stage
        colors = ['green' if r['passed'] else 'red' for r in results]
        ax1.bar(stages, success_rates, color=colors, alpha=0.7)
        ax1.set_title('Success Rate by Curriculum Stage')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_ylim(0, 100)
        for i, v in enumerate(success_rates):
            ax1.text(i, v + 2, f'{v:.1f}%', ha='center')
            
        # Progress rates by stage
        ax2.bar(stages, progress_rates, color='blue', alpha=0.7)
        ax2.set_title('Progress Rate by Curriculum Stage') 
        ax2.set_ylabel('Progress Rate (%)')
        ax2.set_ylim(0, 100)
        for i, v in enumerate(progress_rates):
            ax2.text(i, v + 2, f'{v:.1f}%', ha='center')
            
        # Learning curves for last few stages
        for i, result in enumerate(results[-3:]):  # Last 3 stages
            if len(result['success_rates']) > 0:
                episodes = range(1, len(result['success_rates']) + 1)
                rolling_success = [np.mean(result['success_rates'][max(0, j-9):j+1]) 
                                 for j in range(len(result['success_rates']))]
                ax3.plot(episodes, rolling_success, label=f'Stage {len(results)-2+i}')
                
        ax3.set_title('Learning Curves (Last 3 Stages)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Success Rate (10-ep avg)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Curriculum difficulty progression
        thresholds = [s['threshold']*100 for s in self.curriculum_stages[:len(results)]]
        points = [s['points'] for s in self.curriculum_stages[:len(results)]]
        
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(stages, thresholds, 'ro-', label='Success Threshold (cm)')
        line2 = ax4_twin.plot(stages, points, 'bs-', label='Trajectory Points')
        
        ax4.set_ylabel('Success Threshold (cm)', color='red')
        ax4_twin.set_ylabel('Trajectory Points', color='blue')
        ax4.set_title('Curriculum Difficulty Progression')
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'curriculum_learning_results_{timestamp}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"   📈 Curriculum results saved: {filename}")
        plt.show()
        
    def test_final_capability(self, agent):
        """Test agent's final capability on multiple difficulty levels"""
        test_configs = [
            {'name': 'EASY', 'threshold': 0.15, 'points': 8},
            {'name': 'MEDIUM', 'threshold': 0.08, 'points': 12}, 
            {'name': 'HARD', 'threshold': 0.05, 'points': 16},
            {'name': 'EXPERT', 'threshold': 0.03, 'points': 20}
        ]
        
        print(f"   Testing on {len(test_configs)} difficulty levels...")
        
        for config in test_configs:
            env = self.env_module.Robot4DOFDrawingEnv(
                max_episode_steps=50,
                success_threshold=config['threshold'],
                enable_domain_randomization=False
            )
            
            # Test for 10 episodes
            successes = 0
            total_progress = 0
            
            for episode in range(10):
                obs, info = env.reset()
                episode_success = False
                max_progress = 0
                
                for step in range(env.max_episode_steps):
                    obs_array = obs['observation'] if isinstance(obs, dict) else obs
                    action = agent.choose_action(obs_array)  # No exploration noise
                    obs, reward, done, truncated, info = env.step(action)
                    
                    max_progress = max(max_progress, info['trajectory_progress'])
                    if info['is_success']:
                        episode_success = True
                        
                    if done or truncated:
                        break
                        
                if episode_success:
                    successes += 1
                total_progress += max_progress
                
            success_rate = successes / 10 * 100
            avg_progress = total_progress / 10 * 100
            
            print(f"      {config['name']:8s}: {success_rate:5.1f}% success, {avg_progress:5.1f}% progress")

def main():
    """Run curriculum learning training"""
    trainer = CurriculumTrainer()
    agent, results = trainer.run_curriculum()
    return agent, results

if __name__ == "__main__":
    agent, results = main()
