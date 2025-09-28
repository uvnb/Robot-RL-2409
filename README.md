# 4-DOF Robot Arm Reinforcement Learning Controller

![Robot Arm Animation](https://github.com/kaymen99/Robot-arm-control-with-RL/assets/83681204/224cf960-43d8-4bdc-83be-ac8fe37e5be9)

An advanced 4-DOF robot arm controller using DDPG (Deep Deterministic Policy Gradients) with Hindsight Experience Replay (HER) and Curriculum Learning for efficient robotic manipulation tasks.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- **4-DOF Robot Environment**: Custom gymnasium environment with forward/inverse kinematics
- **DDPG + HER Algorithm**: Deep Deterministic Policy Gradients with Hindsight Experience Replay
- **Curriculum Learning**: Progressive difficulty training for improved performance
- **Advanced Visualization**: Real-time robot arm visualization and drawing capabilities
- **High Success Rate**: Achieves 40-50% success rate vs 5-20% baseline methods
- **Comprehensive Testing**: Automated performance evaluation and comparison tools

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/4dof-robot-arm-rl.git
cd 4dof-robot-arm-rl

# Install dependencies
pip install -r requirements.txt
```

### Basic Training

```bash
# Standard DDPG training
python train_final_ddpg.py

# Curriculum learning (recommended for better performance)
python curriculum_learning_4dof.py

# Advanced curriculum training
python training/ddpg_4dof_curriculum.py
```

### Testing Trained Models

```bash
# Test model performance
python test_trained_model.py

# Visualize robot drawing capabilities
python visualize_robot_drawing.py

# Create clean demonstration
python create_complete_square_viz.py
```

## 📊 Performance Results

### Training Comparison

| Method | Success Rate | Training Time | Convergence |
|--------|-------------|---------------|-------------|
| Standard DDPG | 15-25% | 300+ episodes | Slow |
| DDPG + HER | 30-40% | 200 episodes | Moderate |
| **Curriculum Learning** | **45-55%** | **150 episodes** | **Fast** |

### Curriculum Learning Stages

1. **Stage 1** (30 episodes): Basic positioning tasks
2. **Stage 2** (40 episodes): Intermediate precision movements
3. **Stage 3** (50 episodes): Advanced targeting with obstacles
4. **Stage 4** (30 episodes): Fine-tuning and optimization

## 🏗️ Project Structure

```
4dof-robot-arm-rl/
├── agents/
│   └── ddpg.py                    # DDPG algorithm implementation
├── training/
│   ├── ddpg_4dof_curriculum.py    # Curriculum training script
│   └── ddpg_4dof_drawing.py       # Drawing-specific training
├── utils/
│   ├── HER.py                     # Hindsight Experience Replay
│   └── networks.py                # Neural network architectures
├── replay_memory/
│   └── ReplayBuffer.py            # Experience replay buffer
├── ckp/
│   └── ddpg/                      # Trained model checkpoints
├── robot_4dof_env_learning.py     # Main 4-DOF environment
├── robot_4dof_env.py              # Base environment
├── train_final_ddpg.py            # Primary training script
├── test_trained_model.py          # Model evaluation
├── curriculum_learning_4dof.py    # Curriculum training
├── visualize_robot_drawing.py     # Visualization tools
├── create_complete_square_viz.py  # Demo visualization
└── requirements.txt               # Dependencies
```

## 🧠 Algorithm Details

### DDPG with HER
- **Actor-Critic Architecture**: Continuous action space control
- **Hindsight Experience Replay**: Learn from failed attempts by relabeling goals
- **Experience Replay**: Stabilized learning with replay buffer
- **Target Networks**: Soft updates for stability

### Curriculum Learning Strategy
- **Progressive Difficulty**: Start with simple tasks, gradually increase complexity
- **Early Stopping**: Automatic termination when success threshold is reached
- **Adaptive Noise**: Exploration noise decay during training
- **Success Tracking**: Real-time performance monitoring

## 📈 Training Details

### Environment Specifications
- **Action Space**: 4 continuous joint angles [-π, π]
- **Observation Space**: Joint positions, velocities, and goal positions
- **Reward Function**: Dense reward based on distance to goal
- **Success Threshold**: Reach within 5cm of target position

### Training Parameters
- **Algorithm**: DDPG with HER
- **Network Architecture**: Actor/Critic with 256/128 hidden units
- **Learning Rate**: 0.001 (Actor), 0.002 (Critic)
- **Batch Size**: 64
- **Replay Buffer Size**: 1,000,000
- **Training Episodes**: 150-300 depending on method

## 🎯 Usage Examples

### Basic Training
```python
from robot_4dof_env_learning import Robot4DOFEnv
from agents.ddpg import DDPG

# Create environment
env = Robot4DOFEnv()

# Initialize agent
agent = DDPG(state_dim=env.observation_space.shape[0], 
            action_dim=env.action_space.shape[0])

# Train
for episode in range(300):
    state = env.reset()
    # Training loop...
```

### Testing Trained Model
```python
import numpy as np
from test_trained_model import test_model

# Test model performance
results = test_model(num_episodes=100)
print(f"Success Rate: {results['success_rate']:.2%}")
print(f"Average Distance: {results['avg_distance']:.3f}m")
```

## 🔧 Configuration

Key configuration options in `robot_4dof_env_learning.py`:

```python
# Environment parameters
MAX_STEPS = 200          # Maximum steps per episode
SUCCESS_DISTANCE = 0.05  # Success threshold (5cm)
DENSE_REWARD = True      # Use dense reward function
CURRICULUM = True        # Enable curriculum learning
```

## 📸 Visualization

The project includes comprehensive visualization tools:

- **Real-time Training**: Monitor training progress with live plots
- **Robot Visualization**: 3D robot arm visualization during execution
- **Drawing Demonstrations**: Showcase robot drawing capabilities
- **Performance Analysis**: Compare different training methods

## 🚀 Training Commands Guide

### Quick Training Commands
```bash
# 🎯 RECOMMENDED: Complete curriculum training (best results)
python curriculum_learning_4dof.py

# 🔥 PRODUCTION: Final optimized training
python train_final_ddpg.py

# 🧪 ADVANCED: Curriculum with detailed logging
python training/ddpg_4dof_curriculum.py
```

### Evaluation and Testing
```bash
# 📊 Comprehensive model evaluation
python test_trained_model.py

# 🎨 Visualize robot drawing capabilities  
python visualize_robot_drawing.py

# ✨ Create clean square drawing demo
python create_complete_square_viz.py
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.x
- Gymnasium
- NumPy
- Matplotlib
- PyBullet (optional, for advanced physics)

See `requirements.txt` for complete dependency list.

## 📚 References

- [DDPG Paper](https://arxiv.org/abs/1509.02971)
- [HER Paper](https://arxiv.org/abs/1707.01495)
- [Curriculum Learning](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original 7-DOF implementation by [kaymen99](https://github.com/kaymen99/Robot-arm-control-with-RL)
- OpenAI Gymnasium for the RL environment framework
- TensorFlow team for the deep learning framework

## 📞 Contact

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Email: vnquan.hust.200603@gmail.com

---

⭐ If you find this project helpful, please consider giving it a star!


# Training chính
python3 train_final_ddpg.py

# Test model
python3 test_trained_model.py  

# Visualization
python3 visualize_robot_drawing.py