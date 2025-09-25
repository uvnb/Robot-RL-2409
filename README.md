# Robot arm control with Reinforcement Learning

![anim](https://github.com/kaymen99/Robot-arm-control-with-RL/assets/83681204/224cf960-43d8-4bdc-83be-ac8fe37e5be9)

This project focuses on controlling robot arms using continuous reinforcement learning algorithms: DDPG (Deep Deterministic Policy Gradients) and TD3 (Twin Delayed Deep Deterministic Policy Gradients). The project supports both 7-DOF (PandaReach-v3) and custom 4-DOF robot configurations with Hindsight Experience Replay (HER) for enhanced learning.

## 🚀 New Features

### 🎯 4-DOF Robot Support
- **Custom 4-DOF Environment**: Complete implementation with forward/inverse kinematics
- **Hardware Ready**: Real robot interface adapter for deployment
- **Optimized Training**: Achieved 50-55% success rates with proper parameter tuning
- **Spatial Analysis**: Comprehensive workspace visualization and performance analysis

### 🎓 Advanced Training Methods
- **Standard DDPG + HER**: Enhanced with early stopping and noise decay regularization
- **Curriculum Learning**: Progressive difficulty training for improved convergence
- **Performance Comparison**: Side-by-side analysis of different training approaches

### 📊 Comprehensive Analysis Tools
- **Model Testing**: Detailed evaluation with spatial distribution analysis
- **Training Visualization**: Multi-plot analysis including success rates, distances, and losses
- **Method Comparison**: Automated comparison between training approaches

## Continuous RL Algorithms

<p align="justify">
Continuous reinforcement learning deals with environments where actions are continuous, such as the precise control of robotic arm joints or controlling the throttle of an autonomous vehicle. The primary objective is to find policies that effectively map observed states to continuous actions, ultimately optimizing the accumulation of expected rewards. Several algorithms have been specifically developed to address this challenge, including DDPG, TD3, SAC, PPO, and more.
</p>

### 1- DDPG (Deep Deterministic Policy Gradients)

<p align="justify">
DDPG is an actor-critic algorithm designed for continuous action spaces. It combines the strengths of policy gradients and Q-learning. In DDPG, an actor network learns the policy, while a critic network approximates the action-value (Q-function). The actor network directly outputs continuous actions, which are evaluted by the critic network to find the best action thus allowing for fine-grained control.
</p>

### 2- TD3 (Twin Delayed Deep Deterministic Policy Gradients)

<p align="justify">
TD3 is an enhancement of DDPG that addresses issues such as overestimation bias. It introduces the concept of "twin" critics to estimate the Q-value (it uses two critic networks instead of a single one like in DDPG), and it uses target networks with delayed updates to stabilize training. TD3 is known for its robustness and improved performance over DDPG.
</p>

## Hindsight Experience Replay

<p align="justify">
Hindsight Experience Replay (HER) is a technique developed to address the challenge of sparse and binary rewards in RL environments. For example, in many robotic tasks, achieving the desired goal is rare, and traditional RL algorithms struggle to learn from such feedback (agent always gets a zero reward unless the robot successfully completed the task which makes it difficult for the algorithm to learn as it doesn't know if the steps done were good or not).
</p>

<p align="justify">
HER tackles this issue by reusing past experiences for learning, even if they didn't lead to the desired goal. It works by relabeling and storing experiences in a replay buffer, allowing the agent to learn from both successful and failed attempts which significantly accelerates the learning process.
</p>

Link to HER paper: https://arxiv.org/pdf/1707.01495.pdf

## How to run

### 🔥 Quick Start - 4-DOF Robot (Recommended)

```bash
# Train 4-DOF robot with standard DDPG + HER
python3 training/ddpg_4dof_training.py

# Train with curriculum learning (advanced)
python3 training/ddpg_4dof_curriculum.py

# Test trained model performance
python3 test_trained_model.py

# Compare training methods
python3 compare_training_methods.py
```

### 📈 Training Results Visualization

After training, the system automatically generates comprehensive visualizations:
- **Training Progress**: Score evolution and loss curves
- **Success Rate**: Success rate over episodes with moving averages
- **Distance Analysis**: Distance to goal with success threshold visualization

### 🏭 Original 7-DOF Support

```bash
# Original DDPG with HER training
python3 training/ddpg_her.py

# TD3 with HER training
python3 training/td3_her_training.py

# Visualize results
python3 plot_results.py
```

### 🔧 Customization Options

- **Hyperparameters**: Modify in `/agents/` folder (learning rates, discount factors, etc.)
- **Network Architecture**: Configure in `/utils/networks.py`
- **Environment Settings**: Adjust in robot environment files
- **Training Parameters**: Episodes, thresholds, and optimization steps

## 📊 Results

### 4-DOF Robot Performance

Our optimized 4-DOF implementation achieved significant improvements:
- **Success Rate**: 50-55% (vs 0% baseline)
- **Average Distance**: Reduced from 25cm to 15cm
- **Training Efficiency**: Better results in 100 episodes vs 500 previously
- **Convergence**: Stable learning with proper actor/critic loss behavior

### Training Method Comparison

| Method | Success Rate | Convergence Speed | Best Use Case |
|--------|-------------|------------------|---------------|
| Standard DDPG+HER | 50-55% | Moderate | General purpose, simple setup |
| Curriculum Learning | Variable | Faster | Complex tasks, fine precision |

### Original 7-DOF Results

The training of both agents was done in the colab environment:

<div align="center">
<table>
<tr>
<td><img src="https://github.com/kaymen99/Robot-arm-control-with-RL/assets/83681204/957ff11a-e785-4349-9135-960001aa9990" /></td>
<td><img src="https://github.com/kaymen99/Robot-arm-control-with-RL/assets/83681204/1b824c15-02ba-47b1-8260-f913ff282c14" /></td>
</tr>
<br />
<tr>
<td><img src="https://github.com/kaymen99/Robot-arm-control-with-RL/assets/83681204/f89cd3b8-0ce4-4a1f-ad60-f8c629885345" /></td>
<td><img src="https://github.com/kaymen99/Robot-arm-control-with-RL/assets/83681204/e344edf9-c955-4a18-82e2-76cc3df399da" /></td>
</table>
</div>

## 🏗️ Project Structure

```
├── training/
│   ├── ddpg_4dof_training.py      # 4-DOF standard training
│   ├── ddpg_4dof_curriculum.py    # 4-DOF curriculum training
│   ├── ddpg_her.py                # Original 7-DOF DDPG
│   └── td3_her_training.py        # Original 7-DOF TD3
├── agents/
│   ├── ddpg.py                    # DDPG implementation
│   └── td3.py                     # TD3 implementation
├── utils/
│   ├── HER.py                     # Hindsight Experience Replay
│   └── networks.py                # Neural network architectures
├── robot_4dof_env.py              # Custom 4-DOF environment
├── robot_4dof_adapter.py          # Hardware interface
├── test_trained_model.py          # Model testing & evaluation
├── compare_training_methods.py    # Training comparison tool
└── plot_results.py                # Visualization utilities
```

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install gymnasium numpy matplotlib tensorflow panda-gym
   ```

2. **Train 4-DOF Robot**:
   ```bash
   python3 training/ddpg_4dof_training.py
   ```

3. **Test Performance**:
   ```bash
   python3 test_trained_model.py
   ```

4. **Deploy to Real Robot** (when ready):
   ```python
   from robot_4dof_adapter import Robot4DOFAdapter
   # Configure for your specific hardware
   ```

## 💡 Key Improvements Implemented

### Enhanced Training Features
- **Early Stopping**: Automatic termination when success threshold (40%) is reached
- **Noise Decay**: Regularization with decay factor 0.995 starting after episode 60
- **Success Threshold Optimization**: Adjusted from 5cm to 10cm for realistic precision
- **Enhanced Logging**: Detailed progress with emoji indicators and distance metrics

### Analysis & Visualization
- **Comprehensive Testing**: 100-episode evaluation with spatial analysis
- **Workspace Visualization**: Robot reachability and performance mapping
- **Method Comparison**: Automated side-by-side performance analysis
- **Training Monitoring**: Real-time progress tracking and statistics

<!-- Contact -->
## Contact

If you have any questions, feedback, or issues, please don't hesitate to open an issue or reach out to me: aymenMir1001@gmail.com.

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Notes

- cd /home/quan/Robot-arm-control-with-RL && git add . (Thêm tất cả thay đổi vào staging)
- cd /home/quan/Robot-arm-control-with-RL && git commit -m "Update training parameters and add visualization features" (Commit với message mô tả)
- cd /home/quan/Robot-arm-control-with-RL && git push (Đẩy code lên GitHub)

## backup
- cd /home/quan && cp -r Robot-arm-control-with-RL Robot-arm-control-with-RL-backup-$(date +%Y%m%d_%H%M%S)

## tranfer 7 dof to 4dof

📋 Tóm tắt quá trình tạo training cho 4DOF:
robot_4dof_env.py - Môi trường simulation cho robot 4DOF
ddpg_4dof_training.py - Script training chính
test_4dof_env.py - Test môi trường trước khi training
monitor_4dof_training.py - Monitor tiến trình training

## lệnh training 
🚀 LỆNH CHẠY TRAINING MODEL
1️⃣ CHẠY VERSION OPTIMIZED (RECOMMENDED)
cd /home/quan/Robot-arm-control-with-RL
python3 training/ddpg_4dof_optimized.py
2️⃣ CHẠY VERSION CỞ BẢN
python training/ddpg_4dof_training.py
3️⃣ QUICK TEST (nếu muốn test nhanh)
python training/ddpg_4dof_quick_test.py

📁 Dựa trên structure của bạn:
🎯 KHUYẾN NGHỊ:
Chạy version OPTIMIZED vì nó có:
✅ Parameters tối ưu hơn (200 episodes, batch_size=128)
✅ Enhanced logging với màu sắc và emoji
✅ Better early stopping logic
✅ Advanced statistics tracking