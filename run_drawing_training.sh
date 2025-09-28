#!/bin/bash

# 🎨 4-DOF Robot Drawing Training Script
# Enhanced DDPG+HER training with stability improvements

echo "🎨 === 4-DOF ROBOT DRAWING TRAINING ==="
echo "🔧 Enhanced DDPG+HER with optimized parameters"

# Set environment variables for better GPU performance
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0

# Create backup before training
echo "📁 Creating backup before training..."
cd /home/quan && cp -r "Robotarm-RL-backup-2.baseon2 (1)" "Robotarm-RL-backup-drawing-$(date +%Y%m%d_%H%M%S)"

# Change to project directory
cd "/home/quan/Robotarm-RL-backup-2.baseon2 (1)"

echo "📍 Current directory: $(pwd)"
echo "🧪 Testing environment first..."

# Test environment
echo "🔍 Running environment tests..."
python3 test_drawing_env.py

if [ $? -eq 0 ]; then
    echo "✅ Environment tests passed!"
else
    echo "❌ Environment tests failed. Please check the setup."
    exit 1
fi

echo ""
echo "🚀 Starting optimized DDPG+HER training..."
echo "Parameters:"
echo "  - Max episodes: 300"
echo "  - Enhanced rewards for drawing tasks"
echo "  - Domain randomization enabled"
echo "  - Improved exploration strategy"
echo "  - Early stopping at 60% success rate"
echo ""

# Start training
python3 training/ddpg_4dof_drawing.py

# Check training results
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo "📊 Results saved to:"
    echo "  - results_4dof_drawing.npz"
    echo "  - training_results_4dof_drawing.png"
    echo "  - Model checkpoints in ckp/ddpg/"
    
    # Show final results
    echo ""
    echo "📈 Final Training Summary:"
    ls -la results_4dof_drawing.npz training_results_4dof_drawing.png 2>/dev/null
    
else
    echo "❌ Training failed. Check error messages above."
    exit 1
fi

echo ""
echo "🏁 Training script completed!"
echo "Next steps:"
echo "  1. Check training plots for performance analysis"
echo "  2. Test trained model with: python3 test_trained_model.py"
echo "  3. Deploy to real robot when ready"
