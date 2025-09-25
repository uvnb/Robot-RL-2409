#!/bin/bash
# filepath: /home/quan/Robot-arm-control-with-RL/train_now.sh
# INSTANT HIGH-PERFORMANCE TRAINING

clear
echo "🚀 IMPROVED HIGH-PERFORMANCE 4-DOF ROBOT TRAINING"
echo "================================================"
echo ""

# Backup previous results
if [ -f "results_4dof.npz" ]; then
    echo "📦 Backing up previous results..."
    mv results_4dof.npz results_4dof_backup_$(date +%Y%m%d_%H%M%S).npz
fi

echo "🎯 Starting IMPROVED training with:"
echo "   - 150 episodes (increased from 100)"
echo "   - 8cm precision threshold (stricter)"  
echo "   - Enhanced hyperparameters"
echo "   - 60% success rate target"
echo ""

# Start improved training
python3 training/ddpg_4dof_training.py

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 TRAINING SUCCESSFUL!"
    echo "======================"
    
    # Show generated files
    echo ""
    echo "📊 Generated files:"
    ls -la *optimized* 2>/dev/null | head -5
    
    echo ""
    echo "🧪 To test performance:"
    echo "   python3 test_trained_model.py"
    
    echo ""
    echo "📈 View results:"
    echo "   training_progress_4dof_optimized.png"
    
else
    echo "❌ Training failed - check errors above"
    exit 1
fi
