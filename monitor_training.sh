#!/bin/bash
# filepath: /home/quan/Robot-arm-control-with-RL/monitor_training.sh
# Monitor training progress

clear
echo "🔍 MONITORING 4-DOF ROBOT TRAINING"
echo "=================================="
echo ""

# Check if training is running
if pgrep -f "ddpg_4dof_training.py" > /dev/null; then
    echo "✅ Training is RUNNING (PID: $(pgrep -f ddpg_4dof_training.py))"
else
    echo "❌ Training is NOT running"
    echo ""
    echo "To start training:"
    echo "   nohup python3 training/ddpg_4dof_training.py > training_output.log 2>&1 &"
    exit 1
fi

echo ""

# Show recent training progress
if [ -f "training_output.log" ]; then
    echo "📊 LATEST TRAINING OUTPUT:"
    echo "========================="
    tail -20 training_output.log | grep -E "(Ep [0-9]+:|📊|🎯|🏆|PROGRESS|SUCCESS)"
    echo ""
    echo "========================="
    echo ""
    
    # Count episodes completed
    episodes_completed=$(grep -c "^Ep [0-9]*:" training_output.log 2>/dev/null || echo "0")
    successes=$(grep -c "✅" training_output.log 2>/dev/null || echo "0")
    
    if [ "$episodes_completed" -gt 0 ]; then
        success_rate=$(echo "scale=1; $successes * 100 / $episodes_completed" | bc -l 2>/dev/null || echo "0.0")
        echo "📈 CURRENT STATS:"
        echo "   Episodes completed: $episodes_completed"
        echo "   Successes: $successes"
        echo "   Current success rate: ${success_rate}%"
        echo ""
    fi
else
    echo "❌ No training output log found"
fi

# Show generated files
echo "📁 GENERATED FILES:"
echo "=================="
ls -la *.png *.npz 2>/dev/null | grep -E "(4dof|training)" | head -5
echo ""

# Show system resources
echo "💻 SYSTEM RESOURCES:"
echo "==================="
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory: $(free -h | awk 'NR==2{printf "%.1f%% (%s/%s)", $3*100/$2, $3, $2}')"
echo ""

echo "🔄 MONITORING OPTIONS:"
echo "====================="
echo "  monitor_training.sh     - Run this script again"
echo "  tail -f training_output.log  - Live output"
echo "  pkill -f ddpg_4dof      - Stop training"
echo "  python3 plot_results.py - View current results"
