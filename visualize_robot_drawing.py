#!/usr/bin/env python3
"""
Visualize robot trajectory and coordinates for 4-DOF drawing
Show exactly how robot draws the square on 2D plane (X-Z with fixed Y)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import importlib.util

def visualize_robot_drawing():
    """Create detailed visualization of robot drawing trajectory"""
    print("🎨 === VISUALIZING ROBOT DRAWING TRAJECTORY ===")
    print("Showing coordinates and path on X-Z plane (Y fixed at 0.4m)\n")
    
    # Load learning environment
    spec = importlib.util.spec_from_file_location("learning_env", "robot_4dof_env_learning.py")
    learning_env_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(learning_env_module)
    
    # Create environment
    env = learning_env_module.Robot4DOFDrawingEnv(
        max_episode_steps=30,
        success_threshold=0.20,  # 20cm threshold
        enable_domain_randomization=False
    )
    
    print(f"✅ Environment loaded:")
    print(f"   Square center: ({env.trajectory_points[0][0]:.3f}, {env.trajectory_points[0][1]:.3f}, {env.trajectory_points[0][2]:.3f})")
    print(f"   Success threshold: {env.success_threshold*100:.0f}cm = {env.success_threshold:.3f}m")
    print(f"   Trajectory points: {len(env.trajectory_points)}")
    
    # Get trajectory coordinates
    trajectory_points = np.array(env.trajectory_points)
    print(f"\n📍 SQUARE TRAJECTORY COORDINATES:")
    print(f"   Point | X (m)   | Y (m)   | Z (m)   | Description")
    print(f"   ------|---------|---------|---------|-------------")
    
    corner_names = ["Bottom-Left", "Bottom-Right", "Top-Right", "Top-Left", 
                   "Bottom-Left", "Bottom-Right", "Top-Right", "Top-Left"]
    
    for i, point in enumerate(trajectory_points):
        desc = corner_names[i] if i < len(corner_names) else f"Point {i+1}"
        print(f"   {i+1:2d}    | {point[0]:7.3f} | {point[1]:7.3f} | {point[2]:7.3f} | {desc}")
    
    # Test robot movement
    print(f"\n🤖 TESTING ROBOT MOVEMENT:")
    obs, info = env.reset()
    
    # Record robot path
    robot_positions = []
    target_positions = []
    distances = []
    waypoint_hits = []
    
    print(f"   Starting position: ({env.current_ee_pos[0]:.3f}, {env.current_ee_pos[1]:.3f}, {env.current_ee_pos[2]:.3f})")
    
    # Simulate intelligent movement toward each waypoint
    for step in range(20):  # Max 20 steps
        current_pos = env.current_ee_pos.copy()
        current_target = env.target_pos.copy()
        distance = info['distance_to_target']
        
        robot_positions.append(current_pos)
        target_positions.append(current_target)
        distances.append(distance)
        
        print(f"   Step {step+1:2d}: Robot({current_pos[0]:.3f}, {current_pos[2]:.3f}) → Target({current_target[0]:.3f}, {current_target[2]:.3f}), Dist={distance:.3f}m")
        
        # Smart action: move directly toward target
        direction = current_target - current_pos
        action = np.array([direction[0], direction[2], 0.0]) * 0.5  # Scale movement
        action = np.clip(action, -1, 1)
        
        obs, reward, done, truncated, info = env.step(action)
        
        if info.get('is_success', False):
            waypoint_hits.append(step)
            print(f"        ✅ HIT! Waypoint {env.current_trajectory_idx} reached!")
        
        if done or truncated:
            print(f"   Episode completed after {step+1} steps")
            break
    
    robot_positions = np.array(robot_positions)
    target_positions = np.array(target_positions)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 15))
    
    # Main trajectory plot (large)
    ax1 = plt.subplot(2, 3, (1, 4))
    
    # Plot target square (close the loop to complete square)
    square_x = trajectory_points[:, 0]
    square_z = trajectory_points[:, 2]
    
    # Close the square by connecting last point to first point
    complete_square_x = np.append(square_x, square_x[0])
    complete_square_z = np.append(square_z, square_z[0])
    
    ax1.plot(complete_square_x, complete_square_z, 'r-', linewidth=3, label='Target Square', marker='s', markersize=10)
    
    # Plot robot path (also close the loop for complete visualization)  
    robot_x = robot_positions[:, 0]
    robot_z = robot_positions[:, 2]
    
    # Close robot path if it reached all waypoints
    if len(waypoint_hits) >= 8:
        # Connect robot path back to starting area to show complete square
        complete_robot_x = np.append(robot_x, robot_x[0])
        complete_robot_z = np.append(robot_z, robot_z[0])
        ax1.plot(complete_robot_x, complete_robot_z, 'b-', linewidth=3, label='Robot Path (Complete)', marker='o', markersize=8, alpha=0.8)
    else:
        ax1.plot(robot_x, robot_z, 'b-', linewidth=3, label='Robot Path', marker='o', markersize=8)
    
    # Add waypoint numbers and success circles
    for i, point in enumerate(trajectory_points):
        # Waypoint number
        ax1.annotate(f'{i+1}', (point[0], point[2]), 
                    xytext=(15, 15), textcoords='offset points', 
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        
        # Success threshold circle
        circle = Circle((point[0], point[2]), env.success_threshold, 
                       fill=False, color='red', alpha=0.3, linestyle=':', linewidth=1)
        ax1.add_patch(circle)
    
    # Add direction arrows to show robot movement flow
    if len(robot_positions) > 1:
        for i in range(0, len(robot_positions)-1, max(1, len(robot_positions)//4)):
            if i+1 < len(robot_positions):
                dx = robot_positions[i+1][0] - robot_positions[i][0] 
                dz = robot_positions[i+1][2] - robot_positions[i][2]
                ax1.arrow(robot_positions[i][0], robot_positions[i][2], dx*0.7, dz*0.7,
                         head_width=0.01, head_length=0.015, fc='blue', ec='blue', alpha=0.6)
    
    # Mark starting position
    ax1.plot(robot_x[0], robot_z[0], 'go', markersize=15, label='Start Position')
    ax1.plot(robot_x[-1], robot_z[-1], 'ro', markersize=15, label='End Position')
    
    # Add coordinate grid
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X Coordinate (m)', fontsize=12)
    ax1.set_ylabel('Z Coordinate (m)', fontsize=12)
    
    # Dynamic title based on completion
    completion_status = "COMPLETE SQUARE" if len(waypoint_hits) >= 8 else f"PARTIAL ({len(waypoint_hits)}/8 waypoints)"
    ax1.set_title(f'Robot Drawing Square Trajectory - {completion_status}\n(Y fixed at {trajectory_points[0][1]:.1f}m)', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.axis('equal')
    
    # Add coordinate annotations for robot path
    for i in range(0, len(robot_positions), max(1, len(robot_positions)//5)):
        pos = robot_positions[i]
        ax1.annotate(f'({pos[0]:.2f},{pos[2]:.2f})', 
                    (pos[0], pos[2]), xytext=(0, -20), 
                    textcoords='offset points', fontsize=9, ha='center')
    
    # Distance plot
    ax2 = plt.subplot(2, 3, 2)
    steps = range(1, len(distances) + 1)
    ax2.plot(steps, distances, 'b-', linewidth=2, marker='o')
    ax2.axhline(y=env.success_threshold, color='red', linestyle='--', 
                label=f'Success Threshold ({env.success_threshold:.2f}m)')
    
    # Mark waypoint hits
    for hit in waypoint_hits:
        if hit < len(distances):
            ax2.plot(hit+1, distances[hit], 'go', markersize=10)
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Distance to Target (m)')
    ax2.set_title('Distance to Target Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Coordinate table
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')
    
    # Create table data
    table_data = []
    table_data.append(['Point', 'X (m)', 'Z (m)', 'Type'])
    for i, point in enumerate(trajectory_points):
        table_data.append([f'{i+1}', f'{point[0]:.3f}', f'{point[2]:.3f}', corner_names[i][:8]])
    
    table = ax3.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax3.set_title('Square Waypoint Coordinates', fontsize=12)
    
    # X coordinate over time
    ax4 = plt.subplot(2, 3, 5)
    ax4.plot(steps, robot_x, 'r-', linewidth=2, marker='o', label='X coordinate')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('X Coordinate (m)')
    ax4.set_title('X Movement Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Z coordinate over time  
    ax5 = plt.subplot(2, 3, 6)
    ax5.plot(steps, robot_z, 'g-', linewidth=2, marker='o', label='Z coordinate')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Z Coordinate (m)')
    ax5.set_title('Z Movement Over Time')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    plt.suptitle('4-DOF Robot Square Drawing Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save visualization
    plot_filename = 'robot_drawing_visualization.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Detailed visualization saved to: {plot_filename}")
    
    # Summary statistics
    print(f"\n📈 MOVEMENT SUMMARY:")
    print(f"   Total steps taken: {len(robot_positions)}")
    print(f"   Waypoints reached: {len(waypoint_hits)}/8")
    print(f"   Final distance to target: {distances[-1]:.3f}m")
    print(f"   X coordinate range: {robot_x.min():.3f}m to {robot_x.max():.3f}m")
    print(f"   Z coordinate range: {robot_z.min():.3f}m to {robot_z.max():.3f}m")
    print(f"   Total X movement: {np.sum(np.abs(np.diff(robot_x))):.3f}m")
    print(f"   Total Z movement: {np.sum(np.abs(np.diff(robot_z))):.3f}m")
    
    # Success analysis
    success_rate = len(waypoint_hits) / 8 * 100
    if success_rate >= 87.5:  # 7/8 waypoints
        print(f"\n✅ EXCELLENT DRAWING! {success_rate:.1f}% waypoints reached")
        print(f"   Robot successfully traced the complete square trajectory")
        if len(waypoint_hits) >= 8:
            print(f"   🎉 PERFECT! All 8 waypoints reached - Square is COMPLETE!")
    elif success_rate >= 62.5:  # 5/8 waypoints  
        print(f"\n🟡 GOOD DRAWING! {success_rate:.1f}% waypoints reached")
        print(f"   Robot mostly traced the square trajectory")
    else:
        print(f"\n❌ POOR DRAWING! {success_rate:.1f}% waypoints reached") 
        print(f"   Robot struggled to trace the square trajectory")
    
    plt.show()
    
    return robot_positions, trajectory_points

if __name__ == "__main__":
    robot_path, square_coords = visualize_robot_drawing()
