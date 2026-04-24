#!/bin/bash
# Usage:
#   ./start_warehouse.sh          — full system
#   ./start_warehouse.sh test     — aruco + camera only

WS=~/ROS/yahboom_ws
S="source /opt/ros/humble/setup.bash && source $WS/install/setup.bash && cd $WS"
SESSION="warehouse"

tmux kill-session -t $SESSION 2>/dev/null
tmux new-session -d -s $SESSION

# Each command wrapped with 'trap kill 0' so Ctrl+C kills all child processes
wrap() {
    echo "bash -c 'trap \"kill 0\" EXIT SIGINT SIGTERM; $1'"
}

if [ "$1" = "test" ]; then
    tmux rename-window -t $SESSION:1 "bringup"
    tmux send-keys -t $SESSION:1 "$(wrap "$S && ros2 launch yahboomcar_bringup yahboomcar_bringup_launch.py")" Enter

    tmux new-window -t $SESSION -n "camera"
    tmux send-keys -t $SESSION:2 "$(wrap "sleep 3 && $S && ros2 launch yahboom_esp32_camera yahboom_esp32_camera_launch.py")" Enter

    tmux new-window -t $SESSION -n "nav"
    tmux send-keys -t $SESSION:3 "$(wrap "sleep 5 && $S && ros2 launch yahboomcar_nav navigation_dwb_launch.py")" Enter

    tmux new-window -t $SESSION -n "aruco"
    tmux send-keys -t $SESSION:4 "$(wrap "sleep 8 && $S && ros2 run warehouse_vision aruco_detector")" Enter

    tmux new-window -t $SESSION -n "rviz2"
    tmux send-keys -t $SESSION:5 "$(wrap "sleep 6 && $S && ros2 launch yahboomcar_nav display_launch.py")" Enter

else
    tmux rename-window -t $SESSION:1 "bringup"
    tmux send-keys -t $SESSION:1 "$(wrap "$S && ros2 launch yahboomcar_bringup yahboomcar_bringup_launch.py")" Enter

    tmux new-window -t $SESSION -n "camera"
    tmux send-keys -t $SESSION:2 "$(wrap "sleep 3 && $S && ros2 launch yahboom_esp32_camera yahboom_esp32_camera_launch.py")" Enter

    tmux new-window -t $SESSION -n "nav"
    tmux send-keys -t $SESSION:3 "$(wrap "sleep 5 && $S && ros2 launch yahboomcar_nav navigation_dwb_launch.py")" Enter

    tmux new-window -t $SESSION -n "aruco"
    tmux send-keys -t $SESSION:4 "$(wrap "sleep 8 && $S && ros2 run warehouse_vision aruco_detector")" Enter

    tmux new-window -t $SESSION -n "patrol"
    tmux send-keys -t $SESSION:5 "$(wrap "sleep 12 && $S && ros2 launch warehouse_vision patrol_launch.py")" Enter

    tmux new-window -t $SESSION -n "rviz2"
    tmux send-keys -t $SESSION:6 "$(wrap "sleep 6 && $S && ros2 launch yahboomcar_nav display_launch.py")" Enter

fi

tmux attach-session -t $SESSION
