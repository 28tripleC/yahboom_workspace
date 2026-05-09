#!/usr/bin/env bash
set -euo pipefail

WS="${WS:-$HOME/ROS/yahboom_ws}"

open_tab() {
    local title="$1"
    local dir="$2"
    local command="${3:-}"

    gnome-terminal \
        --tab \
        --title="$title" \
        --working-directory="$dir" \
        -- bash -lc "
            if [ -f /opt/ros/humble/setup.bash ]; then
                source /opt/ros/humble/setup.bash
            fi

            if [ -f \"$WS/install/setup.bash\" ]; then
                source \"$WS/install/setup.bash\"
            fi

            cd \"$dir\"
            $command
            exec bash
        "
}

if ! command -v gnome-terminal >/dev/null 2>&1; then
    echo "gnome-terminal was not found. Install it or use tmux instead."
    exit 1
fi

open_tab "bringup" "$WS" "ros2 launch yahboomcar_bringup yahboomcar_bringup_launch.py"
open_tab "camera" "$WS" "sleep 3; ros2 launch yahboom_esp32_camera yahboom_esp32_camera_launch.py"
open_tab "nav" "$WS" "sleep 5; ros2 launch yahboomcar_nav navigation_dwb_launch.py"
open_tab "rviz2" "$WS" "sleep 6; ros2 launch yahboomcar_nav display_launch.py"
