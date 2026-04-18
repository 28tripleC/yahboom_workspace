# src/warehouse_vision/test/test_patrol_node.py
import os
import tempfile
import pytest
import yaml


def load_waypoints_from_yaml(path):
    """Helper that mirrors the logic we'll add to patrol_node."""
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"Waypoints file not found: {expanded}")
    with open(expanded) as f:
        data = yaml.safe_load(f)
    waypoints = data.get('waypoints', [])
    if not waypoints:
        raise ValueError("Waypoints file is empty or has no 'waypoints' key")
    return [(wp['x'], wp['y'], wp['oz'], wp['ow']) for wp in waypoints]


def test_load_waypoints_returns_tuples():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'waypoints': [
            {'x': 1.0, 'y': 2.0, 'oz': 0.0, 'ow': 1.0},
            {'x': 3.0, 'y': 4.0, 'oz': -0.5, 'ow': 0.85},
        ]}, f)
        path = f.name
    try:
        result = load_waypoints_from_yaml(path)
        assert result == [(1.0, 2.0, 0.0, 1.0), (3.0, 4.0, -0.5, 0.85)]
    finally:
        os.unlink(path)


def test_load_waypoints_raises_if_file_missing():
    with pytest.raises(FileNotFoundError):
        load_waypoints_from_yaml('/tmp/does_not_exist_xyz.yaml')


def test_load_waypoints_raises_if_empty():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'waypoints': []}, f)
        path = f.name
    try:
        with pytest.raises(ValueError, match="empty"):
            load_waypoints_from_yaml(path)
    finally:
        os.unlink(path)
