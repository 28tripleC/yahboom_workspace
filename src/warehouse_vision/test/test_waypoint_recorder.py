# src/warehouse_vision/test/test_waypoint_recorder.py
import os
import tempfile
import yaml


def append_waypoint_to_yaml(path: str, x: float, y: float, oz: float, ow: float):
    """Mirrors the save logic we'll put in waypoint_recorder.py."""
    expanded = os.path.expanduser(path)
    if os.path.exists(expanded):
        with open(expanded) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    waypoints = data.get('waypoints', [])
    waypoints.append({'x': x, 'y': y, 'oz': oz, 'ow': ow})
    data['waypoints'] = waypoints
    with open(expanded, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def test_append_creates_file_if_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'waypoints.yaml')
        append_waypoint_to_yaml(path, 1.0, 2.0, 0.0, 1.0)
        assert os.path.exists(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data['waypoints'] == [{'x': 1.0, 'y': 2.0, 'oz': 0.0, 'ow': 1.0}]


def test_append_accumulates_waypoints():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'waypoints.yaml')
        append_waypoint_to_yaml(path, 1.0, 2.0, 0.0, 1.0)
        append_waypoint_to_yaml(path, 3.0, 4.0, -0.5, 0.85)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert len(data['waypoints']) == 2
        assert data['waypoints'][1] == {'x': 3.0, 'y': 4.0, 'oz': -0.5, 'ow': 0.85}


def test_append_preserves_existing_waypoints():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'waypoints.yaml')
        with open(path, 'w') as f:
            yaml.dump({'waypoints': [{'x': 9.0, 'y': 9.0, 'oz': 0.0, 'ow': 1.0}]}, f)
        append_waypoint_to_yaml(path, 1.0, 2.0, 0.0, 1.0)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert len(data['waypoints']) == 2
        assert data['waypoints'][0] == {'x': 9.0, 'y': 9.0, 'oz': 0.0, 'ow': 1.0}
