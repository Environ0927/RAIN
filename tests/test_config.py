from pathlib import Path

from rain.config import load_config


def test_all_versioned_configs_parse():
    root = Path(__file__).parents[1] / "configs"
    paths = list(root.rglob("*.json"))
    assert len(paths) >= 10
    for path in paths:
        assert load_config(path)["schema_version"] == 1
