import json
import pytest


def test_raw_result_schema_example(tmp_path):
    rows = [{"round": 0, "accuracy": .2, "loss": 2.1},
            {"round": 1, "accuracy": .3, "loss": 1.9}]
    path = tmp_path / "rounds.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    decoded = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["round"] for row in decoded] == [0, 1]
