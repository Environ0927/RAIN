import json

import pytest

from rain.cli.train import _prepare_raw_log


def test_resume_truncates_rows_newer_than_checkpoint(tmp_path):
    path = tmp_path / "rounds.jsonl"
    path.write_text("".join(json.dumps({"round": value}) + "\n" for value in range(5)), encoding="utf-8")
    _prepare_raw_log(path, start_round=3, resume=True)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["round"] for row in rows] == [0, 1, 2]


def test_fresh_run_refuses_to_append_existing_log(tmp_path):
    path = tmp_path / "rounds.jsonl"
    path.write_text(json.dumps({"round": 0}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already contains"):
        _prepare_raw_log(path, start_round=0, resume=False)
