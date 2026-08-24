import json

from rain.cli.summarize import summarize


def test_summary_computes_seed_mean_and_std(tmp_path):
    paths = []
    for seed, accuracy in ((1, .6), (2, .8)):
        run = tmp_path / f"run{seed}"; run.mkdir()
        (run / "config.json").write_text(json.dumps({
            "data": {"name": "cifar100"}, "model": {"name": "resnet34-small"},
            "attack": {"name": "none", "malicious_clients": 0},
        }), encoding="utf-8")
        path = run / "rounds.jsonl"
        path.write_text(json.dumps({
            "round": 1, "accuracy": accuracy, "balanced_accuracy": accuracy,
            "asr": None, "wall_seconds": 3, "client_to_server_bytes": 10,
            "server_to_server_bytes": 20, "epsilon": 2,
        }) + "\n", encoding="utf-8")
        paths.append(path)
    result = summarize(paths)
    assert result[0]["runs"] == 2
    assert result[0]["accuracy_mean"] == .7
    assert abs(result[0]["accuracy_std"] - .1) < 1e-12
