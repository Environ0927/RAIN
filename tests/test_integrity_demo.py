from rain.cli.integrity_demo import run_checks


def test_integrity_demo_detects_every_supported_tamper_case():
    report = run_checks(clients=4, dimension=11, seed=3)
    assert report["accepted_clean_batch"] is True
    assert report["all_tampering_detected"] is True
    assert all(report["cases"].values())
