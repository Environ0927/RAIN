# Anonymous review package contents

The package contains the source implementation, tests, locked configurations,
paper manifests, reviewer documentation, Slurm templates, and pinned external
baseline installers needed to inspect and execute the artifact.

It intentionally excludes datasets, checkpoints, incomplete experiment outputs,
private cluster operations, exploratory stress and seed-selection utilities,
caches, deployment archives, third-party checkouts, and Git metadata. External
repositories are installed at the exact commits recorded in
`paper/external_baselines.json`.

`ARTIFACT_MANIFEST.sha256` covers every distributed file except the manifest
itself. Run:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m rain.cli.audit_artifact --root . --strict-open-science
```

The strict open-science check succeeds only after the anonymous artifact URL is
configured and the package contains no detected identity-bearing metadata.
