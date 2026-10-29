# SuperDex research code

This is a code-only snapshot of the active Lambda workstation project at source
revision `82e34d18c39ac0344432ddb5499d3e811550b5d1`, synchronized on 2026-10-29
(Asia/Shanghai). Four GPU experiment processes were running when inspected.
Synchronization did not modify the workstation checkout or restart its jobs.

The current source includes the A1 predictability experiment pipeline, shared
PPO-Sim collection, W0/W1 world models, evaluation, and bounded reset-memory
diagnostics. Experiment completion or positive research results are not implied
by this source snapshot.

## Layout

- `scripts/a1_full_pipeline.py`: A1 orchestration.
- `scripts/train_a1_shared_collector.py`: shared PPO-Sim collection.
- `scripts/train_a1_world_model.py`: world-model training.
- `scripts/a1_world_model.py`: world-model definitions.
- `scripts/a1_prediction_eval.py`: prediction evaluation.
- `scripts/diagnose_a1_reset_memory.py`: bounded reset-memory diagnostics.
- `configs/`: versioned experiment settings and budgets.
- `requirements-*.lock.txt`: recorded environment dependencies.
- `activate_mjw.sh`: workstation environment selection.

Treat this `research/` directory as the project root when using the scripts.
The workstation's existing environment, parent Wuji assets, prepared datasets,
and runtime files are required; this is not a self-contained training package.
Read the entry point and its configuration before starting a new experiment.
No experiment is launched by downloading this branch.

## Scope and provenance

Source scripts, launchers, dependency locks, and configurations are preserved
from the source revision. `source_manifest.json` records their source SHA-256
hashes. The source `.gitignore` is extended with credential exclusions; this
README replaces the workstation's historical run log.

The non-secret `configs/workstations/lambda.env` contains only environment,
device-selection, and path defaults. It contains no credentials.

Datasets, trajectories, checkpoints, model weights (including MANO), caches,
experiment outputs, private research PDFs, SSH material, and the workstation's
Git history are not included. Credentials must stay outside Git. Authenticate
locally with SSH or the connected GitHub integration.

The existing project website is retained at repository root.
