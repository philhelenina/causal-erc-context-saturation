# Run scripts

These launchers reproduce the multi-seed experimental layout used during development. They assume execution from the repository root or a configured `SENTICCRYSTAL_ROOT` environment variable.

```bash
export SENTICCRYSTAL_ROOT=/path/to/causal-erc-context-saturation
bash scripts/run/run_all_n10_flat.sh
bash scripts/run/run_all_n10_hier.sh
```

The scripts in `cluster/` are machine-specific examples and may require edits for GPU IDs, paths, or scheduler environments.
