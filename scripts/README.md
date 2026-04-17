# NPU Remote Execution Scripts

These scripts let Claude Code (and the 9-agent system) execute code on the remote NPU server while keeping all source editing, plan management, and analysis on the local Mac.

## Prerequisites

1. SSH key auth set up to the NPU server (see `.claude/agents/README.md` for one-time setup).
2. `~/.ssh/config` has an entry named `npu` (or override via `NPU_HOST=...`).
3. `ssh npu "hostname"` succeeds without password prompt.

## Scripts

| Script | Purpose |
|---|---|
| `npu_sync.sh` | Rsync the local project to the NPU server. Called before every remote run. Creates a default `.rsync-exclude` on first run. |
| `npu_run.sh "<cmd>"` | Run an arbitrary command remotely in the project directory. |
| `npu_test.sh <op>` | Full test loop for one operator: sync → pytest → layout check → pull logs to `logs/`. |
| `npu_shell.sh` | Drop into an interactive shell on the NPU, landing in the project directory. |

## Environment overrides

All scripts honor these env vars (defaults in parentheses):

- `NPU_HOST` (`npu`) — SSH host alias
- `NPU_PATH` (`~/pypto-multi`) — remote project directory

Example: `NPU_PATH=~/scratch/pypto-multi ./scripts/npu_sync.sh`

## Examples

```bash
# One-time verify
./scripts/npu_run.sh "hostname && uname -a && python --version"

# Sync only
./scripts/npu_sync.sh

# Full test loop for the relu operator
./scripts/npu_test.sh relu

# Run a specific test with extra pytest args
./scripts/npu_test.sh matmul -k precision -v

# Interactive debugging shell on the NPU
./scripts/npu_shell.sh
```

## Adjusting `npu_test.sh`

The default test command is `python -m pytest custom/<op>/ -v`. If your project uses a different entry point (e.g. a custom `test_runner.py` or Makefile), edit the ssh block inside `npu_test.sh`. The Verification Agent reads the log under `logs/<op>.log` regardless of how it was produced.

## Log layout

- `logs/<op>.log` — latest log from the NPU, overwritten each run (matches what lives on the NPU)
- `logs/<op>_YYYYMMDD_HHMMSS.log` — timestamped archive of each local orchestration run
- `logs/<op>_latest.log` — symlink to the most recent timestamped archive
