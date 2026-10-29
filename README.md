# Orchard

A small job queue for two Macs. It sends long, CPU-heavy jobs from the Mac you
work on to a spare Mac on the same network, pauses them the moment someone uses
the spare Mac, and refuses any job the spare Mac can't run correctly.

It was built for one setup: a fast Apple-silicon MacBook Air that's always in use
and an old Intel MacBook Pro that was sitting idle. The main users are AI coding
agents (Claude Code, OpenCode) that run tests, builds and CI passes all day.
My agents and I built it together.

```
$ orchard run -b -- sh ./job.sh
placement: pro (remote)
  - background job offloaded to pro
queued: <job-id> (worker: pro)
```

The job's files go to the worker, the command runs there, and anything it writes
to `out/` comes back to `./.orchard-out/<job-id>/out/`.

## What it does

- **Gets out of the way.** The worker checks input idle time, power and thermal
  state about every 5 seconds. Keyboard or mouse input in the last 180 s freezes
  the job's process group with SIGSTOP; it resumes after 210 s without input.
  Battery power, battery under 20% (even on AC) and thermal throttling pause it
  too. A job paused for over 30 minutes is requeued and resumes from its
  checkpoints. These defaults come from BOINC.
- **Refuses instead of guessing.** Automatic placement checks required tags, RAM,
  free disk, and whether the job's Python requirements resolve for the worker's
  platform (via `uv pip compile --python-platform`). A job no machine can run
  correctly fails with a reason instead of running somewhere wrong.
- **Falls back.** If the worker is unavailable for about 15 minutes, queued jobs
  run on the controller instead, except jobs that need the worker. A job that
  loses contact mid-run is requeued, and the abandoned process is killed once the
  worker is back.
- **Stays small on the worker.** The worker side is one standard-library Python
  file that runs under the stock `/usr/bin/python3` (3.9+). Everything lives in
  `~/.orchard`; `rm -rf ~/.orchard` uninstalls it. No daemon, no sudo, no open
  ports beyond SSH.

## Requirements

- Two Macs on one network, with key-based SSH from the controller to the worker
  (Remote Login on).
- On the controller: [uv](https://docs.astral.sh/uv/). Orchard runs on Python
  3.12+, which uv provides.
- On the worker: nothing beyond macOS. On the first remote run Orchard installs uv
  and a managed CPython into `~/.orchard`.

## Install

```sh
git clone https://github.com/NuPhiberOptik77/orchard.git
cd orchard
uv sync
./bin/orchard init --host <worker>.local    # add --user <name> if the account name differs
./bin/orchard workers                        # probe the worker
./bin/orchard selftest                       # runs a job locally and on the worker
./bin/orchard dispatch --install             # keep the dispatcher running under launchd
```

Use `./bin/orchard` rather than `uv run orchard`; the launcher sidesteps a
hidden-`.pth` problem with editable installs on macOS. Add an alias or put
`bin/` on your PATH.

`orchard discover` lists Macs advertising SSH over Bonjour if you need the
worker's name.

## Use

| Command | What it does |
| --- | --- |
| `orchard run -b -- <cmd>` | Queue a job and return immediately |
| `orchard run -- <cmd>` | Run and wait; Ctrl-C detaches and the job carries on |
| `orchard build -t x86_64 -t swift -b -- swift test` | Build job: ships git-tracked files only, keeps a persistent `.build` cache on the worker |
| `orchard status`, `queue`, `jobs`, `workers` | What's happening (`--json` for scripts) |
| `orchard logs <job>` | Tail a job's log |
| `orchard cancel <job>` | Stop a queued or running job |
| `orchard stats` | Runs per machine, wait times, holds by reason |
| `orchard doctor` | Health check of the dispatcher, workers and disk |

Useful options: `-w local|<worker>` forces placement (problems become warnings),
`-t <tag>` requires a capability (`x86_64`, `arm64`, `xcode`, `swift`...),
`--timeout <s>` kills a job that hangs, and `--env NAME` forwards one environment
variable (only declared names are sent, and values are masked in output).

A job can also be a YAML file (`orchard run -f job.yaml`):

```yaml
name: embed-dataset
python: "3.12"
cmd: ["python", "embed.py", "--shard", "3"]
env: [HF_TOKEN]
timeout_s: 3600
requires: {mem_gb: 2, disk_gb: 1}
```

Long jobs should write checkpoints to `$ORCHARD_CKPT_DIR` and resume when
`$ORCHARD_RESUME` is set.

## Using it with AI coding agents

Tell your agents about it in their instructions file (for Claude Code,
`CLAUDE.md`): what the worker can't run, to always pass `--timeout`, to submit
with `-b` and check back once, and that a placement error is a correctness guard,
not something to work around with `-w`.

`bin/orchard-offload-hook.py` is an optional Claude Code `PreToolUse` hook. It
catches a few heavy commands (full pytest runs, `torchrun`, training scripts) and
tells the agent to send them through `orchard offload` instead:

```json
{
  "hooks": {
    "PreToolUse": [
      {"matcher": "Bash", "hooks": [{"type": "command", "command": "/path/to/orchard/bin/orchard-offload-hook.py"}]}
    ]
  }
}
```

## Menu-bar apps (optional)

- `./app/menubar/build.sh` builds the controller app: queue and worker status,
  notifications, and a dispatcher start/stop toggle.
- `sh app/workerbar/build.sh`, run on the worker, builds a status app with a
  pause toggle for whoever uses that Mac. It builds with the Command Line Tools
  alone.

Both are ad-hoc signed and install to `~/Applications`.

## Limits

Orchard is a personal tool, shaped around one user, two Macs and one home
network. The worker runs one job at a time. It moves code as a tarball over SSH,
so it suits source trees, not large datasets. There's no GPU support, and recent
PyTorch, TensorFlow and JAX releases don't ship Intel-Mac packages, so the
platform check keeps those jobs off an Intel worker.

[SPEC.md](SPEC.md) is the full design: placement, preemption, the job contract,
and the incidents that shaped them.

## Tests

```sh
uv run pytest
```

`orchard selftest` is the end-to-end check (`--local-only` skips the worker).

## License

MIT. See [LICENSE](LICENSE).
