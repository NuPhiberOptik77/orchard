# Orchard — personal compute spillover for two Macs

Status: M4 complete — all milestones done. M1/M2 smoke-tested on both machines
(remote runs with dependencies, live pause/resume preemption, queue + dispatcher,
worker GC). M3: controller menu-bar app on the Air + worker menu-bar app on the Pro,
launch-at-login and auto-dispatch, owner pause flag, `orchard status --json`, fair
managed-CPython benchmark, `orchard discover`. M4: build-job class with capability
tags and incremental SwiftPM builds on the Pro (verified end-to-end). Owner: personal
use (single user, two machines, one LAN).

## 1. Problem & goals

Two Macs sit on the same Wi-Fi LAN: a fast Apple-silicon MacBook Air (primary, always
in use) and an older Intel MacBook Pro (secondary, mostly idle). We want long-running
data/ML jobs and, later, build jobs to flow to the idle machine automatically, without
ever making the primary or the secondary feel worse to use.

Goals:

- `orchard run -- cmd` decides where a job executes (local vs. remote) using declared
  job requirements and measured worker capabilities — never guesswork.
- Remote work is opportunistic: it pauses the moment the secondary's owner is active
  and resumes later, BOINC-style.
- Correctness first: platform mismatches (arch, Python, missing wheels) are detected at
  submit time, not mid-run.
- Tiny, reversible footprint on workers: everything lives under `~/.orchard`.

Non-goals (for now): GPU distribution, live migration of a running process, native
Windows workers, mobile apps, automatic macOS version upgrades. Linux workers,
including a Windows PC running Linux under WSL2, are supported but opt-in per job
(§5, `contrib/windows-wsl/`).

## 2. Verified facts (probed 2026-10-18)

Machines:

| | Air (controller + worker) | Pro (worker) |
|---|---|---|
| Model | MacBook Air M5, Mac17,4 | MacBook Pro 2019 13", MacBookPro15,4 |
| CPU | Apple M5, 10 cores | Intel i5-8257U, 4c/8t @ 1.4 GHz |
| RAM | 16 GB | 8 GB |
| OS | macOS 27.0.1 | macOS 15.4 (last supported: it cannot run macOS 26+) |
| Toolchain | Xcode 27.1, Swift 6.4, SDK 27.0 | CLT only (Swift 6.0.3, SDK 15.2); no Xcode, no Homebrew |
| Disk free | ~67 GB | **~7 GB (critical constraint)** |
| Link | Wi-Fi, measured ~43 MB/s (350 Mbps), 55–70 ms RTT to Pro | same |
| SSH | key auth already working (passwordless) | ditto |

Upstream support facts that shape the design:

- `uv` is the 2026 industry standard for Python env management (74% most-admired,
  SO 2025; >30% of new Python repos; enterprise-scale migrations). It self-bootstraps
  via curl and needs no Homebrew.
- python-build-standalone ships CPython 3.10–3.15 for `x86_64-apple-darwin`
  (macOS 10.15+), so `uv` can provision the Pro's interpreter with zero system changes.
- PyTorch: last Intel-Mac binaries = 2.2.2 (binaries discontinued 2.3.0, Jan 2024).
  JAX: Mac x86 wheels dropped in 0.5.0 (Jan 2025). TensorFlow: last Intel-Mac wheels = 2.16 (dropped in 2.17.0, Jul 2024).
  => torch/jax/tf jobs are hard-pinned to the Air; the Pro takes numpy/pandas/
  sklearn/polars-class CPU work.
- BOINC's preference model is the reference implementation for opportunistic desktop
  computing; its defaults are adopted below almost verbatim.
- `taskpolicy` (background QoS), `pmset -g therm`, `pmset -g batt`, `ioreg
  HIDIdleTime`, `caffeinate` all exist on both machines — the preemption primitives
  are available without installing anything.
- Both machines' `/usr/bin/rsync` is openrsync (protocol 29, no real delta), so v1
  avoids rsync entirely; transport is tar.gz over SSH into a content-addressed store.

## 3. Architecture

```
Air (controller, always on)                       Pro (worker, opportunistic)
┌─────────────────────────────┐                  ┌──────────────────────────────┐
│ orchard CLI (uv project)    │   SSH (push)     │ ~/.orchard/                  │
│  - job intake & placement   │ ──────────────▶  │   bin/orchard-agent.py       │
│  - CAS (sha256 blobs)       │                  │   tools/uv  python/  envs/   │
│  - state/jobs/<id>/         │   status JSON    │   inbox/<sha>.tgz (CAS)      │
│  - state/queue/ + dispatch  │ ◀──────────────  │   jobs/<id>/{work,log,status}│
│                             │                  │   policy.json (preemption)   │
└─────────────────────────────┘                  └──────────────────────────────┘
```

- Controller: the `orchard` CLI itself. No server daemon; `orchard dispatch --watch`
  is run on demand (or started from the menu-bar app). Owns config, queue, state,
  CAS, and placement decisions.
- Menu-bar app (M3, Air): a thin SwiftUI client (`app/menubar/`, `MenuBarExtra`,
  ad-hoc-signed `Orchard.app`). It polls `orchard status --json` (12 s, overlap-
  guarded), shows worker/queue/job state, starts/stops `dispatch --watch` via the
  pidfile, queues a test job, and posts notifications on job completion. All logic
  stays in the CLI — the app is a viewer/remote-control, not a second brain.
- Agent: a single stdlib-only Python file (`agent.py`), Python ≥3.9 compatible so it
  runs under stock `/usr/bin/python3` on any Mac before any bootstrap. It supervises
  child jobs (its own process group), writes status JSON + logs, supports
  pause/resume via SIGSTOP/SIGCONT, enforces timeouts.
- Transport: SSH in push mode. Jobs are launched detached (`nohup`) so they survive
  SSH/controller disconnects; the controller polls status over SSH. ControlMaster is
  used to keep polling cheap.
- Data: every input tree and artifact tree is a tar.gz blob addressed by sha256 in a
  content-addressed store. Local CAS on the Air; remote inbox CAS on the Pro.

## 4. Job model

YAML at submit (or a `-- cmd` shortcut); internally converted to JSON for the agent
(agent has no YAML dependency).

```yaml
name: embed-dataset
kind: run                 # or "build": build jobs skip the Python env sync
tags: [x86_64, swift]     # capability tags the worker must satisfy
python: "3.12"            # uv-managed interpreter spec
requirements: uv-exports # auto-captured from .venv (pip freeze) or requirements.txt
env: [HF_TOKEN]           # names of env vars to forward (captured at submit)
cmd: ["python", "embed.py", "--shard", "3"]
timeout_s: 3600
requires: {mem_gb: 2, disk_gb: 1}
```

Contract: the job reads inputs from the working directory and writes artifacts to
`./out/`. Local runs execute in the current working directory; remote runs execute in
a staged copy on the worker. Artifacts are always pulled back to
`./.orchard-out/<job-id>/`.

Env forwarding: `env:`/`--env` names are looked up in the submitter's environment,
stored in the job spec (same-user homes only), and merged into the job's process env
on whichever machine runs it. Only declared names ever leave the Air; undeclared
variables are not forwarded and missing ones are reported at submit.

Checkpoint contract (settled in M2): the agent exports `ORCHARD_JOB_ID`,
`ORCHARD_ATTEMPT` (the launch number: 1, then one higher after each requeue or
fallback; also written to status.json and the `finished` event),
`ORCHARD_CKPT_DIR` (= `<job dir>/ckpt`, deliberately outside the staged workdir so
re-staging never wipes it) and, when the checkpoint dir is non-empty at start,
`ORCHARD_RESUME=1`. Jobs write checkpoints into `$ORCHARD_CKPT_DIR`. A preempted job
is first SIGSTOPed in place (all of its group except `wrap`, which stays up to
record a command killed while paused); if the pause exceeds 30 min it is killed and requeued,
restarting with the last checkpoint (or from scratch if it wrote none).

Input shipping: projects are tarred on the Air (streamed through a temp file, not
held in RAM) and pushed into a content-addressed inbox on the worker. Inside a git
repo the selection is gitignore-aware — build jobs ship tracked files only, run jobs
ship tracked plus untracked-but-not-ignored files, so `data/`-style dirs never cross
the wire. Outside a repo it falls back to a walk with standard excludes. Trees over
the size cap (2048 MB, `ORCHARD_MAX_TAR_MB` to raise) are rejected at submit with an
actionable error instead of being transferred.

Artifact collection is streamed to disk on the Air and guarded: `out/` over 2 GB
warns (`ORCHARD_ARTIFACT_WARN_MB` to change), and a pull that would leave the Air
under 3 GB free is refused with the artifacts left on the worker
(`ORCHARD_ARTIFACT_FORCE=1` overrides). The `du` probe doubles as the size check, so
nothing large is ever buffered in RAM.

Build jobs (M4, `orchard build ... -- <cmd>`): for compile/test/lint runs on a
specific architecture. They ship the working tree — tracked files only (`git ls-files`
when the cwd is a git repo; otherwise a walk with the standard excludes) — skip Python
env sync and preemption still applies. Each controller-side project path maps to a
persistent tree `~/.orchard/builds/<sha256(cwd)[:12]>/work` on the worker: on every
run the sources are staged fresh into a swap dir, the previous `.build` is moved into
it, and the tree is atomically replaced. This keeps the package path stable across
runs, which is required for incremental SwiftPM builds (clang's module cache and PCH
embed absolute paths; a per-job dir with a symlinked `.build` cache poisons them).
Measured on the Pro for a tiny swift-testing package: 148 s cold, ~10 s cached,
2.4 s after a one-file edit. Build output (`.build`) stays on the worker; `out/` is
collected as usual.

## 5. Placement & eligibility

Eligibility gates (hard): platform tag (`aarch64|x86_64-apple-darwin`), RAM, free disk
(worker must retain ≥2 GB after the job), and a submit-time cross-platform resolve:
the requirements set is dry-run resolved for the worker's platform with
`uv pip compile --python-platform <platform>`; failure = ineligible, with the reason
recorded. Capability tags (M4): a job's `tags:`/`--tag` must be a subset of the
worker's tags — probed by the agent (machine arch, `macos<major>`, `swift`, `clang`,
`git`, `make`, `uv`, `xcode` vs `clt`) unioned with static `tags` from workers.toml.
Missing tags read `missing tags: <list>`; when a tagged job has no eligible worker it
errors out instead of silently running locally (tagged builds on the wrong arch are
never useful).

OS tags: the agent adds `os:macos` or `os:linux` to the worker's tags, and Linux
workers resolve requirements against `x86_64|aarch64-unknown-linux-gnu`. A job's
`os:*` tags are an allow-list, not required tags, and a job with none allows only
`macos`. So no existing job can land on a Linux worker: it has to ask with
`-t os:linux`, or `-t os:any` for anywhere. An agent too old to report an OS tag is
a Mac. A mismatch reads `worker OS is linux; job allows macos (add -t os:linux to
allow it)`, and the fallback rule below treats it like any other job the Air can't
run.

Policy (M1):

- `--where auto` (default): interactive jobs run locally; `--background` jobs prefer
  eligible remote workers (score = weight × benchmark, weight defaults to 0.2 for the
  Pro as the benchmark-relative tiebreaker), falling back to local when ineligible.
- Benchmarks now run under a uv-managed CPython 3.12 (`~/.orchard/python` on both
  machines, installed on demand) so interpreter loops compare like-for-like: measured
  ≈40M ops/s (Air M5) vs ≈12.6M (Pro i5) — ≈3× — after the M3 fix (previously
  stock 3.9 vs project 3.14, not comparable).
- `--where local|pro|NAME` forces placement.

Background queue (M2): `orchard run --background` enqueues a job in
`~/.orchard/state/queue/<id>.json`; `orchard dispatch [--watch]` processes the queue.
A queued job starts only when its worker is reachable, still eligible (RAM/disk) and
idle: no HID input for ≥ 180 s + 30 s margin, on AC power, no thermal throttle. On a
Linux worker idle and power come from the host status file (§6); a status other than
`ok`, or a missing idle time, holds the job. One job per worker at a time. Running jobs are polled; finished ones are collected;
long paused ones are preempted and requeued.

The worker also refuses a second job while *anything* runs on it: every execution
(interactive or queued) writes a liveness-checked marker under
`~/.orchard/state/running/<id>.json`, and the dispatcher holds queued jobs when the
caps report a live `busy_job` — so a background job never stomps an interactive one
(or the shared build cache). Stale markers are pruned by pid-group liveness and
boot id and, on Linux, by the group leader's start time (field 22 of
`/proc/<pid>/stat`), because `wsl --terminate` restarts a distro with the same boot
id and reuses its pids. A group whose leader has exited can't be dated, so it goes
by liveness alone.
`orchard cancel <id|prefix>` stops a queued or running job: remote kills go through
the agent (with orphan-retry if the worker is unreachable), local kills signal the
job's process group. The agent's `kill` checks the pid file against the job's boot
id and start time, then signals the job's whole process group, so a command that
outlived its `wrap` (SIGKILLed, say) ends too. The running marker stays until
nothing of the group is left. A pause stops the group and lets `wrap` go on only
once it has started the command (`command.pid` beside `exit.json`), so a pause
that lands before that holds. When `wrap` ends with some of its group still alive,
the agent ends the rest. Only one dispatcher processes the queue at a time: `--watch`
holds a pidfile, and the one-shot form takes an O_EXCL tick lock, so a manual tick
can't double-start jobs alongside the watcher. Fencing: before launching, the
dispatcher asks the worker whether the job's process group is already alive (agent
`running` check); if so it adopts the job instead of relaunching — closing the
window where the Air dies after the remote agent started but before `running` was
recorded. Repeated launch failures count against `max_launch_errors` (default 6)
and then fail the job loudly instead of retrying forever; infra holds (unreachable,
busy, battery) reset that count so network flapping can't exhaust it, and total
starts are capped by `max_attempts` (default 20) to end death spirals. Forwarded
env values are masked in `queue --json`/`status --json` output.

Unavailability fallback: a job held because its worker is unreachable, paused, in
use, on battery, or thermally throttled flips to running locally once it has waited
`fallback_after_s` (dispatch.json, default 900) — the Air just does the work. Jobs
whose tags or platform needs exclude the Air (e.g. `x86_64`) keep waiting instead,
with the reason shown in the queue note. A job that loses contact mid-run for
`max_misses` polls (default 40 ≈ 10 min) is requeued locally the same way. A job whose
worker answers but whose progress (`attempt`, `elapsed_s`) hasn't moved for
`stall_polls` polls (default 4) is checked for a live agent process; if there is
none, as after the machine restarted under it, it is requeued on the same worker.
Every job runs under the agent's `wrap`, which writes `exit.json` when the command
ends, so a job that ran on to the end after its agent died is settled from that
record and collected, not run again; the settled status replaces the stale copy
collected with it. A job stopped by a signal its `wrap` also got,
as in a shutdown, is not settled and runs again. So is a job that signals its own
group, such as a shell with `trap 'kill 0' EXIT`: `wrap` records that signal too,
so if the agent also died the job runs again. A signal that reaches `wrap` before
its command has started is passed on once the command exists; one that lands in
the instant after, and so reaches the command directly as well, arrives twice.
When `settle` gives no clear answer
the job is left running and asked about after another quiet spell.
If the
Air can't run it, the job stays running on its worker, with the reason in its note
and one `held` event, and stops counting as that worker's busy job so the rest of
its queue can still fall back. Its status is collected as usual once the worker
answers; a worker that answers with no status for the job fails it. An abandoned
remote job from a lost-contact fallback, or from a cancel whose remote kill failed,
is remembered as an orphan, with its attempt number. When the worker answers
again, the dispatcher reads the orphan's status before acting:

- still running or paused: killed;
- finished under a different attempt number: ignored as stale;
- succeeded while the local rerun is queued or running: adopted. Its log, status
  and artifacts are collected as the job's own, the rerun is stopped, and the
  rerun's log and status move to `<state dir>/rerun`;
- any other finish (a failure, or a success after the rerun already ended):
  collected beside the rerun into `<state dir>/orphan` and
  `.orchard-out/<id>/orphan`.

When neither `settle` nor the plain status gives a clear answer, the orphan is
asked about again on later passes, up to `ORPHAN_ASKS` (20, five minutes at the
default poll). After that a readable status that still says running is killed,
through the agent's fences, and an orphan with no readable status is let go. A
running orphan with no pid yet (its agent may still be starting) whose kill finds
nothing is asked about again the same way, and after `ORPHAN_ASKS` passes it is let
go as a no-op. A kill that fails because ssh did (exit 255 or a timeout), or that
the agent could not record (exit 3), keeps the orphan for the next pass.

Each case logs an `orphan` event with the outcome and `duplicate: true` when the
job was launched again after the orphan's attempt. Contact loss can't be told apart from a dead worker in time to avoid
the rerun, so a job whose side effects must not repeat should key them on
`ORCHARD_JOB_ID` and check `ORCHARD_ATTEMPT`.

Preempted runs: a run paused past `kill_paused_after_s` is sent `kill`, which
sends SIGTERM and leaves an empty `kill-requested` file in the job folder (one
already there is kept). A group that is already gone gets neither, and the agent
exits 1 as for any run that isn't running. On macOS a group isn't gone until its
agent has reaped the wrap (under a second). An agent that can't create the file
sends nothing and exits 3, which the dispatcher treats like a failed ssh
and asks again. The run's final status says whether the file was there
(`kill_requested`), so a run that ended by itself can be told from one the kill
ended; a run started by an agent too old to keep track (no `kill_marker` in its
status) says nothing and is taken to have been reached. From then on its agent
doesn't pause it again, and SIGKILLs it if it is still running `KILL_GRACE_S` (60 s)
later. The entry is requeued with `preempted_attempt`, and before starting it again
the dispatcher asks whether that run is still alive. While it is, the job waits on
that worker instead of starting a second copy, and once `kill_grace_s` (default 120)
has passed the dispatcher sends `terminate` (SIGKILL at once). A worker that can't
be reached is a hold like any other, so after `fallback_after_s` the job reruns on
the Air and the old run becomes an orphan marked `preempted`: it is killed once the
worker answers, and its output is collected beside the rerun. Its exit is the
preemption, so it isn't adopted, unless its final status says the kill never reached
it: that run ended by itself and is settled like any other orphan, so a success is
adopted. A kill whose reply was lost shows up as the run's final status under the
attempt being preempted. The job is requeued from it, unless that status says the
kill never reached the run; then the job is finished as the run ended. A status from any other
attempt is fenced off: an earlier attempt's counts as a missed poll, a later one's
is recorded. Preemptions don't count toward `max_attempts`. Each launch deletes the
last attempt's `status.json` first, so an early poll finds no status instead of how
that attempt ended.

Cancel during a start: when a cancel lands while the dispatcher is starting or
adopting a job, the dispatcher stops what it started, locally or on the worker
(leaving an orphan if the kill can't be sent yet). `orchard cancel` re-reads the
entry after acting and acts again if a tick started the job meanwhile, and it also
kills a requeued preempted run that may still be going. Any other orphan outcome
leaves the cancel in place. A run that ended by itself while the cancel was acting
(the agent found nothing to signal, or a local group was already gone) keeps the
state the tick records for it, and a local run whose status already says it ended is
reported as "already <state>" and left for the tick. The exception is a cancel whose
earlier pass left an orphan: the local run is still left alone, but the orphan is
recorded and the cancel stands.

Local liveness: local jobs record the agent's pid; if the status file still says
running but the process is gone (e.g. the Air shut down mid-job), the dispatcher
marks the entry failed instead of tracking it forever.

## 6. Preemption policy (M2, BOINC defaults adopted)

| Preference | Default | Source |
|---|---|---|
| Suspend when user active | yes; "in use" = input in last 3 min | BOINC default |
| Suspend on battery | yes | BOINC default |
| Suspend when non-orchard CPU > 25% | **off by default** (opt-in: `policy.json` → `max_other_cpu_pct`) | BOINC default available but noisy: macOS `ps` CPU is a decaying recent average and Linux's is each process's lifetime average, so a young process reads high on Linux. The dispatcher's own polls (`sshd`, `sshd-session`) add a few percent on any worker. The agent's own process group and the job's don't count; with no `ps` the reading is unknown and the limit is ignored |
| Thermal preemption | pause below limit 50, resume at 60 (hysteresis); macOS only | `pmset -g therm` |
| Short suspends | SIGSTOP in place (instant resume) | BOINC "leave tasks in memory" |
| Long suspends (>30 min) | dispatcher kills (SIGTERM, then SIGKILL after 60 s) and requeues; resumes from checkpoint; doesn't count toward `max_attempts` | BOINC unload model |
| Job process class | full speed; `caffeinate -is -w <pid>` beside the job (outside its group, so a job's `kill 0` can't end it). `background_qos: true` opts into nice 10 + `taskpolicy -b` | measured 2.4x slower wrapped on the Pro; preemption already protects the user |
| Low battery | pause below 20% even on (weak) AC | 2026-10-19 incident: 8 W charger drained the Pro under load |

Preemption applies only to jobs launched with `preempt: true` (remote background
jobs); interactive/local runs never pause. `~/.orchard/policy.json` on the worker is
re-read on every poll (~5 s), so policy edits apply live. Defaults:
`pause_if_user_active=true, idle_threshold_s=180, pause_on_battery=true,
thermal_pause_below=50, thermal_resume_at=60, max_other_cpu_pct=null,
background_qos=false, pause_on_low_battery_pct=20`.

Linux workers can't see the host's input or power, and under WSL the guest can't
either. The host writes them to `~/.orchard/host-status.json` (`ts`, `idle_seconds`,
`power`, `battery_pct`, `disk_free_gb`, `source`); `contrib/windows-wsl/` has the
Windows writer: a startup task that starts the distro and rewrites the file every
10 s, counting idle time from boot while nobody is signed in and passing on what a
sign-in task reports from the user's session otherwise, so neither a reboot nor a
sign-out takes the worker offline. The agent fails closed:
a file that is missing, unreadable, incomplete or older than 60 s by its mtime reads
as "user active, power unknown", which holds new jobs and pauses running ones, and
`host_status` says which. The thermal check is skipped there, and
`max_other_cpu_pct` sees only the guest's processes.

## 7. Resource guards

- Worker must retain ≥2 GB free disk after staging (Pro has ~7 GB: jobs are refused
  otherwise). Worker GC (`agent.py gc [max_age_days] [min_free_gb]`, defaults 14 / 3)
  removes unprotected job dirs and inbox blobs older than max_age_days, or until
  min_free_gb is restored; running/paused jobs are never touched, unless their
  process is gone and their status is older than max_age_days (the machine
  restarted under them). Nothing schedules it; the WSL contrib adds a cron line.
- On a Linux worker free disk is the smaller of statvfs and the host's
  `disk_free_gb`: a WSL disk is a file on the host's C: and can't outgrow it.
- RAM ceiling enforced on the 8 GB Pro via `requires.mem_gb`.
- Per-job timeout enforced by the agent (SIGTERM to process group, then SIGKILL).
- All remote state lives in `~/.orchard` — `rm -rf ~/.orchard` is the full uninstall.

### Service level objectives (what "works" means)

- **No duplicate execution**: a job's process group is never run twice — fencing check
  before every launch + adoption on restart (v0.6.2). A preempted run that may still be alive holds
  its job on that worker. The exceptions are fallbacks from a worker that stopped
  answering (§5): the old run is killed once the worker answers again, and its output
  is kept apart from the rerun's. A preempted run whose kill never arrived and that
  then ends on its own is finished as it ended, not rerun; an agent too old to say
  whether a kill reached the run counts as reached, so that run is rerun from its
  checkpoint.
- **No job loss**: every submitted job reaches a terminal state (done/failed/cancelled)
  with a note, even across dispatcher restarts.
- **Fallback bound**: when the worker is unavailable/reserved, an eligible queued job
  starts on the Air within `fallback_after_s` + one tick (~15 min).
- **Control latency**: cancel takes effect within one tick (~15 s); adopted jobs are
  reattached within one tick of dispatcher start.
- **Interruption cost**: a preempted job loses at most one checkpoint interval; re-runs
  resume via `$ORCHARD_CKPT_DIR` / `$ORCHARD_RESUME`.
- **Verification**: the pytest suite (unit) and `orchard selftest` (E2E canary — local
  job + remote job, artifacts and forwarded env asserted) must both pass; treat the
  canary as the release gate after any code change.

### Observability

- `~/.orchard/state/events.jsonl` is the durable trace: one JSON line per lifecycle
  event (`submitted`, `launched`, `held`, `adopted`, `fallback`, `requeued`,
  `finished`, `failed`, `cancelled`, `orphan`). "Held" carries the scheduler reason and is
  logged once per distinct reason per job (re-logged on dispatcher restart, which
  doubles as a restart marker). Rotation: one generation to `events.jsonl.1` at
  5 MB. Appends never raise — logging can't break job control.
- `orchard stats [--days N] [--json]`: runs per machine, finished counts, queue wait
  percentiles (submit→start), holds grouped by reason (digits normalized), recent
  events. This answers "has the worker actually done work / why is the Air holding".
- `orchard doctor`: control plane (dispatch pid, stale locks), workers (reachability,
  agent sha freshness, disk, paused-by-owner, idle), local disk/blob-cache/bench-cache,
  queue anomalies (queued while dispatch stopped, held > 1 h; a job waiting out a
preempted run is timed from the preempt). Exit 1 on failures.
- The menu-bar app pipes `dispatch --watch` output to /dev/null, so the event log —
  not that output — is the historical record.

## 8. Footprint & layout

Controller (Air): `~/.orchard/{workers.toml, dispatch.json (optional), app.json,
python/ (managed bench interpreter), state/jobs/<id>/, state/queue/<id>.json,
state/events.jsonl (+ .1), state/dispatch.pid, cache/blobs/, playground/}` plus
`./.orchard-out/<id>/` per project. Both menu-bar apps build under `app/menubar/build/` and `app/workerbar/build/`
in the repo and install to `~/Applications/` (they run from there; `app.json` points
the controller app back at the repo). Worker (Pro): `~/.orchard/{bin, tools/uv, python/,
envs/<key>, cache, inbox, jobs/<id>, builds/<key>/work (persistent build trees),
policy.json (optional)}`. Env cache key = sha256(python spec +
lock)[:16], so identical jobs reuse one venv. Measured after M2 smoke tests: Pro
`~/.orchard` = 105 MB total (67 MB managed CPython 3.12, 38 MB uv, ~350 KB
agent/jobs/envs). Full uninstall = `rm -rf ~/.orchard` (the uv-installed python
downloads are not used by anything outside Orchard).

## 9. Security

Single-user, LAN-only, same account on both machines. SSH key auth (already
established); no passwords stored; no inbound ports opened on the worker; no sudo
anywhere; agent executes only job JSON it is given, as the same user.

## 10. Milestones

- **M1 (done):** CLI + agent, static worker config, SSH push, job YAML, uv
  bootstrap on worker, submit-time platform resolve, resource guards, local + remote
  end-to-end, logs + artifact pull-back.
- **M2 (done):** background queue + dispatcher (`dispatch [--watch]`, `queue`),
  BOINC-style idle/battery/thermal preemption with live `policy.json`, SIGSTOP/
  SIGCONT pause-resume, kill-and-requeue of long pauses, checkpoint contract, worker
  GC. Smoke-tested: live pause at job start (aggressive policy) → resume after
  relaxing policy → success; queued job dispatched, collected, artifacts intact;
  GC freed expected MBs while protecting a running job.
- **M3 (done):** controller menu-bar app (SwiftUI `MenuBarExtra`, ad-hoc signed,
  notifications on job completion, dispatch start/stop, auto-dispatch and launch-at-login
  toggles via `SMAppService`, test-job button); worker menu-bar app on the Pro (job/idle
  status, pause/resume toggles the owner pause flag, launch-at-login); owner pause flag
  (`policy.json {"paused": true}` pauses remote jobs within one poll and makes the
  dispatcher hold queued work); fair benchmark under managed CPython 3.12 on both
  machines; `orchard discover` (Bonjour `_ssh._tcp` browse); `orchard status --json` +
  `--json` on workers/queue/jobs. Verified: worker app compiles and runs on the Pro
  with CLT-only Swift 6.0.3; controller auto-starts dispatch on launch; E2E owner-pause
  (job queued while paused → held "worker paused by owner" → flag cleared → dispatched,
  succeeded, artifact collected).
- **M4 (done):** build-job class (`orchard build`, `kind: build`, `--tag`), capability
  tags on workers (probed + static), tracked-file shipping, persistent per-project
  work trees with `.build` caching on the worker, GC coverage for build trees.
  Verified end-to-end on the Pro: `orchard build -t x86_64 -t swift -b -- swift test`
  → queued → dispatched when idle → tests passed on x86 (148 s cold, ~10 s cached,
  2.4 s incremental); untagged-job failure mode (`no worker with required tags`) and
  tag routing both confirmed. Note: CLT-only machines cannot run XCTest (`swift test`
  with iOS-style XCTest needs full Xcode); Swift Testing ships with CLT.

## 11. Open questions

- Whether to upgrade both machines to Homebrew rsync 3.4 for delta transfers once
  datasets outgrow full-blob shipping (data-driven).
- Duty-cycle policy (run only at night / max hours per day) — deferred until real
  workload patterns emerge; the owner pause flag covers the ad-hoc case.

Resolved in M3.5: worker app = read-only status + pause/resume toggle, no policy
editor; it builds fine with CLT-only Swift 6.0.3 (~2 min first build). Launch-at-login
= `SMAppService.mainApp` toggles in both apps + an auto-dispatch toggle in the
controller app (no launchd LaunchAgent needed).

### Known gaps

- `kill_remote` reads every agent exit code other than 0, 3 and 255 as "not
  running", including a missing `python3` (127). Only 1 means that.
- A process that leaves the job's group (`setsid`) counts as other CPU, and the
  job's kill doesn't reach it.
- macOS has no process start time to check, so the pid fence there rests on
  `boot_id` alone.
- The two SIGSTOPs of a pause are 50 ms apart. A process forked in that gap on macOS
  can run on unpaused.
- A run that no signal ends (uninterruptible sleep) holds its job on that worker
  with no fallback; the dispatcher keeps sending `terminate`.
- A preempted run that ends on its own after its kill reply was lost is finished as
  it ended only when the agent records that no kill reached it. An older agent that
  doesn't record this is taken as reached, and the run reruns from its checkpoint.
- A cancel saved in the instant between a tick's last re-read of the entry and its
  write can still be overwritten.
- The mirror case: a tick that records a run's end in the instant between cancel's
  last re-read and its save is overwritten by the cancel.
- Cancel saves the copy of the entry it read before its kill, so it can re-arm an
  orphan that a tick settled meanwhile.
- A previous exec of the same job that stalls past its relaunch can clear the new
  attempt's kill marker.
- A worker that keeps returning another attempt's status counts those polls as
  misses, so the job ends in a lost-contact fallback after `max_misses`.

## 12. Dev notes

- Use `./bin/orchard` to run the CLI from this repo. Reason: uv/Hatchling write the
  editable-install `.pth` with the macOS `hidden` flag on some rebuild paths, and
  hardened `site.py` (Homebrew Python 3.12+) skips hidden `.pth` files, breaking
  `uv run orchard` intermittently. If that happens:
  `chflags nohidden .venv/lib/python*/site-packages/_editable_impl_*.pth`.
  The `bin/orchard` launcher bypasses the editable install entirely (PYTHONPATH=src).
- `taskpolicy` lives at `/usr/sbin/taskpolicy` (both machines); the agent resolves
  tool paths via `shutil.which`, so don't hardcode `/usr/bin/...`.
- Remote bootstrap is automatic on first remote run: uv installer → `~/.orchard/tools`,
  managed CPython → `~/.orchard/python`, job venvs → `~/.orchard/envs/<key>`.
- The two Macs can use different account names: set `user` for the worker in
  `~/.orchard/workers.toml` (or `orchard init --user`). SSH key auth from the
  controller account to the worker account must already work.
- Remote commands run under zsh: `dd of=~/...`-style arguments don't get tilde
  expansion (that's a bash assignment-word rule); use `$HOME`/absolute paths in ssh
  one-liners.
- Preemption smoke tests: write `~/.orchard/policy.json` on the worker
  (`{"idle_threshold_s": 999999}` pauses any preemptible job; `{"idle_threshold_s": 0}`
  resumes it), then remove the file to restore defaults. Dispatch tuning (poll
  interval, kill-after-paused, hold on battery, local-fallback grace) via
  `~/.orchard/dispatch.json` on the controller.
- Controller menu-bar app: `./app/menubar/build.sh` builds + installs
  `~/Applications/Orchard.app` (ad-hoc signed, `LSUIElement`) and writes
  `~/.orchard/app.json` with the repo path. It shells out to `bin/orchard status --json`
  every 12 s and can start/stop `orchard dispatch --watch` via
  `~/.orchard/state/dispatch.pid`; on launch it auto-starts dispatch when that toggle is
  on and the pidfile is empty. macOS privacy redacts other apps' status items in
  screenshots — verify the icon visually or via
  `lsappinfo find bundleID=io.orchard.menubar` (UIElement) plus observed
  `orchard status` child processes of the app pid.
- The menu-bar app starts dispatch with the GUI's bare PATH (`/usr/bin:/bin:/usr/sbin:/sbin`),
  and local jobs inherit the dispatcher's environment. `cli.main()` therefore appends
  the `dispatcher.BASE_PATH` dirs a PATH lacks (`dispatcher.merged_path`, the same merge
  the launchd plist uses), so `env.require_uv()` finds Homebrew's uv. Before 2026-10-29
  every local launch from an app-started dispatcher failed "uv not found on PATH".
- Worker menu-bar app: `sh app/workerbar/build.sh` on the Pro (`swift build -c release`
  works CLT-only; ship the source dir over ssh, no repo checkout needed), installs
  `~/Applications/OrchardWorker.app` (bundle id `io.orchard.worker`). It polls
  `~/.orchard/jobs/*/status.json` + `/usr/bin/python3 ~/.orchard/bin/orchard-agent.py
  status` every 8 s; Pause writes `~/.orchard/policy.json {"paused": true}` (merged,
  other keys preserved). `open` over ssh works for launching it; verify with `pgrep`.
- Owner pause flag semantics: `{"paused": true}` in worker `policy.json` pauses running
  remote jobs on the next agent poll (≤5 s) and shows in caps as `policy_paused`, so
  the dispatcher holds queued jobs ("worker paused by owner") and the controller app
  appends "· paused by owner" to the worker line. Local runs are never paused.
- Jobs launched via the dispatcher collect artifacts to `<cwd>/.orchard-out/<job-id>/out/`
  (`cwd` = where `orchard run` was invoked), not into `~/.orchard/state/jobs/`; that
  directory keeps only log.txt/status.json/placement.json/spec.json.
- `orchard dispatch --watch` watches the mtime of `src/orchard/*.py` and re-execs itself
  (unlinking the pidfile first) when code changes. A long-running dispatch otherwise
  keeps stale modules in memory after edits, which presents as per-entry
  "dispatch error: …" notes until it is restarted by hand. `os.execv` keeps the same
  pid, so an unchanged pid does *not* mean the restart didn't happen.
- Worker perf notes (measured 2026-10-19 on the Pro): the old default wrap
  (`nice -n 10` + `taskpolicy -b`) cost ~2.4x throughput (13.8M → 5.7M ops/s); jobs now
  run at full speed unless `background_qos: true`. The bench in `collect_caps` is cached
  for 24 h in `~/.orchard/state/bench.json` (keyed by interpreter path) because every
  status poll used to re-run it. `ensure_agent_uploaded` touches
  `~/.orchard/.metadata_never_index` so Spotlight stops indexing job/build trees.
  SSH uses ControlMaster (ControlPersist 300 s) — without it, per-poll connections
  storm `opendirectoryd` on the worker (~23% CPU observed).
- Power matters more than software on the Pro: an 8 W charger discharges the battery
  under load even while "on AC" (risking shutdown mid-job and software throttling);
  the agent now pauses below 20% battery (`pause_on_low_battery_pct`) even when pmset
  reports AC. If a worker feels mysteriously slow, check `pmset -g batt` /
  `ioreg -rn AppleSmartBattery` (adapter watts, negative amperage = discharging).
- Compare benches only within one interpreter: on the Pro, CLT py3.9 sustains
  ~10.5M ops/s and uv py3.12 ~13.4M (both `BENCH_CODE`); a hot turbo burst briefly
  reads ~18M and one-off readings are not reliable. The wrapped-vs-plain deltas
  (2.4x on py3.12, 1.3x on py3.9) are the trustworthy numbers.
- Do not cache SwiftPM build dirs behind symlinks across different job directories:
  clang module caches and PCHs embed absolute paths and fail with "PCH was compiled
  with module cache path …" / "missing required module 'SwiftShims'". Build jobs keep
  one stable work tree per project instead (see section 4).
- XCTest is not available in CLT-only toolchains (`swift test` → `no such module
  'XCTest'`); Swift Testing ships with the CLT (`import Testing`, tools-version 6.0,
  `platforms: [.macOS(.v14)]` for the async macros). On the Pro prefer `swift build`
  or swift-testing suites; keep XCTest suites on the Air.
- The agent runs under the **stock `/usr/bin/python3` (3.9.6 on both Macs)**, while the
  CLI/tests run on the venv (3.14). Keep `agent.py` stdlib-only and 3.9-compatible;
  `tests/test_agent.py::test_agent_runs_under_stock_python` enforces it by executing
  the agent with the stock interpreter. `orchard selftest` is the E2E canary (local +
  remote leg, artifacts + env forwarding asserted); `--local-only` skips the worker leg.
- Event call sites: launches in `engine.launch_remote`/`launch_local` (one
  implementation shared by CLI and dispatcher), submissions/cancels/finished in
  `cli.py`, everything scheduler-side (held/adopted/fallback/requeued/terminal) in
  `dispatcher.py`. Keep it that way when adding launch paths, or `orchard stats`
  undercounts.
- `env.capture_requirements` drops local-path requirements from `uv pip freeze`:
  `name @ file://…` and any `-e …` line (editable installs of the project itself
  resolve to Air-absolute paths with spaces and can never work on the Pro). The
  pytest suite isolates the event log (`conftest.py` autouse fixture) so tests
  never write into the real `events.jsonl`.
