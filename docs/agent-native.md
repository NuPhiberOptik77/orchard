# Agent-native Orchard

Orchard started as spillover: a job the Air doesn't need to run goes to a free
machine. AI coding agents change what "a job" is. An agent edits, runs a
check, reads the failure, edits again, dozens of times an hour, and most of
those checks are ones it (or the fleet) has already run on an identical tree.
This document designs six features that turn the fleet from "somewhere else
to run a command" into machines that cooperate on one agent's work. They are
listed in build order: each one leans on the ones before it.

| # | Feature | Command | Leans on |
| --- | --- | --- | --- |
| 1 | Receipts | `orchard receipt`, `run --reuse` | — |
| 2 | Machine consensus | `orchard consensus`, `run --consensus` | receipts |
| 3 | Many worlds | `orchard fork` | receipts, consensus runner |
| 4 | Speculative runs | `orchard speculate` | receipts (reuse is the payoff) |
| 5 | Living replicas | `orchard replica` | receipts, speculation |
| 6 | Checking alongside the AI | `orchard live` | replicas, speculation |

Everything here is opt-in. A plain `orchard run`, `build`, `offload` or a
queued job behaves exactly as before; the only change on those paths is that
a finished job also leaves a receipt.

How we measure it: **edit to verdict** (seconds from an agent's edit to a
trustworthy pass/fail) and, for the agent loop as a whole, **rounds, tokens
and minutes to a green fix**, with and without each feature. The baseline is
a full CI-style run on the Pro (about 10.5 minutes for the reference project)
or the Air (about 2.7).

---

## 1. Receipts

**Journey.** An agent says "tests pass". Today you take its word, or rerun
everything. With receipts, `orchard receipt <job>` shows a sealed record:
the exact input tree, command, environment names, machine, exit code and the
digest of what it produced. If the agent asks for the same check on the same
tree again, `orchard run --reuse -- npm test` answers in milliseconds from
the receipt instead of running for minutes.

**Mechanism.**

- At launch, Orchard hashes the tree that ships (the same file list
  `iter_shipped` tars: path, exec bit, content or symlink target) into a
  **tree digest**, and writes `inputs.json` in the job's state folder.
- The **work key** is the digest of everything that decides the result:
  tree digest, command, kind, Python spec, requirements text, required tags,
  and the forwarded env names *and values* (values are folded into the
  digest and never stored).
- When the job finishes, Orchard seals `receipt.json`: the inputs, the worker
  (name, architecture, OS family), state, exit code, timings, and digests of
  the log and of the collected `out/` folder. The seal is HMAC-SHA256 with a
  per-install key (`~/.orchard/receipt.key`, created 0600 on first use).
- A receipt index (`~/.orchard/state/receipts/<work key>/<job>.json`) makes
  "has this exact work already passed?" one directory lookup.

**What the seal proves.** That this Orchard install recorded these facts and
nobody edited them since. It is tamper-evident, not third-party proof:
whoever holds the key can mint receipts. Worker-held signing keys (so a
worker vouches for its own run) are the next step and need an asymmetric
signature; the stdlib has none, so it waits until a dependency is worth it.

**Reuse rules.** Only `succeeded` receipts whose seal verifies are reused,
only for an identical work key, and only if the receipt's worker satisfies
the job's required tags. Reuse assumes the command is a function of its
inputs. Commands that read the network, the clock or files outside the tree
can still differ, which is why reuse is a flag, not a default.

**Modules.** New `receipts.py`; hooks in `engine.launch_remote`,
`engine.prepare_local_job`, `cli.run_local`, `cli.run_remote` and
`dispatcher.finish`. The agent adds `machine` and `os_family` to its status
so receipts name the architecture that really ran the job.

**Measure.** Share of agent checks answered from receipts in a working day;
seconds saved per hit.

## 2. Machine consensus

**Journey.** A test fails. Is it the agent's change, a flaky test, a
platform difference or a busy machine? Today that costs a human (or several
agent rounds) to find out. `orchard consensus -- npm test` reruns a failure
on the same machine and on every other capable one, and prints one label:

- `pass`: the first run passed; nothing to decide.
- `real`: it fails everywhere. Fix the code.
- `flaky`: the same machine passed on a rerun.
- `platform-only`: it fails only on one architecture or OS.
- `load-sensitive`: it failed on a busy machine and passed on idle ones.

**Fleet memory.** Each verdict is stored with a **failure signature** (the
last error-looking line of the log, with numbers, paths and hex ids
normalised). When the same signature comes back, the verdict says how it was
labelled before ("seen 3 times, flaky each time"), so agents stop chasing a
known flake, and repeat offenders become a list worth fixing.

**Modules.** New `consensus.py` (the classifier and memory are pure
functions) and `fleet.py` (runs a command on a chosen machine as a child
`orchard run`, so placement, environment sync and receipts are all reused).

## 3. Many worlds

**Journey.** An agent is torn between three fixes. Instead of trying them one
after another (three rounds, three waits), it writes three patches and runs
`orchard fork --patch a.diff --patch b.diff --patch c.diff -- npm test`. Each
patch gets its own copy of the tree on its own machine, they all run at
once, and Orchard prints the winner with its diff stat. One round instead of
three.

**Mechanism.** Each world is a copy-on-write clone of the shipped tree
(`cp -c` on APFS, a plain copy elsewhere) with one patch applied by
`git apply`. Worlds are spread over the free, capable machines and run
through `fleet.py`; each world leaves a receipt. With `--baseline`, an
unpatched world runs too, so "fails before, passes after" is proved, not
assumed.

**Modules.** New `worlds.py`.

**Measure.** Wall time to the first passing candidate versus trying the same
patches in sequence.

## 4. Speculative runs

**Journey.** The agent is still typing its explanation when the checks it is
about to ask for have already run on an idle machine. When it does ask
(`orchard run --reuse -- npm test`), the answer is a receipt that already
exists.

**Mechanism.**

- Each project has a list of likely checks: added by hand
  (`orchard speculate add -- npm test`) or learned from receipts
  (`orchard speculate learn` takes the commands most often run in this
  project).
- `orchard speculate run` computes the tree's work keys and queues, in the
  background, every check that has no receipt yet. `orchard speculate watch`
  does that each time the tree settles after a change, and cancels its own
  speculative jobs that went stale (never anyone else's jobs).
- Speculative jobs carry a `spec-` name prefix so they are easy to tell
  apart, and use the normal queue: a worker runs them only when it is idle.

**Modules.** New `speculate.py`.

**Measure.** Hit rate (speculated checks later asked for) and wasted CPU
(speculated checks never asked for).

## 5. Living replicas

**Journey.** Today each remote job ships the tree, unpacks it, and installs
dependencies before the first test runs. A replica is a warm copy of the
project that lives on a worker: dependencies installed once, caches kept.
After an edit, `orchard replica run -w pro -- npx jest src/foo` sends only
the changed files and starts the test in the warm tree. `orchard replica
check` splits the project's checks (or one suite, sharded) across every
replica and prints a live state: "tree 3f2a91: green on pro and local, red
on laptop".

**Mechanism.**

- `orchard replica up -w <worker> [--prep 'npm ci' --prep-key package-lock.json]`
  creates `~/.orchard/replicas/<project key>/work` on the worker.
- Sync is a manifest diff: the worker keeps the manifest of what it holds,
  Orchard sends a tar of only added or changed files plus a list of deleted
  ones. Untracked-and-ignored folders like `node_modules` never ship and are
  never deleted, which is what keeps the replica warm.
- `--prep` runs again only when a `--prep-key` file changes.
- `replica run` clears `out/`, runs the command through the normal agent (so
  pause, kill and timeouts all work), collects `out/`, and seals a receipt
  against the local tree digest, since sync made the replica equal to it.
- `replica check --split 'npx jest --shard={i}/{n}'` runs shard *i* of *n* on
  each replica at once and merges the verdicts.

**Self-healing.** Consensus memory plugs in here: a failure whose signature
was labelled `flaky` or `load-sensitive` before is rerun once on another
replica before it is reported.

**Not in this cut.** Resident watchers (a `tsc --watch` that never exits)
and streaming edits as they are typed. Both need a long-lived process on
the worker; the replica layout is where they will live.

**Modules.** New `replica.py` and a small worker-side script,
`replica_remote.py`, sent per call like the agent.

**Measure.** Edit to verdict through a replica versus a fresh job.

## 6. Checking alongside the AI

**Journey.** The model is writing a file. The moment it closes a function,
that function is checked. If the check fails, generation stops and the model
gets the exact failure while the context is still fresh, instead of
finishing a 300-line file built on a broken first function.

**Mechanism.**

- `orchard live` reads a stream of edits as JSON lines on stdin
  (`{"path": ..., "text": ...}` chunks for a file being written). It keeps a
  shadow copy of the tree, finds **syntactic boundaries** in the text so far
  (Python: the prefix parses; brace languages: depth back to zero after a
  closing brace), writes the complete prefix into the shadow copy and runs
  the check there. It prints one JSON line per check and exits non-zero at
  the first failure, which is the signal to interrupt the model.
- Checks the model *mentions* in its reasoning (a registered check command
  in backticks) are started speculatively before it asks for them.
- `contrib/live_check_agent.py` is a reference harness on the Claude API: it
  streams a `write_file` tool call, feeds the partial input to `orchard
  live`, and on a failure stops the stream and sends the failure back as the
  next user turn.

**Prior art and limits.** Research on execution-guided decoding checks
partial programs during generation, mostly at the token level inside the
decoder. This works outside the model, with real test suites, at function
boundaries. Claude Code doesn't expose its token stream, so this needs a
custom harness; inside Claude Code the nearest equivalent is speculation on
each completed edit.

**Measure.** Rounds, tokens and minutes to a green fix with and without
interruption, on the same task set.

---

## What could go wrong

- **Wrong reuse.** A non-hermetic check reused across a change it can't see.
  Mitigation: reuse is opt-in, the work key covers env values, and
  `orchard receipt` shows exactly what was hashed.
- **Speculation crowding out real work.** Speculative jobs queue like any
  other. Mitigation: they are cancelled when stale, and `speculate run` caps
  how many it queues at once.
- **Replica drift.** A replica edited by hand on the worker. Mitigation:
  `replica up --verify` rebuilds the manifest from disk before diffing.
