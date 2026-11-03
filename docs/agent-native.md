# Agent-native Orchard

Orchard started as spillover: a job the Air doesn't need to run goes to a free
machine. AI coding agents change what "a job" is. An agent edits, runs a
check, reads the failure, edits again, dozens of times an hour, and most of
those checks are ones it (or the fleet) has already run on an identical tree.
This document designs seven features that turn the fleet from "somewhere else
to run a command" into machines that cooperate on one agent's work. They are
listed in build order: each one leans on the ones before it. Features 1 to 6
are built, as are two pieces of 7 that stand on their own (cross-OS proof,
and shared sight in section 5); the rest of 7 is a proposal.

| # | Feature | Command | Leans on |
| --- | --- | --- | --- |
| 1 | Receipts | `orchard receipt`, `run --reuse` | — |
| 2 | Machine consensus | `orchard consensus` | receipts |
| 3 | Many worlds | `orchard fork` | receipts, consensus runner |
| 4 | Speculative runs | `orchard speculate` | receipts (reuse is the payoff) |
| 5 | Living replicas, shared sight | `orchard replica`, `--sight` | receipts, speculation, consensus |
| 6 | Checking alongside the AI | `orchard live` | replicas, speculation |
| 7 | One mind | `orchard prove`, `-t os:proven` (built); `orchard mind` (proposed) | all of the above |

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

**Known gaps.** Inputs are hashed when the job launches, not when it is
submitted, so a queued job hashes the tree as it is at launch. A local job
runs in the project folder and writes `out/` there, which changes the tree
digest unless `out/` is in `.gitignore`. A queued job that the dispatcher
collects as an orphan (after a restart) is not sealed.

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
- `machine-only`: it fails on one machine and passes on another of the same
  platform that was not busier (a broken cache or a local config file).

`orchard consensus --job <id>` uses a job that already failed as the first
run. The exit code is 0 for `pass`, 1 for `real` and 2 for anything that is
not plainly the code, so an agent can branch on it. "Busy" means a load
average per core of 0.75 or more when the run started.

**Fleet memory.** Each verdict is stored with a **failure signature** (the
last error-looking line of the log, with numbers, paths and hex ids
normalised). When the same signature comes back, the verdict says how it was
labelled before ("seen 3 times, flaky each time"), so agents stop chasing a
known flake, and repeat offenders become a list worth fixing
(`orchard consensus --offenders`).

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

**Mechanism.** Each world is a copy of the files that would ship (a
copy-on-write `clonefile` on APFS, a plain copy elsewhere) with one patch
applied by `git apply`, which changes all of a patch or none of it. A patch
that does not apply is reported, not run. Worlds are spread round-robin over
the capable machines (or the ones named with `--on`) and run through
`fleet.py`, so each leaves a receipt. With `--baseline`, an unpatched world
runs too, so "fails before, passes after" is proved, not assumed. `--first`
starts no more worlds once a patch passes. The winner's world is kept (its
`out/` and logs live there); the rest are deleted unless `--keep`. Exit codes:
0 a winner, 1 none, 2 nothing could run.

Ignored folders a local check needs (`node_modules`, `.venv`, plus any
`--share DIR`) are linked into each world, not copied; they never ship,
because those names are excluded from every tarball. A patch that changes
dependencies should use `--no-share`, or it will test against the old ones.

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
  project). Learning counts only verified receipts of plain `run` jobs with
  the same requirements a speculative run would have, and never counts
  speculation's own jobs, so it cannot feed on itself. It adds only checks
  that are new and keeps any timeout already set.
- `orchard speculate run` computes the tree's work keys and queues, in the
  background, every check that has neither a receipt nor a job on it. At
  most 3 speculative jobs per project are in flight at once (`--max`); the
  rest are held. A submit that fails takes no slot.
- It cancels only jobs speculation itself queued: their ids are recorded per
  project when they are queued, and a job is never judged by its name. A
  stale run (its tree has moved on) and a second job on the same check both
  go.
- If the newest receipt for a check on this exact tree failed, the check
  shows `failed` and is not run again until the tree changes.
- Speculative jobs run with `--no-local`, so they never run on this Mac:
  not at placement, not when a held job falls back, and not when a job that
  lost contact is requeued. They exist to use idle machines, not the one
  being typed on. `--no-local` is an ordinary `orchard run` option too;
  it changes where a job may run, not what it computes, so it is not part
  of the job's receipt key.
- Runs for one project take a lock, so two `run` or `watch` calls cannot
  queue the same check twice.
- `orchard speculate watch` runs once at start and again each time the tree
  changes and then holds still for a few seconds (`--settle`). It retries
  held checks, and any error, once per settle period even if nothing
  changed, and an error never ends the watch. The tree fingerprint covers
  each file's path, mode, size and modification time.
- A queued job ships the tree as it is when it starts, so only running jobs
  can go stale.
- Speculative jobs carry a `spec-` name prefix so they are easy to tell
  apart, and use the normal queue: a worker runs them only when it is idle.
- Check lists and the record of queued ids live in the Orchard state dir,
  one per project, never in the project itself. `orchard speculate list`
  shows each check's state without queueing anything.

**Modules.** New `speculate.py`.

**Measure.** `orchard speculate stats`: hit rate (speculated checks later
asked for), wait saved (their run time, once per reuse) and wasted CPU
(speculated checks never asked for). Only recorded speculative jobs count.

## 5. Living replicas

**Journey.** Today each remote job ships the whole tree, unpacks it and
installs dependencies before the first test runs. A replica is a warm copy
of the project that lives on a body (a worker): dependencies installed once,
caches kept. After an edit, `orchard replica run -- npx jest src/foo` sends
only the changed files to the nearest free replica and starts the test
there. When it goes red, `orchard replica compare` says whether your change
did it, and `orchard replica blame` names the files that broke it.

**One genome, three ages.** The layer takes its shape from the Cleons in
Asimov's Foundation: one genome living in several bodies at different ages.

- The genome is the project tree. Every replica holds the same tree, and a
  digest proves it.
- **Dawn learns.** `orchard replica up -w <worker> --prep 'npm ci'` copies the
  tree, builds its Python env and runs prep. Prep runs again only when a prep
  key changes (lockfiles and manifests such as `package.json` by default;
  `--prep-key` chooses others), so editing source never reinstalls anything.
- **Day acts.** Day replicas run checks.
- **Dusk advises.** One replica may be the dusk (`up --role dusk`, or
  `orchard replica role <worker> dusk`). It holds the last tree a check passed
  on and does not follow your edits. `run --bless` moves it forward after a
  pass, and `replica check` moves it when every check passes on one tree;
  `up` never moves it. A plain run on the dusk is refused ("the dusk moves
  only on green"); `--as-is` runs on its copy without moving it. A blessed
  run that falls back to this Mac still moves the dusk when it passes.
- **Demerzel** is the ledger: every replica run seals a receipt, so the record
  of what passed where outlives any one body.

**Mechanism.**

- The copy lives at `~/.orchard/replicas/<project key>/work` on the body. The
  body keeps a manifest of what it holds (path, kind, exec bit, size, content
  hash) and its digest, computed exactly like a receipt's tree digest, so
  "this replica holds your tree" is a digest comparison, not a guess.
- Sync is a manifest diff: Orchard sends a tar of only the added or changed
  files, a list of deleted ones and the digest it expects the copy to start
  from. If the copy moved since Orchard looked (another sync, or a check that
  edited it), the body refuses, and Orchard reads its manifest again and
  retries. The body unpacks into a temporary folder, refuses any path that
  leaves the copy and any member it did not ask for, replaces a symlinked
  parent with a real folder, and checks the final digest before it answers.
- Each sync has its own names on the body, so two syncs never read each
  other's files. The body notes which files it is about to change before it
  touches them; a sync cut short (a crash, a lost connection) is finished on
  the next look, when those files are hashed again.
- A submodule, pipe or socket in the tree is named and refused ("cannot send
  pipe"): Orchard can only send files and links.
- Ignored folders like `node_modules` never ship and are never deleted, which
  is what keeps the replica warm.
- `replica run` clears untracked files from `out/`, runs the command through
  the normal agent (so pause, kill and timeouts all work), collects `out/` and
  seals a receipt against the tree digest. `--reuse` answers from a sealed
  passing receipt for the same tree, check and prep.
- A job starts only on the tree Orchard checked. The body refuses to begin
  if the copy moved since; Orchard brings it back and tries once more. When
  prep rewrote a tracked file, Orchard says so ("dawn: prep changed 1 tracked
  file(s) (a.txt); sending them back") and puts it back before the run.
- The body holds the copy for that job until Orchard has collected the
  results, then Orchard releases it. A sync cannot land in between.
- A body runs one job at a time: a replica job does not start while any
  other job runs there. Required tags and idle holds apply as usual. Naming
  a held body with `-w` for a single run still runs it, and says so ("note:
  pro: owner active (running anyway: you named it)"); `check`, `compare` and
  `blame` always pass held bodies over.

**Nearest mind.** Without `-w`, `run` ranks the free replicas that can take
the job by how many bytes each would need to be sent, and picks the closest.
A replica that already holds your tree costs nothing. Only if none can take
it does the job run on this Mac (`--no-local` forbids that).

**Succession.** A replica that is busy, or whose body does not answer, is
passed over, and Orchard says why: "succession: pro cannot take it (quiet:
no answer)". The next in line takes the run. A body that goes quiet during
a run (no answer for 10 minutes) is given up on: the run is reported as
unavailable ("succession: pro went quiet mid-run"; the job may still finish
there), and nothing is retried behind your back.

**Drift hygiene.** After each run the body compares its copy with what was
sent. A check that rewrote tracked files (a formatter, a snapshot update, a
cache written into the tree) is named ("hygiene: the check changed 1 tracked
file(s) in the copy: a.txt") and recorded in `drift.json` in the job's state
folder. The next sync puts those files back.

**Augment and borrow.** Some bodies cannot run some checks. There are two
opt-in ways around that; the placement guard itself is unchanged.

- **Augment.** Prep may install tools into the replica's own `tools/bin`
  (`$ORCHARD_REPLICA_TOOLS` names the folder), which goes first on PATH for
  every job in that replica. `up --augment node` records that this replica
  now carries the `node` tag, so a check that needs `-t node` can place there
  even though the body alone could not take it.
- **Borrow.** `replica check` sends a tagged check only to the replicas that
  can run it, and says so: "borrow: npm test needs node: only pro can run it".

**Check, compare and blame.**

- `replica check` runs every speculate check (or one suite split with
  `--split 'pytest --shard {i}/{n}' --shards N`) across the day replicas at
  once, giving the next piece of work to whichever body frees up first. Work
  from a body that goes quiet moves to another; work no body can run is
  reported, not dropped.
- `replica compare -- <cmd>` runs the check on your tree and on the dusk's
  copy at the same time and says which it is: "your change broke it (it
  passes at the dusk tree)", "it fails at the dusk tree too, so it is not
  your change", or green now. If the dusk's copy is no longer the tree it
  was blessed at (edited on the body, say), compare gives no verdict, and the
  next blessed run puts the copy back.
- `replica blame -- <cmd>` finds the fewest of your changed files that
  together make the check fail. It builds mixed trees (the dusk's version of
  most files, yours of a few), tests them in parallel across the day replicas
  with delta debugging, and prints the culprits. It never syncs the dusk,
  and refuses to start when the dusk is not the tree it was blessed at. A
  tree that does not finish (a timeout, a body that went quiet) ends with no
  verdict rather than a wrong one.

**Shared sight.** One run cannot tell a broken change from a sick machine or
bad luck. With `--sight`, `replica run` and `replica check` look at a failure
again straight away, with two pairs of eyes: a rerun on the same body and a
run on a second body, both on the same tree. The second body is the first
free day replica that can run the check (in `check`, one the check could
borrow), else this Mac. A body that cannot start its look hands it to the
next. Two replicas look at once; this Mac looks last, because a run here
works in place and anything it writes would change the tree the other look
ships. The verdict is one of three:

- **code**: it fails on the second body too. "fails twice on pro and on
  laptop: fix the code" (exit 1). This holds even when the rerun on the
  same body did not finish.
- **body**: it fails again on the same body but passes on the other. "fails
  on pro but passes on laptop (same platform): pro is the problem, not your
  change" (exit 2). The reason comes from consensus, so a load or platform
  difference is named.
- **chance**: the rerun on the same body passes, whatever the second body
  says. "failed then passed on pro: flaky" (exit 2).

Anything else is **unsure** (exit 1), never a guess: no second body was
free, or a look the verdict needed did not finish (a body went quiet, or the
tree changed since the first run). A check that writes into files that ship
(an `out/` folder that is not gitignored, say) changes the tree, so keep
outputs ignored. Verdicts go into consensus memory with the failure's
signature, so the next time the same error shows up, sight says how it was
called before. `replica check --sight` looks at each failure one at a time
after the checks finish, adds `sight: <verdict>` to the summary, and exits 2
when every failure was the body or chance. The dusk still moves only on
green. `--sight` cannot be used with `--as-is`, because it looks at your
tree.

**Disk.** A body's normal `gc` removes a replica only when it has been idle
for 14 days or more and nothing is running in it. When disk runs low, `gc`
still spares any replica used in the last 24 hours. `orchard replica down
<worker>` removes one now; if some files cannot be removed it says so and
keeps the replica listed. `orchard replica list` shows each replica's role,
whether it holds your tree, drift and pending prep.

**Not in this cut.**

- Resident watchers (a `tsc --watch` that never exits) and streaming edits as
  they are typed. Both need a long-lived process on the body; the replica
  layout is where they will live.
- Self-healing from consensus memory: running sight on its own, without
  `--sight`, when a failure's signature was labelled `flaky` before.
- **Dreaming.** An idle dusk could replay the day's red checks against small
  mixes of its green tree and the trees that failed, so that when the same
  failure signature comes back, blame is already known.

**Modules.** New `replica.py`; seven `replica-*` verbs in the agent (state,
apply, begin, release, meta, pack, down) and replica entries in its `gc`.

**Measure.** Edit to verdict through a replica versus a fresh job; for
blame, trees tested and minutes to the culprit versus bisecting by hand.

## 6. Checking alongside the AI

**Journey.** The model is writing a file. The moment it closes a function,
that function is checked. If the check fails, generation stops and the model
gets the exact failure while the context is still fresh, instead of
finishing a 300-line file built on a broken first function.

**Mechanism.**

- `orchard live -- <check>` reads JSON lines on stdin:

  ```
  {"path": "src/a.py", "reset": true}   start (or restart) writing a file
  {"path": "src/a.py", "text": "..."}   the next piece of it
  {"path": "src/a.py", "done": true}    the file is finished
  {"think": "..."}                      the model's reasoning or prose
  ```

- It keeps a shadow copy of the project (the files a job would ship, with
  `.venv`, `venv` and `node_modules` linked in; `--link <dir>` adds more).
  It finds **syntactic boundaries** in the text so far: in Python, a `def`
  or `class` (methods too) is closed when the text so far parses and the
  next code line is indented no deeper than it; in brace languages (C
  family, Go, Java, JS/TS, Kotlin, Rust, Swift and others), a block returns
  to depth 0. At each boundary it writes the finished part into the shadow
  copy and runs the check there. A burst of boundaries is coalesced into
  one check of the latest, and `done` always gets a final check.
- The check may use `{path}` and `{unit}` (the file, and the function that
  just closed). The same values are in `ORCHARD_LIVE_PATH` and
  `ORCHARD_LIVE_UNIT`, and `ORCHARD_LIVE_FINAL` is 1 for the finished file.
  `--timeout <s>` kills a check that hangs (its whole process group).
- While an existing `.py` file is rewritten, the definitions the new text
  has not reached yet are taken from the old file, so a test suite still
  imports. Brace languages get no such fill-in, so give them checks scoped
  to the file (`{path}`) rather than the whole suite.
- One JSON line per check goes to stdout: `{"event": "check", "path",
  "line", "unit", "final", "ok", "exit", "seconds"}`, plus `tail` (the last
  30 lines of output) and `why` on a failure. Bad input gets an `error`
  event and is skipped. At the end of input it prints `{"event": "done",
  "checks": n}` and exits 0. The first failure exits 1: the caller's signal
  to stop the model and hand it the failure. Exit 2 means no check command
  was given.
- Checks the model *names* in backticks, in its prose or reasoning, that
  match a registered `orchard speculate` check are queued once on an idle
  machine, so their answer may be waiting (`speculate` events;
  `--no-speculate` turns it off).
- `contrib/live_check_agent.py` is a reference harness on the Claude API
  (`pip install anthropic`, `ANTHROPIC_API_KEY`):

  ```sh
  python contrib/live_check_agent.py --root . --check "python -m pytest -x -q" \
      "Add a slugify() helper to src/text.py, with tests"
  ```

  It streams `write_file` tool calls (with eager input streaming, so the
  file arrives as it is written), feeds the partial contents to `orchard
  live`, and on a failure closes the stream, discards that reply and sends
  the failure (file, line, function, output tail) back as the next user
  turn. A check that fails after the reply is complete goes into the tool
  result instead. It turns on the server-side refusal fallback; that rescues
  refusals only, not overloads. Exit 0 when the model finishes, 1 when it
  stops early.

**Prior art and limits.** Research on execution-guided decoding checks
partial programs during generation, mostly at the token level inside the
decoder. This works outside the model, with real test suites, at function
boundaries. Claude Code doesn't expose its token stream, so this needs a
custom harness; inside Claude Code the nearest equivalent is speculation on
each completed edit.

**Measure.** Rounds, tokens and minutes to a green fix with and without
interruption, on the same task set.


## 7. One mind (proposed, built after 1 to 6)

**The image.** Three people walking together who no longer notice they are
three, because their understanding is so complete that all any of them
perceives is the one walker. For machines that means two things. From the
outside, there is one computer: you give it work and get answers, and no
machine name appears unless you ask. From the inside, no machine keeps a list
of "the others": each one plans over the same set of bodies, itself
included, with no special case for self.

**Journey.** Nobody asks "who takes this?", nobody waits for an answer, and
when one body goes quiet the rest already know exactly what it was doing.
The Mac and the Windows laptop are not two kinds of machine to choose
between; there is one fleet that has learned what each part of it can do.

**Why not a team that talks.** Messages are slow, can be lost, and every
"I'll take this" / "no, I've got it" exchange is a chance to disagree. No
machine can learn a new fact before the bits reach it, so the aim is not to
beat that limit. It is to make sure that the only thing that ever travels is
a new fact, never coordination.

**Mechanism.**

- **One memory.** Receipts, consensus verdicts and file digests are
  replicated to every machine (they are small, signed and append-only, so
  merging is a union). Any machine answers "has this been done, and did it
  pass?" from its own disk.
- **One will.** The plan is a pure function of (shared memory, goal). Every
  machine runs the same planner on the same memory and gets the identical
  assignment, so each knows its own part and everyone else's without being
  told. This is deterministic lockstep, as in strategy games and
  deterministic databases: only inputs travel, and every copy computes the
  same state. Two machines can never claim the same work, because there is
  no claiming.
- **One body sense.** Each machine sends a heartbeat of a few bytes (load,
  heat, which step of the plan it is on) a few times a second. The planner
  folds heartbeats into memory, so a machine that stops beating has its step
  reassigned by every copy at once, identically.
- **One self across Mac and Windows.** Today the wall between them is a
  rule, not the hardware: a job with no `os:` tag is treated as a Mac job,
  because Orchard cannot tell whether it would behave the same on the
  laptop's Linux. The rule stays the default, so existing jobs place exactly
  as before. Learning removes it where it is safe: `orchard prove -- <cmd>`
  runs a check on every reachable machine whatever its OS, and each pass is
  a sealed receipt. A job tagged `-t os:proven` may then go to macOS or to
  any OS whose latest sealed run of the same command in the same project
  passed; a newer failure there takes that OS away again. The answer is
  fixed when the job is submitted, so the dispatcher's later checks agree
  with it, and it never changes the job's work key. When the OSes
  disagree, consensus labels it platform-only and remembers why (a
  case-sensitive path, a missing tool), so the difference becomes knowledge
  instead of a wall. What cannot merge is physical: Xcode, iOS simulators
  and Apple frameworks run only on macOS on Apple hardware. One mind still
  lets each body do what it is able to, the way a hand does what an eye
  can't, and routes that work to the Mac without being told. Windows-native
  work (PowerShell, `.exe` builds) could join through WSL interop as an
  `os:windows` capability; not in this cut.
- **One train of thought** for local models. The Air holds the large model.
  The other machines run small drafts of the same model family that predict
  its next tokens and prepare the checks it will want (features 4 and 6).
  The large model accepts whole runs of predicted tokens in one pass, and
  only token ids cross the network.

**Prior art.** Making many machines look like one is old: single-system-image
clusters (MOSIX, OpenSSI, Kerrighed) migrated processes inside the Linux
kernel, between identical machines on a wired network, and Plan 9 gave a
whole network one namespace. Splitting one model's layers across machines exists (exo,
llama.cpp RPC) and is slow over Wi-Fi with an 8 GB Intel machine in the
mix. Speculative decoding exists inside one machine. Lockstep planning
exists in games and databases. The new part is combining them across
different operating systems over Wi-Fi, without kernel changes, with the
fleet learning from signed evidence what each body can do, so a home fleet
shares one memory, one plan and one train of thought.

**Measure.** Coordination messages per task (target: zero), time to recover
a dead machine's step, and tokens per second for the local model with and
without drafting machines.

---

## What could go wrong

- **Wrong reuse.** A non-hermetic check reused across a change it can't see.
  Mitigation: reuse is opt-in, the work key covers env values, and
  `orchard receipt` shows exactly what was hashed.
- **Speculation crowding out real work.** Speculative jobs queue like any
  other. Mitigation: they are cancelled when stale, and each project has at
  most 3 in flight at once.
- **Replica drift.** A replica edited by hand on the worker. Mitigation:
  before every sync and run the body hashes again any file whose size or
  modified time changed, so the edit shows up as drift and the next sync
  puts it back. A job starts only on the digest Orchard checked.
