# orchard ab

Test whether a setting really helps, and keep the proof.

You changed something and the next run was faster. Was it the change, or was
the machine just quieter? `orchard ab` answers that by running the same command
under each setting, in an order that cancels out drift, and only says "proven"
when the gap can't be explained by the runs simply wandering.

## A real example

A large model answered slowly on a machine that also runs cloud sync apps. Does
pausing those apps help? The `.sh` files here are your own scripts.

```bash
orchard ab \
  --arm running \
  --arm paused \
  --before 'paused=sh pause.sh' \
  --after 'paused=sh resume.sh' \
  --metric 'seconds per word: ([0-9.]+)' \
  --same '(?s)ANSWER:(.*)' \
  --watch box-1 \
  --stop-if 'sh memory-is-low.sh' \
  --timeout 1800 \
  -- sh ask-the-model.sh
```

That runs `ask-the-model.sh` eight times: running, paused, paused, running,
running, paused, paused, running. Before each `paused` run the apps are paused,
and after it they come back, even if the run fails or you stop the test.

The numbers below are from the real test behind this example, run with a
hand-written script before this command existed: every run gave identical text,
and one run didn't count because another job landed on the machine. This is how
`orchard ab` reports those numbers.

```text
ab-20261105-091500-3f2a: proven
  metric: seconds per word: ([0-9.]+) (lower is better)
  running  median 3.5, 3.34 to 3.75, 3 of 4 runs count  [baseline]
  paused   median 3.065, 2.89 to 3.2, 4 of 4 runs count  better, -12.4%, luck 1 in 18
  proof: ~/.orchard/state/ab/ab-20261105-091500-3f2a.json
```

## What each option is for

| Option | What it does |
| --- | --- |
| `--arm NAME` or `--arm NAME:KEY=VAL,KEY=VAL` | One setting, passed to the command as environment variables. The first arm is the baseline. A value may contain commas. |
| `--rounds N` | Runs per arm (default 4). Round one goes forward, round two backward, and so on. |
| `--metric REGEX` | The number to compare: the last match in the output, first group if there is one. Without it, wall-clock seconds. |
| `--higher-better` | A bigger number wins, for things like words per second. |
| `--same REGEX` | Text every run must produce exactly. If it changes the test stops, because a faster wrong answer isn't faster. |
| `--before ARM=CMD`, `--after ARM=CMD` | Shell commands around each of that arm's runs. `--after` always runs, is retried three times, and says so loudly if it still fails. |
| `--watch WORKER` | A run that another Orchard job overlapped on that machine doesn't count, and the job is named in the record. |
| `--stop-if CMD` | Checked every `--poll` seconds during a run. Exit 0 kills the run and ends the test. Use it as a memory guard. |
| `--timeout N` | Seconds before a run is killed. |
| `--show ID`, `--list` | Show a finished test and check it, or list recent ones. |

Each run also gets `ORCHARD_AB_ARM` and `ORCHARD_AB_RUN` in its environment.

## What "proven" means

An arm is **better** or **worse** than the baseline only when their ranges don't
overlap at all, so the worst run of one still beats the best run of the other.
Then the luck check: if a setting did nothing, how often would runs separate
this cleanly anyway?

| Runs per arm | Luck | Counts as proven? |
| --- | --- | --- |
| 4 and 4 | 1 in 35 | yes |
| 4 and 3 | 1 in 18 | yes |
| 3 and 3 | 1 in 10 | yes, just |
| 2 and 2 | 1 in 3 | no: "too few runs" |

This is on purpose blunt. It misses small real gains, which then read as "not
separated". Run more rounds if the gain matters to you. It won't call noise a
win, as long as the runs really are comparable.

Exit codes: **0** proven, **1** not separated, **2** void (a run failed, a hook
failed, the text changed, the guard fired, you stopped it, or too few runs
counted). A test that goes wrong stops rather than burning hours on runs nobody
can use.

## The proof

Each test leaves `~/.orchard/state/ab/<id>.json`, updated after every run and
sealed at the end like a receipt, plus one log per run in `<id>/` with its
SHA-256 in the record. `orchard ab --show <id>` checks the seal and every log,
and exits 1 if anything was edited.

## Limits

- Runs on this machine, one after another. The command can `ssh` anywhere, but
  killing a local `ssh` may leave the remote process running, so do remote
  clean-up in `--after`.
- `--watch` only sees Orchard jobs. Other work on the machine isn't detected,
  which is why the order is mirrored.
- With three or more arms each is compared to the baseline on its own, with no
  correction for making several comparisons. Treat a lone win among many arms
  as a lead and test it again.
- Arm settings and hook commands are written into the record, so don't put
  secrets in them.
- Opt-in, like the other agent-native commands: nothing else in Orchard calls
  it, and plain `orchard run` is unchanged.
