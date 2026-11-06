# orchard ai

Ask a large model spread across your own machines, and keep a record of the answer.

A model can be far bigger than any one machine's memory if it reads most of
itself from disk and borrows another machine's GPU. Running one by hand takes a
dozen steps: keep other jobs off the machine, start the GPU helper and its
tunnel, pause the sync apps, wait for memory, run the engine, watch it, and put
everything back. Forget the last one and the sync apps stay paused. `orchard ai
ask` does all of it for one question and puts everything back however the run
ends, including Ctrl-C.

It is free, it uses only your machines, and nothing leaves them. It suits work
you can start and collect later, not chat: expect seconds per word on old
hardware.

## A real example

One question, asked of a 300 GB model on an old 4 core laptop that borrows a Mac's
GPU. The answer took just under 14 minutes:

```console
$ orchard ai ask "Explain in two sentences why the sky is blue."
--- thinking
We need answer two sentences. Need explain Rayleigh scattering. Concise. Ensure exactly two sentences. Maybe mention sunlight interacts with molecules, shorter blue wavelengths scatter more, reaches eyes from all directions. Second sentence: this scattered blue light dominates sky, while longer wavelengths pass through, and at sunset path longer etc? Need only why sky blue. Two sentences.
--- answer
The sky is blue because sunlight is scattered by molecules in the atmosphere, and shorter blue wavelengths are scattered much more strongly than longer red wavelengths. This scattered blue light reaches our eyes from all directions, making the sky appear blue.
--- end
answered in 825 s
4133.2 ms per word, 116 words, prompt 119.5 s
lowest free memory 334 MB
sync apps back
record ai-20261106-091520-92d5
```

## Setting it up

1. Build the engine on the machine that holds the model, and the helper on the
   machine lending a GPU. See [contrib/ai](../contrib/ai/README.md).
2. Download a model. That and its licence are yours to do: Orchard never fetches
   one.
3. Run `orchard ai setup`. It writes `~/.orchard/ai.toml` for you to fill in.
4. Run `orchard ai setup` again. It checks each machine and changes nothing: the
   engine, the model files, free disk, memory, which GPUs it found, and the
   Orchard worker.

## What happens when you ask

1. **Checks.** It refuses if the engine is already running there, a file is
   missing, the machine can't be reached, or another question is running. A
   refused question never reaches the model.
2. **Holds the machine.** With `worker` set, a sleeping job keeps Orchard's queue
   off that machine. If the queue can't start it within `hold_wait_seconds`, the
   question is refused.
3. **Starts the helper and tunnel**, if there's a `[helper]`.
4. **Pauses the apps** in `[windows] pause`.
5. **Waits for memory**, up to `settle_seconds`, until `start_free_mb` is free.
6. **Runs the engine** and streams the answer to you.
7. **Puts everything back**, in order: engine, apps, tunnel, helper, hold.
8. **Seals a record** in `~/.orchard/state/ai/`.

During the run, every 20 seconds it stops the engine if:

- free memory on the engine machine reads under `stop_free_mb` three times running
- free memory can't be read six times running
- the machine lending the GPU reaches critical memory pressure
- the hold job is paused, which means someone started using the machine
- the run passes `cap_seconds`

If the sync apps can't be confirmed back, the record and the report say so in
capitals. `orchard ai resume` tries again.

## Commands

| Command | What it does |
| --- | --- |
| `orchard ai ask "question"` | Ask. `-` reads the question from stdin. Exit 0 answered, 1 failed, 2 refused, 3 stopped. |
| `orchard ai setup` | Write the settings file, or check the machines against it. |
| `orchard ai show ID` | Show a record and check its seal. |
| `orchard ai list` | Recent questions. |
| `orchard ai resume` | Bring the paused apps back by hand. |

| `ask` option | Environment variable | What it does |
| --- | --- | --- |
| `--words N` | `ORCHARD_AI_WORDS` | Most words in the answer. |
| `--threads N` | `ORCHARD_AI_THREADS` | CPU threads. |
| `--split A,B` | `ORCHARD_AI_SPLIT` | Share of layers per GPU. |
| `--overrides TEXT` | `ORCHARD_AI_OVERRIDES` | Which tensors go where. |
| `--no-pause` | `ORCHARD_AI_NO_PAUSE` | Leave the sync apps running. |
| `--no-hold` | | Don't hold the machine in the queue. |
| `--json` | | Print the record. |

## Settings

They live in `~/.orchard/ai.toml`, never in a repo. This one lends a Mac's GPU to
a Windows laptop running the engine under WSL.

```toml
[engine]
host = "big-box"                # ssh name of the machine that holds the model
folder = "~/llm/engine"
model = "~/llm/models/big-model-00001-of-00006.gguf"
threads = 4
thinking_end = "</think>"       # this model reasons before it answers
devices = "CUDA0,RPC0"
split = "22,78"
overrides = "_exps|_shexp|per_layer_token_embd=CPU"
args = ["--temp", "0", "-no-cnv", "--no-display-prompt", "--no-warmup",
        "--no-op-offload", "-ub", "32", "-b", "32", "--jinja", "-cnv", "-st"]
start_free_mb = 5000
stop_free_mb = 200
worker = "big-box"              # its Orchard worker, so jobs wait for the run
worker_tags = ["os:linux"]

[engine.env]
GGML_CPU_PREFETCH_EXPERTS = "1"
GGML_RPC_BATCH_FETCH = "1"

[helper]                        # runs on the machine you ask from
folder = "~/llm/helper"
device = "MTL0"

[windows]                       # read free memory from Windows, pause apps there
host = "big-box-windows"
pause = ["OneDrive"]
```

Windows counts the memory that matters here, so with `[windows]` free memory is
read from there. Without it, it's read from the engine machine's `/proc/meminfo`.

Other settings, with their defaults: `context` 512, `layers` 99, `words` 200,
`cap_seconds` 2400, `hold_wait_seconds` 900, `settle_seconds` 300, `binary`
`llama-completion`. Under `[helper]`: `binary` `ggml-rpc-server`, `port` 50052,
`tunnel_port` 50072.

`thinking_end` is for a model that reasons before it answers: the mark that ends
its reasoning. With it, `ask` labels the reasoning, keeps it apart from the answer
in the record, and calls a run that hits the word limit before the answer failed.
Leave it empty for a model that doesn't reason first.

## Tuning it

[`orchard ab`](ab.md) finds out whether a setting really helps. The `ask` options
read environment variables so that arms can set them:

```bash
orchard ab --arm a:ORCHARD_AI_THREADS=4 --arm b:ORCHARD_AI_THREADS=8 \
  --metric '([0-9.]+) ms per word' \
  --same '(?s)--- answer\n(.*?)\n--- end' \
  --watch big-box \
  -- orchard ai ask "Explain in two sentences why the sky is blue." --words 32
```

A question's own hold job doesn't count as other load in `--watch`.

Moving work between CPU and GPU can change the wording of an answer, because the
two round numbers differently. `--same` then stops the test at the first changed
run, which is what you want until you've decided the new wording is acceptable.

## What it records

The sealed record holds the question, the answer and any reasoning in plain text
(`answer.txt` beside it keeps exactly what the engine printed), the settings,
the exact files that answered, timings, the lowest free memory seen, whether the
apps came back, and any problems. `orchard ai show ID` says whether it has been
changed since. Anyone who can read your Orchard folder can read your questions.

## Limits

- **Each question loads the model and shuts it down.** Nothing stays loaded
  between questions, so it's slow to start every time.
- **A foreground Orchard job on that machine isn't seen.** Those jobs don't appear
  in the queue. The checks find a running engine, and the hold keeps queued jobs
  off, but a job started in the foreground can still overlap a question.
- **You name the machines.** It doesn't pick free ones or search for the best
  split. Use `orchard ab` to compare splits.
- **One question at a time per Orchard install.** A second is refused while the
  first runs.
- **Pausing apps needs ssh to Windows.** Windows' own OpenSSH server has to be on,
  because WSL can't reach Windows processes when interop is off.
