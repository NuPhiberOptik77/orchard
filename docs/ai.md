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
| `orchard ai serve` | Keep the model ready for OpenAI, Claude and Ollama tools. `--port`, `--keep-alive`. Ctrl-C puts everything back. |
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

## Keep it ready for your tools

`orchard ai serve` keeps the model ready for the tools you already use. It
listens on this machine only and speaks the APIs they expect:

| Your tool speaks | Point it at |
| --- | --- |
| OpenAI Chat Completions or Responses | base URL `http://127.0.0.1:6724/v1` |
| Claude Messages | base URL `http://127.0.0.1:6724` |
| Ollama | `OLLAMA_HOST=127.0.0.1:6724` |

```console
$ orchard ai serve
ai: ready at http://127.0.0.1:6724 for big-model
ai:   OpenAI tools: base URL http://127.0.0.1:6724/v1
ai:   Claude tools: base URL http://127.0.0.1:6724
ai:   Ollama tools: OLLAMA_HOST=127.0.0.1:6724
ai: nothing loads until the first question; it unloads after 600 s without one
```

Nothing loads until the first question. That one loads the model the way `ask`
does: it holds the machine, starts the GPU helper and its tunnel, pauses the
apps and waits for memory. Then the model stays loaded, and the apps stay paused,
so the next questions skip all of that. Everything is put back, in the same
order as after `ask`, when:

- nobody has asked for `keep_alive_seconds`
- someone starts using the machine (Orchard pauses the hold job)
- memory runs low, by the same rules as `ask`
- the engine or its link stops
- the model has been loaded for `max_loaded_seconds`

The next question loads it again. A question that runs past `cap_seconds` is
stopped, and the model stays loaded.

One question runs at a time; the rest wait their turn. While the model loads, a
streamed answer gets a comment line every 10 seconds, so the tool knows the
server is still there. If the model can't be loaded (the machine stays busy,
memory is short, or an `ask` is running), the tool gets a 503 with
`Retry-After: 60`, and the record says why.

Every answer gets a sealed record like an `ask`. Its id comes back in the
`X-Orchard-Record` header, and `orchard ai show ID` checks it. Each loading has
its own record, `ais-...`, with how long it took to load, the lowest free memory
seen, whether the apps came back, and why it unloaded.

**What it lets in.** It listens on `127.0.0.1` only. It refuses a request
addressed to any other host name, and any request from a web page, unless that
page's origin is in `origins`. It has no password: any program you run on this
machine can use it, as with Ollama. On the engine machine, llama.cpp's server
listens on a socket file only your account can open, not a network port, and is
reached through the same ssh that Orchard already uses. If that ssh session ends
for any reason, the engine stops.

**The APIs.** OpenAI and Claude requests go straight to llama.cpp's server, so
they support what it supports. Ollama requests are translated: `chat`,
`generate` (an empty prompt loads the model), `tags`, `show`, `ps` and
`version`, with the options `temperature`, `top_p`, `top_k`, `seed`,
`num_predict`, `stop`, `repeat_penalty`, `min_p`, `presence_penalty` and
`frequency_penalty`, JSON and schema `format`, `tools`, and `think: false`.
There are no images, embeddings, model downloads or deletes, per-request
`keep_alive`, or `num_ctx`: the context is fixed when the model loads. Ollama
tools show a digest; this one names the model setting, not the files' bytes,
because hashing a model this big at every start would take too long.

The server reads the same settings file, plus its own section:

```toml
[server]
port = 6724
keep_alive_seconds = 600        # unload after this long without a question
max_loaded_seconds = 21600      # then unload anyway, once nobody is waiting
context = 0                     # 0 uses engine.context; tools that send long chats need more
name = ""                       # what tools see; empty uses the model's file name
args = []                       # more llama-server flags
origins = []                    # web origins allowed in, like "vscode-webview://..."
socket = "~/.orchard/ai-engine.sock"   # the engine's socket file, on the engine machine
```

It uses llama.cpp's `llama-server` from the engine folder, which the release
that `build-engine.sh` downloads already has. Flags in `engine.args` that only
the one-question engine takes, like `-no-cnv` or `-st`, are left out.

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

- **`ask` loads the model for each question and shuts it down.** `serve` keeps
  it loaded between questions, but its first question still waits for the load.
- **`serve` doesn't start by itself.** Start it yourself, or from your own login
  item. It serves one model, and the engine has to be on a machine reached over
  ssh.
- **A foreground Orchard job on that machine isn't seen.** Those jobs don't appear
  in the queue. The checks find a running engine, and the hold keeps queued jobs
  off, but a job started in the foreground can still overlap a question.
- **You name the machines.** It doesn't pick free ones or search for the best
  split. Use `orchard ab` to compare splits.
- **One question at a time per Orchard install.** A second `ask` is refused while
  the first runs, and `ask` is refused while `serve` has the model loaded.
  `serve` queues its own questions.
- **The machine lending its GPU isn't held in the queue.** Only the engine's
  machine is. A job that Orchard starts on the GPU machine during a question
  keeps running, and if it squeezes that machine's memory, the question stops
  and `serve` unloads the model. Other work comes first.
- **Pausing apps needs ssh to Windows.** Windows' own OpenSSH server has to be on,
  because WSL can't reach Windows processes when interop is off.
