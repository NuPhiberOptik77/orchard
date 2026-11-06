# The engine for `orchard ai`

`orchard ai` runs [llama.cpp](https://github.com/ggml-org/llama.cpp). This
folder holds what makes it work on a model far larger than memory. Nothing here
is needed for anything else in Orchard.

| File | What it does |
| --- | --- |
| `build-engine.sh` | Downloads llama.cpp's release, checks its digests, builds the patched libraries and copies them in. |
| `patches/fast-patch.py` | Stops the engine reading the whole model at every start. `GGML_CPU_PREFETCH_EXPERTS=1` asks the disk for a layer's experts all at once, instead of one page fault at a time. |
| `patches/batch-patch.py` | `GGML_RPC_BATCH_FETCH=1` fetches a step's inputs from a GPU helper in one go. The wire format doesn't change, so the helper needs no rebuild. |

With neither setting on, the libraries take the same paths they always did, apart
from the read-ahead change, which applies only when the model is larger than
memory.

```bash
contrib/ai/build-engine.sh ~/llm/engine
```

On a Mac lending its GPU:

```bash
contrib/ai/build-engine.sh --helper ~/llm/helper
```

The script covers Linux x86_64 with NVIDIA, and Apple silicon helpers, because
those are what it was tested on. For anything else, run the two patch scripts on
a copy of the same llama.cpp tag and build it your way: each stops with a clear
error if the source doesn't match.

The model itself is yours to choose and download. Its licence is between you and
its maker. See [docs/ai.md](../../docs/ai.md) for the settings.
