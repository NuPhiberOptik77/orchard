# CPU sampler

Shows where one program's main thread spends its CPU time, without rebuilding
the program. It was written to find out why a large model answered slowly under
llama.cpp, and the report knows llama.cpp's log format.

Linux x86-64 only: it reads the instruction pointer straight from the signal
context.

```bash
gcc -O2 -Wall -shared -fPIC -o libsamp.so samp.c
LD_PRELOAD=$PWD/libsamp.so SAMP_OUT=$PWD/run.bin SAMP_NAME=llama-completion \
  llama-completion <your usual arguments> 2> run.err
python3 report.py run.bin run.err
```

- `SAMP_OUT` is where samples go. A copy of `/proc/self/maps` lands next to it
  as `<SAMP_OUT>.maps`.
- `SAMP_NAME` is the program to sample (default `llama-completion`). Anything
  else that starts with the library loaded runs untouched, so helpers are safe.
- Samples come every 10 ms of the thread's own CPU time, up to 64 MB of them,
  so a thread that is waiting costs nothing and shows nothing.
- `report.py <samples> <err log> [min percent]` resolves names with `nm -C`,
  takes the seconds per word from llama.cpp's `eval time =` line, and prints a
  call tree and a list of the functions the time was spent in.

It shows where the time went, not what would make it faster. To prove a change
helps, test it with [`orchard ab`](../../docs/ab.md).
