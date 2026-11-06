# Patches a copy of llama.cpp b12091 for models far larger than RAM. usage: python3 -I fast-patch.py <src dir>
# 1. No load-time read-ahead when the model is larger than RAM (it read all 300 GB at every start).
# 2. GGML_CPU_PREFETCH_EXPERTS=1: before a layer's expert products, ask the kernel for every selected expert slice
#    at once, so the disk sees one deep queue of reads instead of serial page faults (measured 160-250 -> 1,060-1,220 MB/s).
import sys

src = sys.argv[1]


def patch(path, old, new):
    p = f'{src}/{path}'
    text = open(p).read()
    assert text.count(old) == 1, (path, old[:60], text.count(old))
    open(p, 'w').write(text.replace(old, new))


patch('src/llama-model-loader.cpp', '''#include <unordered_set>
''', '''#include <unordered_set>

#ifndef _WIN32
#include <unistd.h>
#endif
''')

patch('src/llama-model-loader.cpp', '''    if (use_mmap || lazy.any()) {
        mappings.reserve(files.size());''', '''    if (use_mmap || lazy.any()) {
#ifndef _WIN32
        // read-ahead over a model larger than RAM reads every byte at load and evicts most of it again;
        // without it, tensors used in place from the mapping are read when they are first needed
        if (prefetch && use_mmap) {
            size_t total = 0;
            for (const auto & file : files) {
                total += file->size();
            }
            const long pages     = sysconf(_SC_PHYS_PAGES);
            const long page_size = sysconf(_SC_PAGESIZE);
            if (pages > 0 && page_size > 0 && total > (size_t) pages * (size_t) page_size) {
                LLAMA_LOG_INFO("%s: model (%zu MiB) is larger than RAM, skipping read-ahead at load\\n", __func__, total >> 20);
                prefetch = false;
            }
        }
#endif
        mappings.reserve(files.size());''')

patch('ggml/src/ggml-cpu/ggml-cpu.c', '''static thread_ret_t ggml_graph_compute_thread(void * data) {''', '''#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

// Expert weights used in place from a memory-mapped model that does not fit in RAM page-fault in as small
// serial reads. With GGML_CPU_PREFETCH_EXPERTS=1, the first MUL_MAT_ID of a layer asks the kernel for the slice
// of every expert the layer selected, in all of that layer's MUL_MAT_IDs, before any thread touches them.
static bool ggml_cpu_prefetch_experts_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_CPU_PREFETCH_EXPERTS");
        enabled = env != NULL && atoi(env) != 0;
    }
    return enabled;
}

// whether nodes[node_n] is the graph's first MUL_MAT_ID reading these expert ids
static bool ggml_cpu_first_of_ids(const struct ggml_cgraph * cgraph, int node_n) {
    const struct ggml_tensor * ids = cgraph->nodes[node_n]->src[2];
    for (int i = 0; i < node_n; i++) {
        const struct ggml_tensor * t = cgraph->nodes[i];
        if (t->op == GGML_OP_MUL_MAT_ID && t->src[2] == ids) {
            return false;
        }
    }
    return true;
}

static void ggml_cpu_prefetch_experts(const struct ggml_cgraph * cgraph, int node_n) {
#if defined(__linux__)
    const struct ggml_tensor * ids = cgraph->nodes[node_n]->src[2];
    const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
    for (int i = node_n; i < cgraph->n_nodes; i++) {
        const struct ggml_tensor * t = cgraph->nodes[i];
        if (t->op != GGML_OP_MUL_MAT_ID || t->src[2] != ids) {
            continue;
        }
        const struct ggml_tensor * w = t->src[0];
        uint64_t seen[16] = { 0 }; // experts already requested for this weight (the first 1024)
        for (int64_t r = 0; r < ids->ne[1]; r++) {
            for (int64_t k = 0; k < ids->ne[0]; k++) {
                const int32_t e = *(const int32_t *) ((const char *) ids->data + r*ids->nb[1] + k*ids->nb[0]);
                if (e < 0 || e >= w->ne[2]) {
                    continue;
                }
                if (e < 1024) {
                    if (seen[e / 64] & (1ull << (e % 64))) {
                        continue;
                    }
                    seen[e / 64] |= 1ull << (e % 64);
                }
                const uintptr_t beg = ((uintptr_t) w->data + (size_t) e * w->nb[2]) & ~(page - 1);
                const uintptr_t end = (uintptr_t) w->data + (size_t) (e + 1) * w->nb[2];
                madvise((void *) beg, end - beg, MADV_WILLNEED);
            }
        }
    }
#else
    GGML_UNUSED(cgraph);
    GGML_UNUSED(node_n);
#endif
}

static thread_ret_t ggml_graph_compute_thread(void * data) {''')

patch('ggml/src/ggml-cpu/ggml-cpu.c', '''        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        // TODO: move fused-op detection''', '''        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        // every thread takes the same branch, so the barrier is reached by all of them
        if (node->op == GGML_OP_MUL_MAT_ID && ggml_cpu_prefetch_experts_enabled() && ggml_cpu_first_of_ids(cgraph, node_n)) {
            if (state->ith == 0) {
                ggml_cpu_prefetch_experts(cgraph, node_n);
            }
            ggml_barrier(state->threadpool);
        }

        // TODO: move fused-op detection''')
print('patched', src)
