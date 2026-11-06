#!/usr/bin/env python3
# Batched RPC fetches for a split's inputs, opt-in with GGML_RPC_BATCH_FETCH=1. Off, both libraries take the path they always took.
# Wire format is unchanged: the helper sees the same GET_TENSOR requests, they just arrive together. No helper rebuild.
# usage: python3 -I batch-patch.py <source dir>
import sys
src = sys.argv[1]

def patch(path, old, new):
    full = f'{src}/{path}'
    text = open(full).read()
    assert text.count(old) == 1, f'{path}: {old[:60]!r} found {text.count(old)} times'
    open(full, 'w').write(text.replace(old, new))

RPC = 'ggml/src/ggml-rpc/ggml-rpc.cpp'
BACKEND = 'ggml/src/ggml-backend.cpp'

patch(RPC, '#include <cstring>\n', '#include <cstdlib>\n#include <cstring>\n')

patch(RPC, '// RPC client-side implementation\n', '''// One GET_TENSOR of a batch: the request, and where its reply goes.
struct rpc_fetch {
    rpc_msg_get_tensor_req   request;
    void                   * output;
    size_t                   output_size;
};

// All the requests in one write, then each reply in order. Replies come back in request order on one stream.
static bool send_rpc_fetches(socket_ptr sock, const std::vector<rpc_fetch> & fetches) {
    std::vector<uint8_t> wire;
    for (const auto & fetch : fetches) {
        const uint8_t  cmd_byte   = RPC_CMD_GET_TENSOR;
        const uint64_t input_size = sizeof(fetch.request);
        const size_t   at         = wire.size();
        wire.resize(at + sizeof(cmd_byte) + sizeof(input_size) + input_size);
        memcpy(wire.data() + at, &cmd_byte, sizeof(cmd_byte));
        memcpy(wire.data() + at + sizeof(cmd_byte), &input_size, sizeof(input_size));
        memcpy(wire.data() + at + sizeof(cmd_byte) + sizeof(input_size), &fetch.request, input_size);
    }
    if (!sock->send_data(wire.data(), wire.size()) || !sock->flush()) {
        return false;
    }
    for (const auto & fetch : fetches) {
        uint64_t out_size;
        if (!sock->recv_data(&out_size, sizeof(out_size)) || out_size != fetch.output_size) {
            return false;
        }
        if (!sock->recv_data(fetch.output, fetch.output_size)) {
            return false;
        }
    }
    return true;
}

// RPC client-side implementation
''')

patch(RPC, '''    void send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size);

    ggml_backend_event_t event_new''', '''    void send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size);
    void send_fetches(std::vector<rpc_fetch> fetches);

    ggml_backend_event_t event_new''')

patch(RPC, '''        size_t                        output_size;
        std::promise<void>            completion;
''', '''        size_t                        output_size;
        std::vector<rpc_fetch>        fetches;
        std::promise<void>            completion;
''')

patch(RPC, '''ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
''', '''// Goes through the queue like every other command, so it stays in order with queued writes and graph runs.
void rpc_dispatcher::send_fetches(std::vector<rpc_fetch> fetches) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = RPC_CMD_GET_TENSOR;
    msg->input = nullptr;
    msg->input_size = fetches.size() * sizeof(rpc_msg_get_tensor_req);
    msg->output = nullptr;
    msg->output_size = 0;
    for (const auto & fetch : fetches) {
        msg->output_size += fetch.output_size;
    }
    msg->fetches = std::move(fetches);
    GGML_ASSERT(queue.push(msg));
    auto future = msg->completion.get_future();
    future.wait();
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
''')

patch(RPC, '''            if (msg_ptr->output) {
                bool status = send_rpc_cmd(sock, msg_ptr->cmd, msg_ptr->input.get(), msg_ptr->input_size, msg_ptr->output, msg_ptr->output_size);
''', '''            if (!msg_ptr->fetches.empty()) {
                bool status = send_rpc_fetches(sock, msg_ptr->fetches);
                RPC_STATUS_ASSERT(status);
            } else if (msg_ptr->output) {
                bool status = send_rpc_cmd(sock, msg_ptr->cmd, msg_ptr->input.get(), msg_ptr->input_size, msg_ptr->output, msg_ptr->output_size);
''')

patch(RPC, '''static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
''', '''// Several tensors in one trip. Tensors that are views of the same bytes (a reshape and its source) ask for them once.
static void ggml_backend_rpc_buffer_get_tensor_n(ggml_backend_buffer_t buffer, const ggml_tensor * const * tensors, void * const * data,
        const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    std::vector<rpc_fetch> fetches;
    std::vector<std::pair<size_t, size_t>> repeats; // tensor, and the fetch that already has its bytes
    fetches.reserve(n_tensors);
    for (size_t i = 0; i < n_tensors; i++) {
        const rpc_tensor tensor = serialize_tensor(tensors[i]);
        size_t j = 0;
        for (; j < fetches.size(); j++) {
            const rpc_msg_get_tensor_req & seen = fetches[j].request;
            if (seen.tensor.data == tensor.data && seen.offset == offsets[i] && seen.size == sizes[i]) {
                break;
            }
        }
        if (j < fetches.size()) {
            repeats.emplace_back(i, j);
            continue;
        }
        rpc_fetch fetch;
        fetch.request.tensor = tensor;
        fetch.request.offset = offsets[i];
        fetch.request.size = sizes[i];
        fetch.output = data[i];
        fetch.output_size = sizes[i];
        fetches.push_back(fetch);
    }
    LOG_DBG2("[%s] tensors: %zu, fetches: %zu\\n", __func__, n_tensors, fetches.size());
    // A trip's requests stay well under a socket buffer, so the helper can't block writing a reply while we are still writing requests.
    constexpr size_t max_per_trip = 32;
    for (size_t i = 0; i < fetches.size(); i += max_per_trip) {
        ctx->dispatcher->send_fetches(std::vector<rpc_fetch>(fetches.begin() + i, fetches.begin() + std::min(i + max_per_trip, fetches.size())));
    }
    for (const auto & repeat : repeats) {
        memcpy(data[repeat.first], fetches[repeat.second].output, sizes[repeat.first]);
    }
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
''')

patch(RPC, '''    /* .set_tensor_n    = */ NULL,
    /* .get_tensor_n    = */ NULL,
    /* .cpy_tensor_n    = */ NULL,
};

static const char * ggml_backend_rpc_buffer_type_name''', '''    /* .set_tensor_n    = */ NULL,
    /* .get_tensor_n    = */ std::getenv("GGML_RPC_BATCH_FETCH") != nullptr ? ggml_backend_rpc_buffer_get_tensor_n : nullptr,
    /* .cpy_tensor_n    = */ NULL,
};

static const char * ggml_backend_rpc_buffer_type_name''')

patch(BACKEND, '''static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
''', '''// GGML_RPC_BATCH_FETCH: fetch a split's inputs from one RPC buffer type together, one round trip instead of one each.
// Only the plain synchronous copy into host memory is batched: an input needing an event, a copy callback or an
// async copy stays on the path in ggml_backend_sched_copy_input. Marks what it fetched in `fetched`.
static void ggml_backend_sched_fetch_inputs(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split, std::vector<char> & fetched) {
    static const bool enabled = getenv("GGML_RPC_BATCH_FETCH") != NULL;
    if (!enabled || split->n_inputs < 2) {
        return;
    }
    const int split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    if (sched->events[split_backend_id][sched->cur_copy] != NULL || split_backend->iface.cpy_tensor_async != NULL) {
        return;
    }

    ggml_backend_buffer_t buf = NULL;
    ggml_backend_t input_backend = NULL;
    std::vector<const struct ggml_tensor *> srcs;
    std::vector<void *> dsts;
    std::vector<size_t> sizes;
    std::vector<int> ids;

    for (int input_id = 0; input_id < split->n_inputs; input_id++) {
        struct ggml_tensor * input = split->inputs[input_id];
        if ((input->flags & GGML_TENSOR_FLAG_INPUT) || ggml_backend_sched_is_host_weight(input) || input->data == NULL) {
            continue;
        }
        ggml_backend_buffer_t src_buf = input->view_src ? input->view_src->buffer : input->buffer;
        if (src_buf == NULL || src_buf->iface.get_tensor_n == NULL || strncmp(ggml_backend_buft_name(src_buf->buft), "RPC", 3) != 0) {
            continue;
        }
        if (buf != NULL && src_buf->buft != buf->buft) {
            continue;
        }
        struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
        if (input_cpy == input || input_cpy->data == NULL || input_cpy->buffer == NULL || !ggml_backend_buffer_is_host(input_cpy->buffer) || ggml_nbytes(input) == 0) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, input);
        if (input_backend != NULL && backend != input_backend) {
            continue;
        }
        GGML_ASSERT(ggml_are_same_layout(input, input_cpy) && "cannot copy tensors with different layouts");
        buf = src_buf;
        input_backend = backend;
        srcs.push_back(input);
        dsts.push_back(input_cpy->data);
        sizes.push_back(ggml_nbytes(input));
        ids.push_back(input_id);
    }
    if (srcs.size() < 2) {
        return;
    }

    ggml_backend_synchronize(split_backend);
    ggml_backend_synchronize(input_backend);
    const std::vector<size_t> offsets(srcs.size(), 0);
    buf->iface.get_tensor_n(buf, srcs.data(), dsts.data(), offsets.data(), sizes.data(), srcs.size());
    fetched.resize(split->n_inputs, 0);
    for (int input_id : ids) {
        fetched[input_id] = 1;
    }
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
''')

patch(BACKEND, '''        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            if (!ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id]);
''', '''        std::vector<char> fetched;
        ggml_backend_sched_fetch_inputs(sched, split, fetched);
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            if ((fetched.empty() || !fetched[input_id]) && !ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id]);
''')
print('patched')
