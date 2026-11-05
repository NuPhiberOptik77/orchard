// CPU-time sampler for one program's main thread, loaded with LD_PRELOAD. Needs no rebuild of the program.
// SAMP_OUT names the output; SAMP_NAME is the program to sample (default llama-completion); anything else runs untouched.
// The timer counts this thread's CPU time, so it fires only while the thread is on a CPU and almost never interrupts a blocking call.
#define _GNU_SOURCE
#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#define DEPTH 48
#define PERIOD_NS 10000000
struct rec { uint64_t t, n, pc[DEPTH]; };

static struct rec * buf;
static size_t cap, used;
static char maps_path[4096];
static uint64_t maps_at;

// open, read, write and close only: all safe in a signal handler.
static void copy_maps(void) {
    int in = open("/proc/self/maps", O_RDONLY), out = open(maps_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    static char block[65536];
    ssize_t got;
    while (in >= 0 && out >= 0 && (got = read(in, block, sizeof(block))) > 0) {
        if (write(out, block, got) != got) {
            break;
        }
    }
    if (in >= 0) close(in);
    if (out >= 0) close(out);
}

static void on_prof(int sig, siginfo_t * si, void * uc_) {
    (void)sig; (void)si;
    const int saved = errno;
    if (used < cap) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        const uint64_t now = (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
        const uint64_t rip = ((ucontext_t *)uc_)->uc_mcontext.gregs[REG_RIP];
        void * stack[DEPTH + 8];
        const int depth = backtrace(stack, DEPTH + 8);
        int from = 0;
        while (from < depth && (uint64_t)stack[from] != rip) from++;
        struct rec * r = &buf[used];
        r->pc[0] = rip;
        r->n = 1;
        for (int i = from + 1; i < depth && r->n < DEPTH; i++) {
            r->pc[r->n++] = (uint64_t)stack[i];
        }
        r->t = now;
        used++;
        if (now - maps_at > 60000000000ull) {
            maps_at = now;
            copy_maps();
        }
    }
    errno = saved;
}

__attribute__((constructor)) static void samp_start(void) {
    const char * out = getenv("SAMP_OUT");
    const char * name = getenv("SAMP_NAME");
    if (out == NULL || strcmp(program_invocation_short_name, name ? name : "llama-completion") != 0) {
        return;
    }
    const size_t bytes = 64u << 20;
    const int fd = open(out, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0 || ftruncate(fd, bytes) != 0) {
        return;
    }
    buf = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (buf == MAP_FAILED) {
        return;
    }
    cap = bytes / sizeof(struct rec);
    snprintf(maps_path, sizeof(maps_path), "%s.maps", out);

    void * first[4];
    backtrace(first, 4); // loads libgcc_s now, not inside the handler

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = on_prof;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGPROF, &sa, NULL);

    struct sigevent ev;
    memset(&ev, 0, sizeof(ev));
    ev.sigev_notify = SIGEV_THREAD_ID;
    ev.sigev_signo = SIGPROF;
    ev._sigev_un._tid = syscall(SYS_gettid);
    timer_t timer;
    if (timer_create(CLOCK_THREAD_CPUTIME_ID, &ev, &timer) != 0) {
        return;
    }
    struct itimerspec every = { { 0, PERIOD_NS }, { 0, PERIOD_NS } };
    timer_settime(timer, 0, &every, NULL);
}

__attribute__((destructor)) static void samp_stop(void) {
    if (buf != NULL) {
        copy_maps();
    }
}
