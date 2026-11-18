// src/core/glm_spec_tl.hpp - STRATA_GLM_SPEC_TL=<file>: a raw event timeline of the pipelined speculative decode
// (decode_spec_pipe), for offline analysis (campaign tools: spec_tl.py).  One line an event, ms since the first one:
//   R <round> <p0>                  the main thread starts a round (p0: the anchor's position)
//   D <i>                           draft i pushed into the window
//   P <t> <pick>                    the main thread has row t's pick
//   S / C / B                       settle done / commit queued / the next anchor's begin queued
//   E <part> <p> <0|1>              part's crew thread starts / ends enqueueing position p's row
//   G <part> <p> <0|1>              part's stream reaches the start / end of position p's row (a host callback)
//   L <dev> <il> <ne> <t0> <t1>     a CPU-lane call (the service thread of CUDA<dev>): its start and end
// Debug only: the stream callbacks cost ~10-30 us each.
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace strata::core::spec_tl {

inline FILE* file() {
    static FILE* f = [] {
        const char* p = std::getenv("MAYA_GLM_SPEC_TL");
        FILE* h = p != nullptr && p[0] ? std::fopen(p, "w") : nullptr;
        if (h != nullptr) std::setvbuf(h, nullptr, _IOFBF, 1 << 20);
        return h;
    }();
    return f;
}
inline bool on() {
    static const bool v = file() != nullptr;
    return v;
}
inline double now_ms() {
    static const auto t0 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

struct Cb {
    char code;
    int a, b, c;
};
inline Cb* cb_slot() {
    static Cb ring[8192];
    static std::atomic<unsigned> at{0};
    return &ring[at.fetch_add(1, std::memory_order_relaxed) % 8192u];
}
inline void CUDART_CB cb_fn(void* p) {
    const Cb* c = (const Cb*) p;
    std::fprintf(file(), "%c %d %d %d %.3f\n", c->code, c->a, c->b, c->c, now_ms());
}
// a stream point: line "<code> a b c <ms>" when the stream reaches it
inline void at_stream(cudaStream_t s, char code, int a, int b, int c) {
    if (!on()) return;
    Cb* x = cb_slot();
    *x = Cb{code, a, b, c};
    cudaLaunchHostFunc(s, cb_fn, x);
}
inline void host(char code, int a = 0, int b = 0, int c = 0) {
    if (!on()) return;
    std::fprintf(file(), "%c %d %d %d %.3f\n", code, a, b, c, now_ms());
}
inline void lane(int dev, int il, int ne, double t0, double t1) {
    if (!on()) return;
    std::fprintf(file(), "L %d %d %d %.3f %.3f\n", dev, il, ne, t0, t1);
}
inline void flush() {
    if (on()) std::fflush(file());
}

}  // namespace strata::core::spec_tl
