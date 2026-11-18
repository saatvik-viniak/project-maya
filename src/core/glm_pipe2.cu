// src/core/glm_pipe2.cu - STRATA_GLM_PIPE2: two conversations' pipelined speculative decodes at once over a split.
//
// decode_spec_pipe (glm_mtp.cu) runs ONE conversation's rounds through the parts: a round's anchor walks every part
// (~34 ms on 4 GPUs) and its drafts follow a stage apart, so each part is busy about half of a round.  Here two
// conversations each run their own rounds - their own verify windows, drafts, picks and commits - and their rows take
// turns on the parts' streams, filling each other's gaps.
//
// What is per conversation on a part: its sequence state arena (seq_bind: the residual, KDA states and backup, DSA
// caches, the position), its verify window (FastState::Win - win / win2 swapped), on the last part its NextN input row,
// drafts and their kept lists (head_x / mtp_tok / spec_qp and the *2 copies), and a few members (kda_bak_cur_,
// mtp_hx_pos_, mtp_sample_, mtp_pipe_q_ - the px_ stash).  pctx_bind(c) swaps them in; every enqueue for conversation
// c on a part runs with the part's lock held and c bound, so a part's stream receives each conversation's work with
// that conversation's pointers.  The kernels are the single-conversation ones: each row's arithmetic is unchanged.
//
// The crew: one thread a part, jobs tagged with their conversation (a row t, a commit, a window's begin share), each
// conversation with its own progress counters, so one's window can begin while the other's rows are still queued.
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/core/glm_ngram_draft.hpp"   // STRATA_GLM_PIPE2_COPY
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/sampler.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <thread>
#include <vector>

namespace gf = strata::kernels::glmf;

namespace strata::core {

bool Glm5Model::pipe2_on() {
    static const bool v = [] {
        const char* e = getenv("MAYA_GLM_PIPE2");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return v;
}

int Glm5Model::pipe_n() {
    static const int v = [] {
        const char* e = getenv("MAYA_GLM_PIPE_N");
        return e != nullptr ? std::max(2, std::min(kPipeN, std::atoi(e))) : 2;
    }();
    return v;
}

namespace {
// STRATA_GLM_PIPE2_COPY=1 (default off): copy drafts.  When a conversation's last STRATA_GLM_PIPE2_COPY_MIN tokens
// (default 4) occurred earlier in its history (the request's prompt, then every token it produced - the server
// registers both, glmng::set_history keyed by pipe2_copy_key), the round's drafts are the up to
// STRATA_GLM_PIPE2_COPY_MAX tokens (default 6, at most 8) that followed there, and no NextN draft step runs that round.
// The verify is unchanged (each row's own pick - greedy, or a draw from the model's distribution - must equal the
// draft to keep it), so the output is the same as without; the NextN cache entry the skipped first draft step would
// have written (position N-1) is written at the commit with the kept rows'.
// STRATA_GLM_PIPE2_PEN_SPEC (on): a request with penalties keeps speculative sampling (0: its drafts greedy, A/B)
static bool pipe2_pen_spec() {
    static const bool on = [] {
        const char* v = getenv("MAYA_GLM_PIPE2_PEN_SPEC");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

int p2_draft_cap() {
    static const int v = [] {
        const char* e = getenv("MAYA_GLM_MTP_DRAFT");
        return e ? std::max(0, std::min(Glm5Model::kWinMax - 1, std::atoi(e))) : 3;
    }();
    return v;
}
}  // namespace

// ---------------------------------------------------------------- binding
void Glm5Model::pctx_swap(int s) {
    FastState* F = fast_;
    if (s == 1) {
        std::swap(F->win, F->win2);
        std::swap(F->head_x, F->head_x2);
        std::swap(F->mtp_tok, F->mtp_tok2);
        std::swap(F->mtp_tok_h, F->mtp_tok_h2);
        std::swap(F->ev_mtp, F->ev_mtp2);
        std::swap(F->spec_qp, F->spec_qp2);
    } else {
        std::swap(F->win, F->win3);
        std::swap(F->head_x, F->head_x3);
        std::swap(F->mtp_tok, F->mtp_tok3);
        std::swap(F->mtp_tok_h, F->mtp_tok_h3);
        std::swap(F->ev_mtp, F->ev_mtp3);
        std::swap(F->spec_qp, F->spec_qp3);
    }
    std::swap(kda_bak_cur_, px_kda_bak_cur_[s]);
    std::swap(mtp_hx_pos_, px_mtp_hx_pos_[s]);
    std::swap(mtp_sample_, px_mtp_sample_[s]);
    std::swap(mtp_pipe_q_, px_mtp_pipe_q_[s]);
}

// conversation c's buffers in (from its stash slot), the bound one's out into that slot
void Glm5Model::pctx_bind(int c) {
    if (c == pctx_) return;
    const int s = pslot_[c];
    pctx_swap(s);
    pslot_[pctx_] = s;
    pslot_[c] = 0;
    seq_bind(pseq_[c]);
    pctx_ = c;
}

// the other conversations' buffers on every part (once each): a window and, on the last part, the NextN rows - each
// stash slot's swapped in, allocated as the bound one's would be, swapped back (every part has conversation 0 bound)
bool Glm5Model::pctx_alloc(int n, std::string& err) {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        std::lock_guard<std::mutex> lk(*m->pmu_);
        cudaSetDevice(m->dev_);
        FastState* F = m->fast_;
        const bool has_hx = F->head_x != nullptr, has_tok = F->mtp_tok != nullptr, has_tok_h = F->mtp_tok_h != nullptr;
        for (int c = 1; c < n; ++c) {
            const int s = m->pslot_[c];
            if (s == 0) {
                err = "glm pipe2: conversation " + std::to_string(c) + " is bound at the allocation";
                return false;
            }
            m->pctx_swap(s);
            const std::string which = "glm pipe2: conversation slot " + std::to_string(s) + "'s ";
            bool ok = m->win_alloc(kWinMax, err);
            if (ok && has_hx && F->head_x == nullptr &&
                cudaMalloc((void**) &F->head_x, (size_t) g_.n_embd * sizeof(float)) != cudaSuccess) {
                cudaGetLastError();
                F->head_x = nullptr;
                err = which + "NextN input row did not allocate";
                ok = false;
            }
            if (ok && has_tok && F->mtp_tok == nullptr &&
                cudaMalloc((void**) &F->mtp_tok, (size_t) kWinMax * sizeof(int)) != cudaSuccess) {
                cudaGetLastError();
                F->mtp_tok = nullptr;
                err = which + "draft tokens did not allocate";
                ok = false;
            }
            if (ok && has_tok_h && F->mtp_tok_h == nullptr) {
                if (cudaHostAlloc((void**) &F->mtp_tok_h, (size_t) kWinMax * sizeof(int), cudaHostAllocPortable) != cudaSuccess ||
                    cudaEventCreateWithFlags(&F->ev_mtp, cudaEventDisableTiming) != cudaSuccess) {
                    cudaGetLastError();
                    err = which + "drafts' read-back did not allocate";
                    ok = false;
                }
            }
            if (ok && F->spec_qp == nullptr && has_tok) {
                if (cudaMalloc(&F->spec_qp, (size_t) kWinMax * sizeof(strata::kernels::SpecList)) != cudaSuccess) {
                    cudaGetLastError();
                    F->spec_qp = nullptr;
                }
            }
            m->pctx_swap(s);
            if (!ok) return false;
        }
    }
    cudaSetDevice(dev_);
    return true;
}

// ---------------------------------------------------------------- the crew
struct Glm5Model::Crew2 {
    struct Job {
        int c, t;        // t >= 0: row t of conversation c's window; t < 0: its commit of -1 - t rows
        bool fresh;      // (row 0) the window's begin waits for the device
    };
    struct Worker {
        std::thread th;
        std::mutex mu;
        std::condition_variable cv;
        std::deque<Job> jobs;
        std::atomic<int> done[Glm5Model::kPipeN] = {};          // each conversation's window rows this part enqueued
        std::atomic<unsigned> commits[Glm5Model::kPipeN] = {};  // ... and commits
        bool quit = false;
    };
    std::vector<std::unique_ptr<Worker>> w;
    std::atomic<bool> failed{false};
    std::mutex emu;
    std::string err;
    bool hold[Glm5Model::kPipeN] = {};                 // the last part's jobs held while conversation c drafts
    std::vector<Job> held[Glm5Model::kPipeN];
    std::vector<unsigned> want[Glm5Model::kPipeN];     // commits each part must reach (win2_commit_wait)
    bool fresh[Glm5Model::kPipeN] = {};
    void fail(const std::string& e) {
        std::lock_guard<std::mutex> lk(emu);
        if (err.empty()) err = e;
        failed.store(true);
    }
    void push(int k, const Job& j) {
        auto& wk = w[(size_t) k];
        {
            std::lock_guard<std::mutex> lk(wk->mu);
            wk->jobs.push_back(j);
        }
        wk->cv.notify_one();
    }
};

void Glm5Model::crew2_stop() {
    if (crew2_ == nullptr) return;
    for (auto& wk : crew2_->w) {
        {
            std::lock_guard<std::mutex> lk(wk->mu);
            wk->quit = true;
        }
        wk->cv.notify_one();
    }
    for (auto& wk : crew2_->w)
        if (wk->th.joinable()) wk->th.join();
    delete crew2_;
    crew2_ = nullptr;
}

static void crew2_start(Glm5Model* head, Glm5Model::Crew2*& crew, int n_parts,
                        const std::function<Glm5Model*(int)>& part_of,
                        const std::function<int(Glm5Model*, int, int, int, bool, std::string&)>& run) {
    if (crew != nullptr) return;
    (void) head;
    crew = new Glm5Model::Crew2();
    Glm5Model::Crew2* C = crew;
    for (int k = 0; k < n_parts; ++k) C->w.emplace_back(new Glm5Model::Crew2::Worker());
    for (int k = 0; k < n_parts; ++k) {
        Glm5Model::Crew2::Worker* wk = C->w[(size_t) k].get();
        Glm5Model::Crew2::Worker* before = k > 0 ? C->w[(size_t) k - 1].get() : nullptr;
        Glm5Model* m = part_of(k);
        wk->th = std::thread([C, wk, before, m, k, run] {
            // a job may run when its conversation has no earlier job here (each conversation's own order) and, a row,
            // when the previous part has enqueued that row: the first such job runs - one conversation's row waiting
            // on the previous part no longer holds up the other's (its rows, its commit)
            // defer: the conversation whose row was not ready on the device yet (run() said 2) - the other one's
            // first job is tried before it again (else that row, back in front, would be picked again and again)
            int defer = -1;
            for (;;) {
                Glm5Model::Crew2::Job j{};
                bool got = false;
                {
                    std::unique_lock<std::mutex> lk(wk->mu);
                    wk->cv.wait(lk, [&] { return wk->quit || !wk->jobs.empty(); });
                    if (wk->jobs.empty()) return;
                    // STRATA_GLM_PIPE2_PRIO=1: of the two conversations' ready first jobs, a commit, then the lower
                    // row (a round's anchor before the other's drafts); else the one queued first
                    static const bool prio = [] {
                        const char* v = getenv("MAYA_GLM_PIPE2_PRIO");
                        return v != nullptr && std::atoi(v) != 0;
                    }();
                    for (int pass = 0; pass < 2 && !got; ++pass) {
                        bool seen[Glm5Model::kPipeN] = {};
                        auto best = wk->jobs.end();
                        for (auto it = wk->jobs.begin(); it != wk->jobs.end(); ++it) {
                            if (seen[it->c]) continue;
                            seen[it->c] = true;
                            if (pass == 0 && it->c == defer) continue;
                            const bool ready = it->t < 0 || before == nullptr ||
                                               before->done[it->c].load(std::memory_order_acquire) > it->t ||
                                               C->failed.load(std::memory_order_relaxed);
                            if (!ready) continue;
                            if (best == wk->jobs.end() || (prio && it->t < best->t)) best = it;
                            if (!prio) break;
                        }
                        if (best != wk->jobs.end()) {
                            j = *best;
                            wk->jobs.erase(best);
                            got = true;
                        }
                        if (defer < 0) break;
                    }
                }
                if (!got) {
                    std::this_thread::yield();
                    continue;
                }
                std::string e;
                if (j.t < 0) {
                    if (!C->failed.load(std::memory_order_relaxed) && run(m, k, j.c, j.t, false, e) == 0) C->fail(e);
                    wk->commits[j.c].fetch_add(1, std::memory_order_release);
                    continue;
                }
                if (!C->failed.load(std::memory_order_relaxed)) {
                    const int r = run(m, k, j.c, j.t, j.fresh, e);
                    if (r == 2) {   // (not ready yet: back in front of its conversation's jobs)
                        {
                            std::lock_guard<std::mutex> lk(wk->mu);
                            wk->jobs.push_front(j);
                        }
                        defer = j.c;
                        std::this_thread::yield();
                        continue;
                    }
                    if (r == 0) C->fail(e);
                }
                if (j.c == defer) defer = -1;
                wk->done[j.c].store(j.t + 1, std::memory_order_release);
            }
        });
    }
}

// ---------------------------------------------------------------- the window, per conversation
bool Glm5Model::win2_begin(int c, int32_t first, const strata::kernels::SamplerParams* sp, bool fresh,
                           std::string& err) {
    const bool hold_last = first < 0;
    if (hold_last) first = -1 - first;
    int64_t p0 = 0;
    {
        std::lock_guard<std::mutex> lk(*pmu_);
        pctx_bind(c);
        if (pos_ + 1 > max_ctx_) {
            err = "glm pipe2: past the context";
            return false;
        }
        p0 = pos_;
    }
    int n_parts = 0;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        ++n_parts;
        std::lock_guard<std::mutex> lk(*m->pmu_);
        m->pctx_bind(c);
        cudaSetDevice(m->dev_);
        if (!m->win_alloc(kWinMax, err)) return false;
        auto& W = m->fast_->win;
        W.p0 = p0;
        W.T = 0;
        W.per_token = false;
        W.piped = true;
        W.bank ^= 1;
        W.pick_sampled = sp != nullptr && (!sp->greedy || sp->penalty_last_n > 0);
        W.spec_n = 0;
        if (W.pick_sampled) W.pick_sp = *sp;
    }
    {
        const volatile int* from = nullptr;
        {
            std::lock_guard<std::mutex> lk(*pmu_);
            pctx_bind(c);
            auto& WH = fast_->win;
            WH.from_h[WH.bank] = 1 << 30;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            from = WH.from_h + WH.bank;
        }
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
            std::lock_guard<std::mutex> lk(*m->pmu_);
            m->pctx_bind(c);
            m->fast_->win.from = from;
        }
    }
    cudaSetDevice(dev_);
    if (crew2_ == nullptr) {
        crew2_start(this, crew2_, n_parts,
                    [this](int k) {
                        Glm5Model* m = this;
                        for (int i = 0; i < k; ++i) m = m->split_next_.get();
                        return m;
                    },
                    [this](Glm5Model* m, int k, int cc, int t, bool fr, std::string& e) -> int {
                        // a row reads the previous part's window too (its hop slot and event): both bound to the
                        // row's conversation, their locks in part order
                        Glm5Model* prev = nullptr;
                        if (t >= 0 && k > 0) {
                            prev = this;
                            for (int i = 0; i + 1 < k; ++i) prev = prev->split_next_.get();
                        }
                        // STRATA_GLM_PIPE2_LAZY (on): the last part takes a row onto its stream only once the
                        // previous part's hop for it is done - its stream then holds no row waiting on another part, so
                        // a draft step the host queues there runs behind at most the row at work
                        static const bool lazy = [] {
                            const char* v = getenv("MAYA_GLM_PIPE2_LAZY");
                            return v == nullptr || std::atoi(v) != 0;
                        }();
                        // STRATA_GLM_PIPE2_LAZY_ALL=1: every part so (a row queued behind another part's work blocks
                        // the stream for the other conversation's rows behind it)
                        static const bool lazy_all = [] {
                            const char* v = getenv("MAYA_GLM_PIPE2_LAZY_ALL");
                            return v != nullptr && std::atoi(v) != 0;
                        }();
                        if (lazy && prev != nullptr && (lazy_all || m->split_next_ == nullptr)) {
                            cudaEvent_t pe = nullptr;
                            const volatile int* from = nullptr;
                            {
                                std::lock_guard<std::mutex> l0(*prev->pmu_);
                                prev->pctx_bind(cc);
                                auto& PW = prev->fast_->win;
                                pe = PW.ev_tok[PW.bank * kPipeMax + t];
                                from = PW.from;
                            }
                            cudaSetDevice(prev->dev_);
                            // (a row past a wrong draft skips its work: no reason to hold it back)
                            if (cudaEventQuery(pe) == cudaErrorNotReady && (from == nullptr || *from > t)) return 2;
                        }
                        std::unique_lock<std::mutex> lp;
                        if (prev != nullptr) {
                            lp = std::unique_lock<std::mutex>(*prev->pmu_);
                            prev->pctx_bind(cc);
                        }
                        std::lock_guard<std::mutex> lk(*m->pmu_);
                        m->pctx_bind(cc);
                        cudaSetDevice(m->dev_);
                        if (t < 0) {
                            m->win_commit_part(-1 - t);
                            return 1;
                        }
                        if (t == 0 && !m->win_pipe_begin_part(fr, e)) return 0;
                        return win_pipe_part(k, t, e) ? 1 : 0;
                    });
    }
    Crew2* C = crew2_;
    C->hold[c] = hold_last;
    C->held[c].clear();
    C->fresh[c] = fresh;
    for (auto& wk : C->w) wk->done[c].store(0);
    return win2_push(c, first, err);
}

bool Glm5Model::win2_push(int c, int32_t token, std::string& err) {
    Crew2* C = crew2_;
    if (token < 0) {
        if (C->hold[c]) {
            C->hold[c] = false;
            for (const auto& j : C->held[c]) C->push((int) C->w.size() - 1, j);
            C->held[c].clear();
        }
        return true;
    }
    const int E = g_.n_embd;
    int t = 0;
    int64_t p = 0;
    {
        std::lock_guard<std::mutex> lk(*pmu_);
        pctx_bind(c);
        auto& W0 = fast_->win;
        t = W0.T;
        p = W0.p0 + t;
        if (t >= kPipeMax || p + 1 > max_ctx_) {
            err = "glm pipe2: past " + std::to_string(kPipeMax) + " pipelined tokens or the context";
            return false;
        }
        const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
        if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
            err = "glm pipe2: no embedding dequantizer";
            return false;
        }
        float* row = W0.emb_h + (size_t) (W0.bank * kPipeMax + t) * E;
        tt->to_float(pack_emb_src_ + (size_t) token * ggml_row_size((ggml_type) pack_emb_type_, E), row, E);
        if (const float* img = image_row(p)) std::memcpy(row, img, (size_t) E * sizeof(float));
    }
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        std::lock_guard<std::mutex> lk(*m->pmu_);
        m->pctx_bind(c);
        auto& W = m->fast_->win;
        W.pick_tok[W.bank * kPipeMax + t] = token;
        W.T = t + 1;
    }
    const Crew2::Job j{c, t, C->fresh[c]};
    for (int k = 0; k < (int) C->w.size(); ++k) {
        if (C->hold[c] && k + 1 == (int) C->w.size()) {
            C->held[c].push_back(j);
            continue;
        }
        C->push(k, j);
    }
    return true;
}

int Glm5Model::win2_pick(int c, int t, int32_t& pick, std::string& err) {
    Crew2* C = crew2_;
    if (C->failed.load()) {
        std::lock_guard<std::mutex> lk(C->emu);
        err = C->err.empty() ? "glm pipe2: a part's enqueue failed" : C->err;
        return -1;
    }
    if (C->w.back()->done[c].load(std::memory_order_acquire) <= t) return 0;
    Glm5Model* tail = this;
    while (tail->split_next_) tail = tail->split_next_.get();
    cudaEvent_t ev = nullptr;
    const int* tok = nullptr;
    {
        std::lock_guard<std::mutex> lk(*tail->pmu_);
        tail->pctx_bind(c);
        auto& WT = tail->fast_->win;
        if (t < 0 || t >= WT.T) {
            err = "glm pipe2: no token " + std::to_string(t) + " in the window";
            return -1;
        }
        const int sl = WT.bank * kPipeMax + t;
        ev = WT.ev_pick[sl];
        tok = WT.tok_h + sl;
    }
    cudaSetDevice(tail->dev_);
    const cudaError_t e = cudaEventQuery(ev);
    cudaSetDevice(dev_);
    if (e == cudaErrorNotReady) return 0;
    if (e != cudaSuccess || gf::launch_errors() > 0) {
        err = std::string("glm pipe2: ") + (e != cudaSuccess ? cudaGetErrorString(e) : "a kernel launch failed");
        return -1;
    }
    pick = *(volatile const int*) tok;
    return 1;
}

bool Glm5Model::win2_settle(int c, int T, std::string& err) {
    Crew2* C = crew2_;
    for (auto& wk : C->w)
        while (wk->done[c].load(std::memory_order_acquire) < T && !C->failed.load()) std::this_thread::yield();
    if (C->failed.load()) {
        std::lock_guard<std::mutex> lk(C->emu);
        err = C->err.empty() ? "glm pipe2: a part's enqueue failed" : C->err;
        return false;
    }
    return true;
}

void Glm5Model::win2_cancel_from(int c, int t) {
    std::lock_guard<std::mutex> lk(*pmu_);
    pctx_bind(c);
    auto& WH = fast_->win;
    if (WH.from_h == nullptr) return;
    WH.from_h[WH.bank] = t;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

bool Glm5Model::win2_commit(int c, int n_keep, int T, int64_t p0, std::string& err) {
    Crew2* C = crew2_;
    if (n_keep < 1 || n_keep > T) {
        err = "glm pipe2: commit of " + std::to_string(n_keep) + " rows of " + std::to_string(T);
        return false;
    }
    if (n_keep < T) {
        win2_commit_wait(c);
        C->want[c].clear();
        for (int k = 0; k < (int) C->w.size(); ++k) {
            C->want[c].push_back(C->w[(size_t) k]->commits[c].load(std::memory_order_acquire) + 1);
            C->push(k, Crew2::Job{c, -1 - n_keep, false});
        }
    }
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        std::lock_guard<std::mutex> lk(*m->pmu_);
        m->pctx_bind(c);
        m->pos_ = p0 + n_keep;
        m->fast_->win.T = 0;
    }
    return true;
}

void Glm5Model::win2_commit_wait(int c) {
    Crew2* C = crew2_;
    if (C == nullptr || C->want[c].empty()) return;
    for (size_t i = 0; i < C->w.size() && i < C->want[c].size(); ++i)
        while (C->w[i]->commits[c].load(std::memory_order_acquire) < C->want[c][i] && !C->failed.load())
            std::this_thread::yield();
    C->want[c].clear();
}

// ---------------------------------------------------------------- the decode
bool Glm5Model::decode_spec_pipe2(int n_conv, const int seqs[], const int32_t pending[],
                                  strata::kernels::SamplerParams* sp[], const int64_t max_new[], const int32_t force_in[],
                                  const std::function<bool(int, int)>& emit, int64_t produced[], int32_t last[],
                                  int32_t force_out[], std::string& err) {
    if (n_conv < 2 || n_conv > kPipeN) {
        err = "glm pipe2: " + std::to_string(n_conv) + " conversations";
        return false;
    }
    Glm5Model* TL = spec_tail_last();
    if (split_next_ == nullptr || TL == nullptr || TL->mtp_il_ < 0 || TL->fast_ == nullptr ||
        TL->fast_->mtpc.cap < 1) {
        err = "glm pipe2: needs a split with the NextN block";
        return false;
    }
    if (crew_ != nullptr) win_crew_stop();   // (the one-conversation crew: its threads would race ours on the streams)
    const Glm5Geometry& g = g_;
    int n_cap = std::min(p2_draft_cap(), std::min(kPipeMax - 1, 8));
    // STRATA_GLM_PIPE_N_DRAFT=<n>: at most n drafts a round while three conversations run (each row is a third of the
    // parts' time: fewer, likelier drafts)
    static const int cap3 = [] {
        const char* v = getenv("MAYA_GLM_PIPE_N_DRAFT");
        return v != nullptr ? std::max(0, std::atoi(v)) : 99;
    }();
    if (n_conv >= 3) n_cap = std::min(n_cap, cap3);
    FastState* FT = TL->fast_;
    // the conversations bound as 0 .. n-1 (0 keeps the buffers the one-conversation decode uses)
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        std::lock_guard<std::mutex> lk(*m->pmu_);
        if (m->pctx_ != 0) m->pctx_bind(0);
        for (int c = 0; c < n_conv; ++c) m->pseq_[c] = seqs[c];
        m->seq_bind(seqs[0]);
    }
    if (!pctx_alloc(n_conv, err)) return false;
    // the spec lists of conversation 0's draws (decode_spec_pipe allocates spec_qp on its first sampled request)
    if (FT->spec_qp == nullptr) {
        cudaSetDevice(TL->dev_);
        if (cudaMalloc(&FT->spec_qp, (size_t) kWinMax * sizeof(strata::kernels::SpecList)) != cudaSuccess) {
            cudaGetLastError();
            FT->spec_qp = nullptr;
        }
        cudaSetDevice(dev_);
    }
    // each conversation's own pinned staging: its draft steps' embeddings (fast_mtp's emb_staged, a row a step) and
    // its NextN cache rows' (mtp_cache_rows' emb_rows_h) - the shared ones would be rewritten by the other
    // conversation's steps before this one's copies ran (the tail's stream holds both conversations' work)
    static float* stage_h[kPipeN][2] = {};
    for (int c = 0; c < n_conv; ++c)
        for (int k = 0; k < 2; ++k)
            if (stage_h[c][k] == nullptr &&
                cudaHostAlloc((void**) &stage_h[c][k], (size_t) (kWinMax + 1) * g.n_embd * sizeof(float),
                              cudaHostAllocPortable) != cudaSuccess) {
                cudaGetLastError();
                stage_h[c][k] = nullptr;
                err = "glm pipe2: the staging rows did not allocate";
                return false;
            }
    const ggml_type_traits* ett = ggml_get_type_traits((ggml_type) pack_emb_type_);
    const size_t erow = ggml_row_size((ggml_type) pack_emb_type_, g.n_embd);
    const auto stage_tok = [&](int c, int step, int32_t tok, int64_t p) -> const float* {
        float* row = stage_h[c][0] + (size_t) step * g.n_embd;
        (void) p;   // (fast_mtp's own staging: the token's embedding row, as here)
        ett->to_float(pack_emb_src_ + (size_t) tok * erow, row, g.n_embd);
        return row;
    };
    // STRATA_GLM_PIPE2_TAU=<0..1>: a draft goes into the window only while the product of the drafts' own draw
    // probabilities q(d) stays at least this (both conversations' rows fill the parts: a doomed row costs a kept one's
    // time).  Each step's kept list comes back with its token (a copy behind it, its own event)
    static const double tau = [] {
        const char* v = getenv("MAYA_GLM_PIPE2_TAU");
        return v != nullptr ? std::max(0.0, std::min(1.0, std::atof(v))) : 0.0;
    }();
    static strata::kernels::SpecList* qh[kPipeN] = {};
    static cudaEvent_t qev[kPipeN] = {};
    if (tau > 0.0)
        for (int c = 0; c < n_conv; ++c)
            if (qh[c] == nullptr) {
                cudaSetDevice(TL->dev_);
                if (cudaHostAlloc((void**) &qh[c], (size_t) kWinMax * sizeof(strata::kernels::SpecList),
                                  cudaHostAllocPortable) != cudaSuccess ||
                    cudaEventCreateWithFlags(&qev[c], cudaEventDisableTiming) != cudaSuccess) {
                    cudaGetLastError();
                    err = "glm pipe2: the kept lists' read-back did not allocate";
                    return false;
                }
                cudaSetDevice(dev_);
            }
    const auto q_back = [&](int c, int step, bool sampled) {   // (TL locked, c bound, its device current)
        if (tau <= 0.0 || !sampled) return;
        cudaMemcpyAsync(qh[c] + step, (const strata::kernels::SpecList*) FT->spec_qp + step,
                        sizeof(strata::kernels::SpecList), cudaMemcpyDeviceToHost, FT->cs);
        cudaEventRecord(qev[c], FT->cs);
    };
    // STRATA_GLM_PIPE2_CHAIN (on where the tail chains drafts on the device - token_embd there or mapped,
    // STRATA_GLM_MTP_CHAIN_HOST): a round's draft steps queued at once (see the begin)
    static const bool chain_env = [] {
        const char* v = getenv("MAYA_GLM_PIPE2_CHAIN");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const bool chain = chain_env && FT->mtp_emb != nullptr;
    static cudaEvent_t sev[kPipeN][kWinMax] = {};
    if (chain && sev[0][0] == nullptr) {
        cudaSetDevice(TL->dev_);
        for (auto& row : sev)
            for (auto& e : row)
                if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
                    cudaGetLastError();
                    err = "glm pipe2: the chain's events did not create";
                    cudaSetDevice(dev_);
                    return false;
                }
        cudaSetDevice(dev_);
    }
    struct PC {
        int phase = 0;   // 0 begin, 1 drafts out, 2 picks, 3 done
        int32_t y = -1;
        int nd = 0, T = 0, t = 0, acc = 0, di = 0;
        int64_t N = 0;
        int32_t winv[kWinMax + 1] = {}, out[kWinMax + 1] = {};
        int32_t force = -1;
        bool pipe_sample = false;
        bool stop = false;   // the conversation ends after this round (its own stop, or the other's)
        double cum = 1.0;   // the pushed drafts' q product (STRATA_GLM_PIPE2_TAU)
        int64_t rounds = 0, nd_sum = 0, hx_miss = 0, acc_sum = 0, gated = 0;
        std::chrono::steady_clock::time_point t_begin;
        double ms_drafts = 0, ms_pick0 = 0, ms_decide = 0, ms_commit = 0, ms_round = 0;
        bool pick0 = false;
        bool copy = false;    // this round's drafts are copied (STRATA_GLM_PIPE2_COPY)
        bool hx_ok = false;   // the NextN input row at N-1 was there at the begin (copy rounds write its entry)
    } pc[kPipeN];
    const auto ms_since = [](std::chrono::steady_clock::time_point a) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    };
    // STRATA_GLM_PIPE2_COPY: each conversation's matcher over its history (see the top of the file)
    static const bool copy_on = getenv("MAYA_GLM_PIPE2_COPY") != nullptr && std::atoi(getenv("MAYA_GLM_PIPE2_COPY")) != 0;
    static const int copy_max = [] {
        const char* v = getenv("MAYA_GLM_PIPE2_COPY_MAX");
        return std::max(1, std::min(glmng::kMaxDraft, v != nullptr ? std::atoi(v) : 6));
    }();
    std::unique_ptr<glmng::NgramMatcher> cm[kPipeN];
    if (copy_on)
        for (int c = 0; c < n_conv; ++c) {
            glmng::NgramConfig cc;
            cc.on = true;
            cc.n_max = copy_max;
            const char* v = getenv("MAYA_GLM_PIPE2_COPY_MIN");
            cc.min_match = std::max(2, std::min(16, v != nullptr ? std::atoi(v) : 4));
            cc.max_match = std::max(cc.max_match, cc.min_match);
            cm[c] = std::make_unique<glmng::NgramMatcher>(cc);
            const std::vector<int32_t> h = glmng::take_history(pipe2_copy_key(this, c));
            if (!h.empty()) cm[c]->append(h.data(), h.size());
        }
    // every token a conversation produces goes through here (its matcher's history grows with it)
    const auto emit_c = [&](int c, int32_t tok) -> bool {
        if (cm[c]) cm[c]->append(tok);
        return emit(c, tok);
    };
    int64_t copy_rounds[kPipeN] = {}, copy_drafts[kPipeN] = {}, copy_acc[kPipeN] = {};
    static const double acc_keep = [] {
        const char* v = getenv("MAYA_GLM_MTP_ACC_WINDOW");
        const double w = v != nullptr ? std::max(8.0, std::atof(v)) : 128.0;
        return (w - 1.0) / w;
    }();
    static const bool pipe_sample_env = [] {
        const char* v = getenv("MAYA_GLM_SPEC_SAMPLE");
        return v != nullptr && std::atoi(v) != 0;
    }();
    // the forced token (the thinking budget, the loop guard) is the conversation's own: force_tok_ holds conversation
    // c's only while its tokens are being decided
    const auto with_force = [&](int c, auto&& fn) {
        force_tok_ = pc[c].force;
        fn();
        pc[c].force = force_tok_;
        force_tok_ = -1;
    };
    static std::vector<int32_t> pen_snap[kPipeN][2];   // each conversation's penalty history copies (rounds alternate)
    bool ok = true;
    // ---- the first tokens: each conversation's pending token fed through the token path, its next token drawn from
    //      the tail's logits (decode_spec_pipe's first token), the NextN input row left by that forward
    for (int c = 0; c < n_conv && ok; ++c) {
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
            std::lock_guard<std::mutex> lk(*m->pmu_);
            m->pctx_bind(c);
        }
        PC& P = pc[c];
        P.force = force_in[c];
        // (penalties: the verify counts each row's history - pick_tokens - and the drafts are drawn without them)
        P.pipe_sample = pipe_sample_env && !sp[c]->greedy && sp[c]->temperature > 0.0f &&
                        (sp[c]->penalty_last_n <= 0 || (pipe2_pen_spec() && sp[c]->host_history != nullptr)) &&
                        (c == 0 ? FT->spec_qp : FT->spec_qp) != nullptr;
        {
            std::lock_guard<std::mutex> lk(*TL->pmu_);
            TL->pctx_bind(c);
            TL->mtp_sample_ = P.pipe_sample ? sp[c] : nullptr;
            TL->mtp_pipe_q_ = P.pipe_sample ? FT->spec_qp : nullptr;
        }
        std::vector<float> lg;
        if (pending[c] >= 0 && !forward({pending[c]}, lg, err)) { ok = false; break; }
        int32_t y = -1;
        cudaSetDevice(TL->dev_);
        y = TL->fast_sample(*sp[c], err);
        cudaSetDevice(dev_);
        if (y < 0) { ok = false; break; }
        with_force(c, [&] { y = forced(y); });
        sp[c]->counter += 1;
        last_tok_ = y;
        produced[c] += 1;
        last[c] = y;
        {
            std::lock_guard<std::mutex> lk(*TL->pmu_);
            TL->pctx_bind(c);
            TL->mtp_hx_pos_ = pos_ - 1;
        }
        force_tok_ = -1;
        const bool more = emit_c(c, y);
        P.force = force_tok_ >= 0 ? force_tok_ : P.force;
        force_tok_ = -1;
        P.y = y;
        if (!more || produced[c] >= max_new[c] || pos_ + 1 >= max_ctx_) P.stop = true, P.phase = 3;
    }
    int drafting = -1;   // the conversation whose drafts are on their way back (ev_mtp / mtp_tok_h are shared)
    const auto all_done = [&] {
        for (int c = 0; c < n_conv; ++c)
            if (pc[c].phase != 3) return false;
        return true;
    };
    // one conversation ending ends the others after their current rounds (the caller schedules the rest)
    const auto stop_others = [&](int c) {
        for (int o = 0; o < n_conv; ++o)
            if (o != c) pc[o].stop = true;
    };
    while (ok && !all_done()) {
        bool progress = false;
        for (int c = 0; c < n_conv && ok; ++c) {
            PC& P = pc[c];
            if (P.phase == 3) continue;
            if (P.phase == 0) {
                if (P.stop) { P.phase = 3; progress = true; continue; }
                // ---- begin: the anchor through every part (the last part held for the drafts)
                {
                    std::lock_guard<std::mutex> lk(*pmu_);
                    pctx_bind(c);
                    P.N = pos_;
                }
                if (P.rounds > 0) P.ms_round += ms_since(P.t_begin);
                P.t_begin = std::chrono::steady_clock::now();
                P.pick0 = false;
                // penalties (the loop guard's short repetition penalty, a request's own): the round's picks count the
                // history as it stands now - a copy the crew's picks read (the request's own vector grows on this
                // thread as tokens come out, maybe while a cancelled row of this round is still being queued)
                const strata::kernels::SamplerParams* spr = sp[c];
                strata::kernels::SamplerParams sp_pen;
                if (sp[c]->penalty_last_n > 0 && sp[c]->host_history != nullptr) {
                    std::vector<int32_t>& hb = pen_snap[c][P.rounds & 1];
                    const std::vector<int32_t>& H = *sp[c]->host_history;
                    const size_t take = std::min(H.size(), (size_t) kPenCap);
                    hb.assign(H.end() - (std::ptrdiff_t) take, H.end());
                    sp_pen = *sp[c];
                    sp_pen.host_history = &hb;
                    spr = &sp_pen;
                }
                if (!win2_begin(c, -1 - P.y, spr, P.rounds % 8 == 0, err)) { ok = false; break; }
                P.copy = false;
                if (cm[c]) {
                    int32_t cd[glmng::kMaxDraft];
                    int k = cm[c]->propose(cd, copy_max);
                    k = (int) std::min<int64_t>(k, max_new[c] - produced[c] - 1);
                    {
                        std::lock_guard<std::mutex> lk(*TL->pmu_);
                        TL->pctx_bind(c);
                        k = (int) std::min<int64_t>(k, TL->max_ctx_ - 1 - P.N);
                        k = std::min(k, std::min(kPipeMax - 1, TL->fast_->mtpc.cap - 1));
                        if (k > 0) {
                            P.hx_ok = TL->mtp_hx_pos_ == P.N - 1;
                            TL->mtp_hx_pos_ = -1;
                            FT->win.spec_n = 0;   // (no draft distribution: each row's pick is the model's own)
                            FT->win.spec_q = FT->spec_qp;
                            FT->win.spec_d = FT->mtp_tok;
                        }
                    }
                    if (k > 0) {
                        P.winv[0] = P.y;
                        for (int i = 0; i < k && ok; ++i) {
                            P.winv[i + 1] = cd[i];
                            ok = win2_push(c, cd[i], err);
                        }
                        if (ok) ok = win2_push(c, -1, err);
                        if (!ok) break;
                        P.copy = true;
                        P.nd = k;
                        P.nd_sum += k;
                        P.di = k;
                        P.T = 1 + k;
                        P.t = 0;
                        P.acc = 0;
                        ++copy_rounds[c];
                        copy_drafts[c] += k;
                        drafting = drafting == c ? -1 : drafting;
                        P.phase = 2;
                        P.ms_drafts += ms_since(P.t_begin);
                        progress = true;
                        continue;
                    }
                }
                int n = mtp_choose(n_cap);
                n = (int) std::min<int64_t>(n, max_new[c] - produced[c] - 1);
                {
                    std::lock_guard<std::mutex> lk(*TL->pmu_);
                    TL->pctx_bind(c);
                    n = (int) std::min<int64_t>(n, TL->max_ctx_ - 1 - P.N);
                    if (TL->mtp_hx_pos_ != P.N - 1) {
                        if (n > 0) ++P.hx_miss;
                        n = 0;
                    }
                    n = std::max(n, 0);
                    P.nd = n;
                    P.nd_sum += n;
                    P.di = 0;
                    if (n > 0) {
                        // the first draft step (decode_spec_pipe's step-by-step drafts: each step's token read back
                        // and fed to the next; each draft goes into the window as it comes)
                        cudaSetDevice(TL->dev_);
                        ok = TL->fast_mtp(P.N - 1, P.y, err, 0, !P.pipe_sample, true, stage_tok(c, 0, P.y, P.N));
                        if (ok) q_back(c, 0, P.pipe_sample);
                        if (ok && chain) cudaEventRecord(sev[c][0], FT->cs);
                        // the chain: every later step queued now, each from the step before on the device (no read
                        // back between them: one wait behind the other conversation's row on this part, not one a
                        // step); each comes back with its own event, the window takes them as they land
                        for (int k = 1; ok && chain && k < n; ++k) {
                            ok = TL->fast_mtp(P.N - 1 + k, -1, err, k, !P.pipe_sample, true, nullptr);
                            if (ok) q_back(c, k, P.pipe_sample);
                            if (ok) cudaEventRecord(sev[c][k], FT->cs);
                        }
                        P.cum = 1.0;
                        cudaSetDevice(dev_);
                    } else {
                        FT->win.spec_n = 0;
                        FT->win.spec_q = FT->spec_qp;
                        FT->win.spec_d = FT->mtp_tok;
                    }
                }
                if (!ok) break;
                P.winv[0] = P.y;
                drafting = n > 0 ? c : -1;
                if (n == 0 && !win2_push(c, -1, err)) { ok = false; break; }
                P.phase = n > 0 ? 1 : 2;
                P.T = 1;
                P.t = 0;
                P.acc = 0;
                progress = true;
                continue;
            }
            if (P.phase == 1) {
                // ---- a draft back: into the window, the next step out (or the last part let go after the last)
                cudaEvent_t dev_ev = nullptr;
                const int* tok_h = nullptr;
                {
                    std::lock_guard<std::mutex> lk(*TL->pmu_);
                    TL->pctx_bind(c);
                    dev_ev = FT->ev_mtp;
                    tok_h = FT->mtp_tok_h;
                }
                if (tau > 0.0 && P.pipe_sample) dev_ev = qev[c];   // (behind the token's copy: both are back)
                if (chain) dev_ev = sev[c][P.di];                  // (the chain: this step's own, behind both copies)
                cudaSetDevice(TL->dev_);
                const cudaError_t e = cudaEventQuery(dev_ev);
                cudaSetDevice(dev_);
                if (e == cudaErrorNotReady) continue;
                if (e != cudaSuccess) { err = std::string("glm pipe2 drafts: ") + cudaGetErrorString(e); ok = false; break; }
                if (!TL->fast_->route_ok(err)) { ok = false; break; }
                const int32_t tok = ((volatile const int*) tok_h)[P.di];
                bool gate = false;
                if (tau > 0.0 && P.pipe_sample) {
                    const strata::kernels::SpecList& L = qh[c][P.di];
                    double q = 0.0;
                    for (int i = 0; i < L.n && i < 64; ++i)
                        if (L.ids[i] == tok) {
                            q = L.p[i];
                            break;
                        }
                    P.cum *= q;
                    gate = P.cum < tau;
                }
                progress = true;
                if (gate) {
                    // not worth a row: the window ends at the drafts so far (the step's work is wasted, the rows are not)
                    ++P.gated;
                    P.nd = P.di;
                } else {
                    P.winv[P.di + 1] = tok;
                    ++P.di;
                    if (!win2_push(c, tok, err)) { ok = false; break; }
                }
                if (P.di < P.nd && chain) continue;   // (already queued)
                if (P.di < P.nd) {
                    std::lock_guard<std::mutex> lk(*TL->pmu_);
                    TL->pctx_bind(c);
                    cudaSetDevice(TL->dev_);
                    ok = TL->fast_mtp(P.N - 1 + P.di, tok, err, P.di, !P.pipe_sample, true,
                                      stage_tok(c, P.di, tok, P.N + P.di));
                    if (ok) q_back(c, P.di, P.pipe_sample);
                    cudaSetDevice(dev_);
                    if (!ok) break;
                    continue;
                }
                drafting = -1;
                {
                    std::lock_guard<std::mutex> lk(*TL->pmu_);
                    TL->pctx_bind(c);
                    TL->mtp_hx_pos_ = -1;
                    FT->win.spec_n = P.pipe_sample ? P.nd : 0;
                    FT->win.spec_q = FT->spec_qp;
                    FT->win.spec_d = FT->mtp_tok;
                }
                if (!win2_push(c, -1, err)) { ok = false; break; }
                P.T = 1 + P.nd;
                P.phase = 2;
                P.ms_drafts += ms_since(P.t_begin);
                continue;
            }
            if (P.phase == 2) {
                // ---- the picks as they come: the first one a draft does not repeat skips the rows past it
                bool decided = false;
                while (!decided) {
                    int32_t pick = -1;
                    const int r = win2_pick(c, P.t, pick, err);
                    if (r < 0) { ok = false; break; }
                    if (r == 0) break;
                    progress = true;
                    if (P.t == 0 && !P.pick0) {
                        P.pick0 = true;
                        P.ms_pick0 += ms_since(P.t_begin);
                    }
                    P.out[P.t] = pick;
                    if (P.t + 1 >= P.T) { decided = true; break; }
                    const int32_t want = (P.t == 0 && P.force >= 0) ? P.force : P.winv[P.t + 1];
                    if (!P.copy) {
                        mtp_hit_[P.t + 1] *= acc_keep;
                        mtp_reach_[P.t + 1] = mtp_reach_[P.t + 1] * acc_keep + 1.0;
                    }
                    if (pick != want) {
                        win2_cancel_from(c, P.t + 1);
                        decided = true;
                        break;
                    }
                    if (!P.copy) mtp_hit_[P.t + 1] += 1.0;
                    ++P.acc;
                    ++P.t;
                }
                if (!ok) break;
                if (!decided) continue;
                P.ms_decide += ms_since(P.t_begin);
                if (!win2_settle(c, P.T, err)) { ok = false; break; }
                spec_steps_ += (uint64_t) P.nd;
                spec_hits_ += (uint64_t) P.acc;
                P.acc_sum += P.acc;
                if (P.copy) copy_acc[c] += P.acc;
                ++P.rounds;
                // ---- the kept tokens out, in order
                bool more = true;
                int kept = 0;
                force_tok_ = P.force;
                for (; kept < P.acc; ++kept) {
                    sp[c]->counter += 1;
                    ++produced[c];
                    const int32_t tk = forced(P.out[kept]);
                    last[c] = tk;
                    const int32_t f0 = force_tok_;
                    force_tok_ = -1;
                    more = emit_c(c, tk) && produced[c] < max_new[c];
                    const int32_t fnew = force_tok_;
                    force_tok_ = fnew >= 0 ? fnew : f0;
                    if (!more || force_tok_ >= 0) {
                        ++kept;
                        break;
                    }
                }
                const int keep = 1 + kept;
                const int32_t y2 = forced(P.out[keep - 1]);
                P.force = force_tok_;
                force_tok_ = -1;
                // ---- the commit (each part's restore + replay on its crew thread), then the NextN cache
                if (!win2_commit(c, keep, P.T, P.N, err)) { ok = false; break; }
                win2_commit_wait(c);
                {
                    std::lock_guard<std::mutex> lk(*TL->pmu_);
                    TL->pctx_bind(c);
                    const size_t sl0 = (size_t) (FT->win.bank * kPipeMax);
                    const float* rows_hx = FT->win.head_x + sl0 * (size_t) g.n_embd;
                    // a copy round ran no first draft step: its NextN cache entry (position N-1: the row before the
                    // round, the anchor's token) from here, before head_x moves on
                    if (P.copy && P.hx_ok && !TL->mtp_cache_rows(P.N - 1, 1, FT->head_x, P.winv, err, false, stage_h[c][0]))
                        ok = false;
                    if (ok && keep > 1 && !TL->mtp_cache_rows(P.N, keep - 1, rows_hx, P.winv + 1, err, false, stage_h[c][1]))
                        ok = false;
                    if (ok) {
                        cudaSetDevice(TL->dev_);
                        cudaMemcpyAsync(FT->head_x, rows_hx + (size_t) (keep - 1) * g.n_embd,
                                        (size_t) g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, FT->cs);
                        TL->mtp_hx_pos_ = P.N + keep - 1;
                        cudaSetDevice(dev_);
                    }
                }
                if (!ok) break;
                P.ms_commit += ms_since(P.t_begin);
                ++mtp_rounds_;
                if (!more) {
                    // A stop inside the kept tokens (a stop token, the request's limit) must stop the other
                    // conversation after its current round too: the caller finishes this one immediately and
                    // reschedules the other through the normal path.  The anchor-stop path below does the same.
                    P.stop = true;
                    P.phase = 3;
                    stop_others(c);
                    progress = true;
                    continue;
                }
                // the anchor: its counter, written now; the next round begins with it
                sp[c]->counter += 1;
                ++produced[c];
                last[c] = y2;
                force_tok_ = P.force;
                const int32_t f0 = force_tok_;
                force_tok_ = -1;
                const bool m2 = emit_c(c, y2);
                P.force = force_tok_ >= 0 ? force_tok_ : f0;
                force_tok_ = -1;
                int64_t pos_c = 0;
                {
                    std::lock_guard<std::mutex> lk(*pmu_);
                    pctx_bind(c);
                    pos_c = pos_;
                    last_tok_ = y2;
                }
                P.y = y2;
                if (!m2 || produced[c] >= max_new[c] || pos_c + 1 >= max_ctx_) P.stop = true;
                // one conversation ending stops the other after its round (the caller schedules the rest)
                if (P.stop) {
                    P.phase = 3;
                    stop_others(c);
                } else {
                    P.phase = 0;
                }
                progress = true;
                continue;
            }
        }
        if (!progress) std::this_thread::yield();
    }
    // ---- the drain: every part's queued work done; conversation 0 bound again (its buffers are the one-conversation
    //      decode's), the samplers' hooks off
    {
        if (crew2_ != nullptr) {
            for (int c = 0; c < n_conv; ++c) win2_commit_wait(c);
        }
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
            cudaSetDevice(m->dev_);
            std::string de;
            if (!glmfast::cuda_ok(cudaStreamSynchronize(m->fast_->cs), "glm pipe2 drain", de) && ok) {
                err = de;
                ok = false;
            }
        }
        if (ok && !TL->fast_->route_ok(err)) ok = false;
        for (int c = n_conv - 1; c >= 0; --c) {
            std::lock_guard<std::mutex> lk(*TL->pmu_);
            TL->pctx_bind(c);
            TL->mtp_sample_ = nullptr;
            TL->mtp_pipe_q_ = nullptr;
            TL->mtp_hx_pos_ = c == 0 ? TL->mtp_hx_pos_ : -1;
        }
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
            std::lock_guard<std::mutex> lk(*m->pmu_);
            m->pctx_bind(0);
        }
        cudaSetDevice(dev_);
    }
    for (int c = 0; c < n_conv; ++c) force_out[c] = pc[c].force;
    // (two conversations: the line as before - "glm pipe2: rounds a + b, ..."; three: "glm pipe3: rounds a + b + c, ...")
    const auto cat = [&](auto&& f) {
        std::string o;
        for (int c = 0; c < n_conv; ++c) o += (c ? " + " : "") + std::to_string((long long) f(c));
        return o;
    };
    std::fprintf(stderr, "glm pipe%d: rounds %s, tokens %s, drafts %s, accepted %s, no NextN row %s, gated %s%s\n", n_conv,
                 cat([&](int c) { return pc[c].rounds; }).c_str(), cat([&](int c) { return produced[c]; }).c_str(),
                 cat([&](int c) { return pc[c].nd_sum; }).c_str(), cat([&](int c) { return pc[c].acc_sum; }).c_str(),
                 cat([&](int c) { return pc[c].hx_miss; }).c_str(), cat([&](int c) { return pc[c].gated; }).c_str(),
                 ok ? "" : " (failed)");
    if (copy_on)
        std::fprintf(stderr, "glm pipe%d: copy rounds %s, copied drafts %s, kept %s\n", n_conv,
                     cat([&](int c) { return copy_rounds[c]; }).c_str(), cat([&](int c) { return copy_drafts[c]; }).c_str(),
                     cat([&](int c) { return copy_acc[c]; }).c_str());
    for (int c = 0; c < n_conv; ++c)
        if (pc[c].rounds > 1)
            std::fprintf(stderr, "glm pipe%d:   conversation %d, ms after a round's begin: drafts in %.1f, first pick %.1f, "
                                 "decided %.1f, committed %.1f | a round %.1f\n", n_conv, c,
                         pc[c].ms_drafts / pc[c].rounds, pc[c].ms_pick0 / pc[c].rounds, pc[c].ms_decide / pc[c].rounds,
                         pc[c].ms_commit / pc[c].rounds, pc[c].ms_round / (pc[c].rounds - 1));
    if (ok && gf::launch_errors() > 0) {
        err = "glm pipe2: a kernel launch failed (see stderr)";
        ok = false;
    }
    return ok;
}

}  // namespace strata::core
