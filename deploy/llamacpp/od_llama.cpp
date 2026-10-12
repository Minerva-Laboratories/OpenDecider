// OpenDecider row features from llama.cpp: the backbone of the llama.cpp serving backend
// (src/opendecider/llamacpp.py; feasibility check in scripts/llamacpp_probe, RESULTS 20.2).
//
// One context with a unified KV cache. The state runs once in sequence 0 (kept until a different state arrives).
// Option rows [question tokens][option tokens] run in batches: every row gets its own sequence, forked from the state
// (llama_memory_seq_cp: with a unified cache the state's KV cells are shared, the gated delta net layers' recurrent
// state is copied), all rows of a batch padded to one length so that llama.cpp keeps them in one ubatch, laid out
// row after row. Read through the eval callback: the residual stream after blocks 8, 16, 24 (l_out-7/15/23) and the
// final-norm output (h_nextn), every row token; and the log-probability of every option token given the previous
// position (logits requested only there). Padding tokens come after a row's real tokens, so they change nothing
// before them (causal).
//
// C API (ctypes):
//   void * odl_init(model.gguf, n_ctx, n_seq_max, n_ubatch, n_gpu_layers, n_threads)   NULL on failure
//   int    odl_n_embd(h)
//   int    odl_state(h, ids, n)                     0 = decoded, 1 = same state as before (kept), < 0 = error
//   int    odl_rows(h, n_rows, tokens, row_len, row_nq, feats, lp)                       0 = ok, < 0 = error
//            tokens: rows concatenated; feats: sum(row_len) x 4 n_embd fp16, [L8 | L16 | L24 | final] per token;
//            lp: sum(row_len - row_nq) float32, the option tokens' log-probabilities
//   int    odl_rows_qcache(h, n_q, q_tokens, q_len, n_opt, o_tokens, o_len, q_feats, o_feats, lp)
//            the same features with the question cache (below); what the backend uses
//   void   odl_free(h)
//
// Build: deploy/llamacpp/build.sh
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static const char * WANT[4] = {"l_out-7", "l_out-15", "l_out-23", "h_nextn"};

struct Capture {
    bool on = false;
    std::vector<float> buf[4];
    int calls[4] = {0, 0, 0, 0};
};

struct OdLlama {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_memory_t mem = nullptr;
    int n_embd = 0, n_vocab = 0, n_seq_max = 0, n_ubatch = 0, n_threads = 4;
    std::vector<int32_t> state;          // tokens in sequence 0
    Capture cap;
};

static int want_idx(const char * name) {
    for (int i = 0; i < 4; i++) if (strcmp(name, WANT[i]) == 0) return i;
    return -1;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    auto * c = (Capture *) ud;
    const int k = want_idx(t->name);
    if (ask) return c->on && k >= 0;
    if (!c->on || k < 0) return true;
    if (t->type != GGML_TYPE_F32) { fprintf(stderr, "od_llama: unexpected type for %s\n", t->name); return false; }
    const int64_t n = ggml_nelements(t);
    c->buf[k].resize(n);
    ggml_backend_tensor_get(t, c->buf[k].data(), 0, n * sizeof(float));
    c->calls[k]++;
    return true;
}

extern "C" {

static void quiet_log(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN && level != GGML_LOG_LEVEL_CONT) fputs(text, stderr);
}

void * odl_init(const char * path, int n_ctx, int n_seq_max, int n_ubatch, int ngl, int n_threads) {
    if (!getenv("OD_LLAMA_VERBOSE")) llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    auto * h = new OdLlama();
    h->model = llama_model_load_from_file(path, mp);
    if (!h->model) { delete h; return nullptr; }
    h->n_embd = llama_model_n_embd(h->model);
    h->n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(h->model));
    h->n_seq_max = n_seq_max; h->n_ubatch = n_ubatch; h->n_threads = std::max(1, n_threads);
    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = std::max(n_ctx, n_ubatch); cp.n_ubatch = n_ubatch; cp.n_seq_max = n_seq_max;
    cp.kv_unified = true;
    cp.cb_eval = cb_eval; cp.cb_eval_user_data = &h->cap;
    cp.no_perf = true;
    h->ctx = llama_init_from_model(h->model, cp);
    if (!h->ctx) { llama_model_free(h->model); delete h; return nullptr; }
    h->mem = llama_get_memory(h->ctx);
    return h;
}

int odl_n_embd(void * hp) { return ((OdLlama *) hp)->n_embd; }

int odl_state(void * hp, const int32_t * ids, int n) {
    auto * h = (OdLlama *) hp;
    if (n <= 0) return -1;
    if ((int) h->state.size() == n && std::equal(ids, ids + n, h->state.begin())) return 1;
    llama_memory_clear(h->mem, true);
    h->state.clear();
    h->cap.on = false;
    for (int a = 0; a < n; a += h->n_ubatch) {          // chunks of n_ubatch tokens (one ubatch each)
        const int m = std::min(h->n_ubatch, n - a);
        llama_batch b = llama_batch_init(m, 0, 1);
        for (int i = 0; i < m; i++) {
            b.token[i] = ids[a + i]; b.pos[i] = a + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
            b.logits[i] = (a + i == n - 1);
        }
        b.n_tokens = m;
        const int rc = llama_decode(h->ctx, b);
        llama_batch_free(b);
        if (rc != 0) { llama_memory_clear(h->mem, true); return -2; }
    }
    h->state.assign(ids, ids + n);
    return 0;
}

static void log_softmax_at(const float * lg, int n_vocab, int tok, float * out) {
    float m = -INFINITY;
    for (int v = 0; v < n_vocab; v++) m = lg[v] > m ? lg[v] : m;
    double s = 0;
    for (int v = 0; v < n_vocab; v++) s += std::exp((double) (lg[v] - m));
    *out = (float) (lg[tok] - m - std::log(s));
}

int odl_rows(void * hp, int n_rows, const int32_t * tokens, const int32_t * row_len, const int32_t * row_nq,
             uint16_t * feats, float * lp) {
    auto * h = (OdLlama *) hp;
    const int ns = (int) h->state.size(), D = h->n_embd;
    if (ns == 0) return -1;
    std::vector<size_t> tok_off(n_rows + 1, 0), lp_off(n_rows + 1, 0);
    for (int r = 0; r < n_rows; r++) {
        if (row_nq[r] < 1 || row_len[r] <= row_nq[r]) return -3;
        tok_off[r + 1] = tok_off[r] + row_len[r];
        lp_off[r + 1] = lp_off[r] + (row_len[r] - row_nq[r]);
    }
    int r0 = 0;
    while (r0 < n_rows) {
        // as many rows as fit: at most n_seq_max - 1 sequences and (rows x padded length) <= n_ubatch tokens
        int r1 = r0, L = 0;
        while (r1 < n_rows && r1 - r0 < h->n_seq_max - 1) {
            const int L2 = std::max(L, (int) row_len[r1]);
            if ((r1 - r0 + 1) * L2 > h->n_ubatch && r1 > r0) break;
            L = L2; r1++;
        }
        const int B = r1 - r0;
        if (B * L > h->n_ubatch) return -4;                 // one row longer than n_ubatch
        for (int s = 1; s <= B; s++) {
            llama_memory_seq_rm(h->mem, s, -1, -1);
            llama_memory_seq_cp(h->mem, 0, s, -1, -1);
        }
        llama_batch b = llama_batch_init(B * L, 0, 1);
        std::vector<int> out_row, out_t;                     // batch index -> (row, token position in row)
        for (int k = 0; k < B; k++) {
            const int r = r0 + k, n = row_len[r], nq = row_nq[r];
            const int32_t * t = tokens + tok_off[r];
            for (int i = 0; i < L; i++) {
                const int j = k * L + i;
                b.token[j] = i < n ? t[i] : t[n - 1];        // pad after the row's real tokens
                b.pos[j] = ns + i; b.n_seq_id[j] = 1; b.seq_id[j][0] = k + 1;
                b.logits[j] = (i >= nq - 1 && i <= n - 2);    // predicts option token i + 1
            }
        }
        b.n_tokens = B * L;
        for (int k = 0; k < 4; k++) h->cap.calls[k] = 0;
        h->cap.on = true;
        const int rc = llama_decode(h->ctx, b);
        h->cap.on = false;
        llama_batch_free(b);
        if (rc != 0) return -5;
        for (int k = 0; k < 4; k++)
            if (h->cap.calls[k] != 1 || (int64_t) h->cap.buf[k].size() != (int64_t) B * L * D) {
                fprintf(stderr, "od_llama: %s captured %d times, %zu values, want 1 x %d\n", WANT[k], h->cap.calls[k],
                        h->cap.buf[k].size(), B * L * D);
                return -6;
            }
        // features of the real tokens, in row order (ubatch layout: row after row, as built above)
        for (int k = 0; k < B; k++) {
            const int r = r0 + k;
            for (int i = 0; i < row_len[r]; i++) {
                uint16_t * dst = feats + (tok_off[r] + i) * 4 * (size_t) D;
                for (int l = 0; l < 4; l++)
                    ggml_fp32_to_fp16_row(h->cap.buf[l].data() + ((size_t) k * L + i) * D, (ggml_fp16_t *) dst + l * D, D);
            }
        }
        // option-token log-probabilities, in parallel over output positions
        std::vector<std::pair<const float *, std::pair<int, float *>>> jobs;
        for (int k = 0; k < B; k++) {
            const int r = r0 + k, n = row_len[r], nq = row_nq[r];
            const int32_t * t = tokens + tok_off[r];
            for (int i = nq - 1; i <= n - 2; i++)
                jobs.push_back({llama_get_logits_ith(h->ctx, k * L + i), {t[i + 1], lp + lp_off[r] + (i - nq + 1)}});
        }
        const int nt = std::min<int>(h->n_threads, (int) jobs.size());
        std::vector<std::thread> th;
        for (int w = 0; w < nt; w++)
            th.emplace_back([&, w] {
                for (size_t j = w; j < jobs.size(); j += nt)
                    log_softmax_at(jobs[j].first, h->n_vocab, jobs[j].second.first, jobs[j].second.second);
            });
        for (auto & x : th) x.join();
        for (int s = 1; s <= B; s++) llama_memory_seq_rm(h->mem, s, -1, -1);
        r0 = r1;
    }
    return 0;
}

// Question cache: every question runs once from the state (its own sequence, one decode per question: a question
// sequence is continued, so it cannot be padded), then its option rows fork from it and carry only their own tokens
// (exact: the question's KV cells are shared, its recurrent state copied). The first option token's log-probability
// comes from the question's last position.
//   q_tokens: questions concatenated (q_len each); n_opt[i] options per question; o_tokens: all options in order
//   (o_len each). Out: q_feats (sum q_len x 4 n_embd), o_feats (sum o_len x 4 n_embd) fp16, lp (sum o_len) float32.
int odl_rows_qcache(void * hp, int n_q, const int32_t * q_tokens, const int32_t * q_len, const int32_t * n_opt,
                    const int32_t * o_tokens, const int32_t * o_len, uint16_t * q_feats, uint16_t * o_feats,
                    float * lp) {
    auto * h = (OdLlama *) hp;
    const int ns = (int) h->state.size(), D = h->n_embd;
    if (ns == 0) return -1;
    if (h->n_seq_max < 3) return -7;
    const int qseq = 1, first_row_seq = 2, max_rows = h->n_seq_max - 2;
    size_t q_off = 0, o_idx = 0, o_off = 0;
    std::vector<float> first_lp_logits;
    for (int qi = 0; qi < n_q; qi++) {
        const int nq = q_len[qi], K = n_opt[qi];
        if (nq < 1 || K < 1) return -3;
        // ---- the question, continuing from the state
        llama_memory_seq_rm(h->mem, qseq, -1, -1);
        llama_memory_seq_cp(h->mem, 0, qseq, -1, -1);
        if (nq > h->n_ubatch) return -4;
        {
            llama_batch b = llama_batch_init(nq, 0, 1);
            for (int i = 0; i < nq; i++) {
                b.token[i] = q_tokens[q_off + i]; b.pos[i] = ns + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = qseq;
                b.logits[i] = i == nq - 1;
            }
            b.n_tokens = nq;
            for (int k = 0; k < 4; k++) h->cap.calls[k] = 0;
            h->cap.on = true;
            const int rc = llama_decode(h->ctx, b);
            h->cap.on = false;
            llama_batch_free(b);
            if (rc != 0) return -5;
            for (int k = 0; k < 4; k++)
                if (h->cap.calls[k] != 1 || (int64_t) h->cap.buf[k].size() != (int64_t) nq * D) return -6;
            for (int i = 0; i < nq; i++)
                for (int l = 0; l < 4; l++)
                    ggml_fp32_to_fp16_row(h->cap.buf[l].data() + (size_t) i * D,
                                          (ggml_fp16_t *) q_feats + ((q_off + i) * 4 + l) * (size_t) D, D);
            const float * lg = llama_get_logits_ith(h->ctx, nq - 1);
            first_lp_logits.assign(lg, lg + h->n_vocab);
        }
        // ---- its options: rows forked from the question, in batches padded to one length (rows are never continued)
        std::vector<size_t> ooff(K + 1, o_off);
        for (int k = 0; k < K; k++) {
            if (o_len[o_idx + k] < 1) return -3;
            ooff[k + 1] = ooff[k] + o_len[o_idx + k];
        }
        std::vector<std::pair<const float *, std::pair<int, float *>>> jobs;
        for (int k = 0; k < K; k++)                         // first option token: from the question's last position
            jobs.push_back({first_lp_logits.data(), {o_tokens[ooff[k]], lp + ooff[k]}});
        auto run_jobs = [&] {
            const int nt = std::min<int>(h->n_threads, (int) jobs.size());
            std::vector<std::thread> th;
            for (int w = 0; w < nt; w++)
                th.emplace_back([&, w] {
                    for (size_t j = w; j < jobs.size(); j += nt)
                        log_softmax_at(jobs[j].first, h->n_vocab, jobs[j].second.first, jobs[j].second.second);
                });
            for (auto & x : th) x.join();
            jobs.clear();
        };
        run_jobs();
        int k0 = 0;
        while (k0 < K) {
            int k1 = k0, L = 0;
            while (k1 < K && k1 - k0 < max_rows) {
                const int L2 = std::max(L, (int) o_len[o_idx + k1]);
                if ((k1 - k0 + 1) * L2 > h->n_ubatch && k1 > k0) break;
                L = L2; k1++;
            }
            const int B = k1 - k0;
            if (B * L > h->n_ubatch) return -4;
            for (int r = 0; r < B; r++) {
                llama_memory_seq_rm(h->mem, first_row_seq + r, -1, -1);
                llama_memory_seq_cp(h->mem, qseq, first_row_seq + r, -1, -1);
            }
            llama_batch b = llama_batch_init(B * L, 0, 1);
            for (int r = 0; r < B; r++) {
                const int k = k0 + r, n = o_len[o_idx + k];
                const int32_t * t = o_tokens + ooff[k];
                for (int i = 0; i < L; i++) {
                    const int j = r * L + i;
                    b.token[j] = i < n ? t[i] : t[n - 1];
                    b.pos[j] = ns + nq + i; b.n_seq_id[j] = 1; b.seq_id[j][0] = first_row_seq + r;
                    b.logits[j] = i <= n - 2;                    // predicts option token i + 1
                }
            }
            b.n_tokens = B * L;
            for (int k = 0; k < 4; k++) h->cap.calls[k] = 0;
            h->cap.on = true;
            const int rc = llama_decode(h->ctx, b);
            h->cap.on = false;
            llama_batch_free(b);
            if (rc != 0) return -5;
            for (int k = 0; k < 4; k++)
                if (h->cap.calls[k] != 1 || (int64_t) h->cap.buf[k].size() != (int64_t) B * L * D) {
                    fprintf(stderr, "od_llama: %s captured %d times, %zu values, want 1 x %d\n", WANT[k],
                            h->cap.calls[k], h->cap.buf[k].size(), B * L * D);
                    return -6;
                }
            for (int r = 0; r < B; r++) {
                const int k = k0 + r, n = o_len[o_idx + k];
                for (int i = 0; i < n; i++)
                    for (int l = 0; l < 4; l++)
                        ggml_fp32_to_fp16_row(h->cap.buf[l].data() + ((size_t) r * L + i) * D,
                                              (ggml_fp16_t *) o_feats + ((ooff[k] + i) * 4 + l) * (size_t) D, D);
                const int32_t * t = o_tokens + ooff[k];
                for (int i = 0; i <= n - 2; i++)
                    jobs.push_back({llama_get_logits_ith(h->ctx, r * L + i), {t[i + 1], lp + ooff[k] + i + 1}});
            }
            run_jobs();
            for (int r = 0; r < B; r++) llama_memory_seq_rm(h->mem, first_row_seq + r, -1, -1);
            k0 = k1;
        }
        llama_memory_seq_rm(h->mem, qseq, -1, -1);
        q_off += nq; o_idx += K; o_off = ooff[K];
    }
    return 0;
}

void odl_free(void * hp) {
    auto * h = (OdLlama *) hp;
    if (!h) return;
    if (h->ctx) llama_free(h->ctx);
    if (h->model) llama_model_free(h->model);
    delete h;
}

}  // extern "C"
