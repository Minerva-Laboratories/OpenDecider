// OpenDecider v3 row features from llama.cpp (P8.2 feasibility probe).
//
// For each case: the state tokens run once with the adapter off (sequence 0); every option row copies that sequence
// (llama_memory_seq_cp: KV for full attention, conv/recurrent state for the gated delta net layers) and runs
// [question tokens][option tokens] with the VeRA-as-LoRA adapter on. Read through the eval callback: the residual
// stream after blocks 8, 16, 24 (l_out-7/15/23) and the final-norm output (h_nextn), all row tokens; from the logits,
// the log-probability of every option token given the previous position (the LM feature).
//
// Input (int32 stream): n_cases, then per case: n_state, ids...; n_questions; per question: n_q, ids...; n_options;
// per option: n_o, ids...
// Output (sequential, same order): per question: question-token features from its first row (n_q x 16384 fp16);
// per option: option-token features (n_o x 16384 fp16), then option-token log-probs (n_o float32).
//
//   od_features model.gguf adapter.gguf in.bin out.bin [n_gpu_layers]
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char * WANT[4] = {"l_out-7", "l_out-15", "l_out-23", "h_nextn"};

struct Capture {
    int n_embd = 0;
    std::vector<float> buf[4];   // n_tokens x n_embd each
    bool on = false;
};

static int want_idx(const char * name) {
    for (int i = 0; i < 4; i++) if (strcmp(name, WANT[i]) == 0) return i;
    return -1;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    auto * c = (Capture *) ud;
    int k = want_idx(t->name);
    if (ask) return c->on && k >= 0;
    if (!c->on || k < 0) return true;
    const int64_t n = ggml_nelements(t);
    c->buf[k].resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, c->buf[k].data(), 0, n * sizeof(float));
    } else {
        fprintf(stderr, "unexpected type for %s\n", t->name); exit(1);
    }
    return true;
}

static std::vector<int32_t> read_ids(FILE * f) {
    int32_t n; if (fread(&n, 4, 1, f) != 1) { fprintf(stderr, "bad input\n"); exit(1); }
    std::vector<int32_t> v(n); if (n && fread(v.data(), 4, n, f) != (size_t) n) { fprintf(stderr, "bad input\n"); exit(1); }
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model.gguf adapter.gguf in.bin out.bin [ngl]\n", argv[0]); return 1; }
    const int ngl = argc > 5 ? atoi(argv[5]) : 99;
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    llama_adapter_lora * ad = std::string(argv[2]) == "none" ? nullptr : llama_adapter_lora_init(model, argv[2]);
    const int n_embd = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    Capture cap; cap.n_embd = n_embd;
    auto cp = llama_context_default_params();
    cp.n_ctx = 16384; cp.n_batch = 8192; cp.n_ubatch = 512; cp.n_seq_max = 2;
    cp.cb_eval = cb_eval; cp.cb_eval_user_data = &cap;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    llama_memory_t mem = llama_get_memory(ctx);

    FILE * fi = fopen(argv[3], "rb"); FILE * fo = fopen(argv[4], "wb");
    int32_t n_cases; if (fread(&n_cases, 4, 1, fi) != 1) return 1;
    std::vector<ggml_fp16_t> h16;
    double t_state = 0, t_rows = 0; long n_rows = 0;
    for (int ci = 0; ci < n_cases; ci++) {
        auto st = read_ids(fi);
        llama_memory_clear(mem, true);
        // state pass: adapter off, no outputs needed except to satisfy the API (last token)
        llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
        auto t0 = std::chrono::steady_clock::now();
        {
            llama_batch b = llama_batch_init((int) st.size(), 0, 1);
            for (size_t i = 0; i < st.size(); i++) {
                b.token[i] = st[i]; b.pos[i] = (llama_pos) i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
                b.logits[i] = i + 1 == st.size();
            }
            b.n_tokens = (int) st.size();
            cap.on = false;
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "state decode failed\n"); return 1; }
            llama_batch_free(b);
        }
        t_state += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (ad) { float s = 1.0f; llama_set_adapters_lora(ctx, &ad, 1, &s); }
        int32_t n_q; if (fread(&n_q, 4, 1, fi) != 1) return 1;
        for (int qi = 0; qi < n_q; qi++) {
            auto q = read_ids(fi);
            int32_t n_o; if (fread(&n_o, 4, 1, fi) != 1) return 1;
            for (int oi = 0; oi < n_o; oi++) {
                auto o = read_ids(fi);
                auto t1 = std::chrono::steady_clock::now();
                llama_memory_seq_rm(mem, 1, -1, -1);
                llama_memory_seq_cp(mem, 0, 1, -1, -1);
                const int nq = (int) q.size(), no = (int) o.size(), n = nq + no;
                llama_batch b = llama_batch_init(n, 0, 1);
                for (int i = 0; i < n; i++) {
                    b.token[i] = i < nq ? q[i] : o[i - nq];
                    b.pos[i] = (llama_pos) (st.size() + i); b.n_seq_id[i] = 1; b.seq_id[i][0] = 1;
                    b.logits[i] = 1;
                }
                b.n_tokens = n;
                cap.on = true;
                if (llama_decode(ctx, b) != 0) { fprintf(stderr, "row decode failed\n"); return 1; }
                cap.on = false;
                llama_batch_free(b);
                t_rows += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count(); n_rows++;
                for (int k = 0; k < 4; k++)
                    if ((int64_t) cap.buf[k].size() != (int64_t) n * n_embd) {
                        fprintf(stderr, "capture %s has %zu values, want %d\n", WANT[k], cap.buf[k].size(), n * n_embd); return 1;
                    }
                // features: [L8 | L16 | L24 | final] per token
                auto dump = [&](int from, int count) {
                    h16.resize((size_t) count * 4 * n_embd);
                    for (int t = 0; t < count; t++)
                        for (int k = 0; k < 4; k++)
                            ggml_fp32_to_fp16_row(cap.buf[k].data() + (size_t) (from + t) * n_embd,
                                                  h16.data() + ((size_t) t * 4 + k) * n_embd, n_embd);
                    fwrite(h16.data(), sizeof(ggml_fp16_t), h16.size(), fo);
                };
                if (oi == 0) dump(0, nq);
                dump(nq, no);
                std::vector<float> lp(no);
                for (int t = 0; t < no; t++) {
                    const float * lg = llama_get_logits_ith(ctx, nq + t - 1);
                    double m = -1e30; for (int v = 0; v < n_vocab; v++) m = lg[v] > m ? lg[v] : m;
                    double s = 0; for (int v = 0; v < n_vocab; v++) s += std::exp(lg[v] - m);
                    lp[t] = (float) (lg[o[t]] - m - std::log(s));
                }
                fwrite(lp.data(), sizeof(float), no, fo);
            }
        }
        if ((ci + 1) % 10 == 0) fprintf(stderr, "%d/%d cases, state %.2f s/case, rows %.3f s/row\n", ci + 1, n_cases,
                                        t_state / (ci + 1), t_rows / n_rows);
    }
    fprintf(stderr, "done: %d cases, state %.3f s/case, rows %.3f s/row (%ld rows)\n", n_cases, t_state / n_cases,
            t_rows / std::max(1L, n_rows), n_rows);
    fclose(fi); fclose(fo);
    llama_free(ctx); llama_model_free(model);
    return 0;
}
