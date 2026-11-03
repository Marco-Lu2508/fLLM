// Logits of every position of a token sequence from llama.cpp's libllama (CPU, f32 KV cache), as one JSON line:
//   {"logits": [...]}
// Reference for scripts/verify_qwen4exp_gguf.py; build against a llama.cpp checkout with
//   g++ -O2 -std=c++17 llama_cpp_logits.cpp -I<llama.cpp>/include -I<llama.cpp>/ggml/include -L<llama.cpp>/build/bin -lllama -lggml -lggml-base \
//       -Wl,-rpath,<llama.cpp>/build/bin -o llama_cpp_logits
// usage: llama_cpp_logits MODEL.gguf TOKEN_ID...
// With the environment variable LLAMA_DUMP_LAYERS=DIR the hyper-connection streams after every layer of a `qwen4exp` model are written to
// DIR/l<layer>_ffn.f32 (raw f32, token x stream x hidden), the files fLLM writes with FLLM_DUMP_LAYERS (compare with scripts/cmpdump.py).
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string g_dump_dir;

static bool dump_callback(struct ggml_tensor * t, bool ask, void *) {
    if (strncmp(t->name, "l_last-", 7) != 0) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    std::vector<char> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    const std::string path = g_dump_dir + "/l" + std::to_string(atoi(t->name + 7)) + "_ffn.f32";
    FILE * f = fopen(path.c_str(), "ab");
    if (f) {
        fwrite(bytes.data(), 1, bytes.size(), f);
        fclose(f);
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf TOKEN_ID...\n", argv[0]);
        return 2;
    }
    std::vector<llama_token> ids;
    for (int i = 2; i < argc; ++i) {
        ids.push_back((llama_token) atoi(argv[i]));
    }
    const int n = (int) ids.size();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (const char * dir = getenv("LLAMA_DUMP_LAYERS")) {
        g_dump_dir = dir;
        cp.cb_eval = dump_callback;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "cannot create a context\n");
        return 1;
    }
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        batch.token[i] = ids[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        return 1;
    }
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    printf("{\"logits\": [");
    for (int i = 0; i < n; ++i) {
        const float * row = llama_get_logits_ith(ctx, i);
        for (int j = 0; j < vocab; ++j) {
            printf("%s%.9g", (i || j) ? ", " : "", row[j]);
        }
    }
    printf("]}\n");
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
