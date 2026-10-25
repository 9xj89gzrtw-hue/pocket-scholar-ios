// Smallest possible proof that llama.cpp runs on the iOS simulator:
// load a GGUF, evaluate one prompt, print the completion to stdout.
#include "llama.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf \"prompt\"\n", argv[0]); return 2; }
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    printf("loaded model: %d params, n_vocab=%d\n", llama_model_n_params(model), llama_vocab_n_tokens(vocab));

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 512; cp.n_threads = 4;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    int n = llama_tokenize(vocab, argv[2], (int)strlen(argv[2]), NULL, 0, true, true);
    llama_token *toks = malloc(sizeof(llama_token) * n);
    llama_tokenize(vocab, argv[2], (int)strlen(argv[2]), toks, n, true, true);
    llama_batch batch = llama_batch_get_one(toks, n);
    if (llama_decode(ctx, batch)) { fprintf(stderr, "decode failed\n"); return 1; }

    struct llama_sampler *sm = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sm, llama_sampler_init_greedy());
    llama_token cur = 0;
    for (int i = 0; i < 40; i++) {
        cur = llama_sampler_sample(sm, ctx, -1);
        if (llama_vocab_is_eog(vocab, cur)) break;
        char buf[256];
        int len = llama_token_to_piece(vocab, cur, buf, sizeof(buf), 0, true);
        if (len > 0) fwrite(buf, 1, len, stdout);
        llama_batch nb = llama_batch_get_one(&cur, 1);
        if (llama_decode(ctx, nb)) break;
    }
    printf("\n-- done --\n");
    llama_sampler_free(sm);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
