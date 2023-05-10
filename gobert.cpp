#include "ggml.h"
#include <unistd.h>
#include <stdio.h>
#include <vector>
#include "bert.h"
#include "bert.cpp"
#include "gobert.h"

struct bert_state {
    bert_ctx* model;
    struct {
        int64_t t_load_us = -1;
        int64_t t_sample_us = -1;
        int64_t t_predict_us = -1;
    } timing;
};

int bert_embeddings(void* params_ptr, void* state_pr, float * res_embeddings) {
    const int64_t t_main_start_us = ggml_time_us();
    bert_params params = *(bert_params*) params_ptr;
    bert_state * st = (bert_state*) state_pr;
    bert_ctx * bctx = st->model;

    int N = bert_n_max_tokens(bctx);
    // tokenize the prompt
    std::vector<bert_vocab_id> tokens(N);
    int n_tokens;
    bert_tokenize(bctx, params.prompt, tokens.data(), &n_tokens, N);
    tokens.resize(n_tokens);
    std::vector<float> embeddings(bert_n_embd(bctx));
    bert_eval(bctx, params.n_threads, tokens.data(), n_tokens, embeddings.data());
    
    for (int i = 0; i < embeddings.size(); i++) {
                res_embeddings[i]=embeddings[i];
             //    printf("embedding %d\n",embeddings[i]);

    }
    return 0;   
}

int bert_bootstrap(const char *model_path, void* state_pr)
// load the model
{
    ggml_time_init();
    bert_state* state = (bert_state*) state_pr;

    const int64_t t_start_us = ggml_time_us();
    bert_ctx * bctx;

    if ((bctx = bert_load_from_file(model_path)) == nullptr) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, model_path);
            return 1;
    }
    printf("loaded\n");

    state->model = bctx;

    state->timing.t_load_us = ggml_time_us() - t_start_us;
    return 0;
}

void* bert_allocate_state() {
    return new bert_state;
}

void bert_free_model(void *state_ptr) {
    bert_state* state = (bert_state*) state_ptr;
    bert_free(state->model);
}

void bert_free_params(void* params_ptr) {
    bert_params* params = (bert_params*) params_ptr;
    delete params;
}

void* bert_allocate_params(const char *prompt,  int threads) {
    bert_params* params = new bert_params;
    params->n_threads = threads;
    params->prompt = prompt;
    
    return params;
}
