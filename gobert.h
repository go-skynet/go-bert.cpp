#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

int bert_token_embeddings(void* params_ptr, void* state_pr, int *inp_tokens,int n_tokens, float * res_embeddings);
int bert_embeddings(void* params_ptr, void* state_pr, float * res_embeddings);
int bert_bootstrap(const char *model_path, void* state_pr);
void* bert_allocate_state() ;
void bert_free_model(void *state_ptr);
void bert_free_params(void* params_ptr) ;
void* bert_allocate_params(const char *prompt,  int threads) ;

#ifdef __cplusplus
}
#endif