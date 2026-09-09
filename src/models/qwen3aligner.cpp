#include "models.h"

void llama_model_qwen3aligner::load_arch_hparams(llama_model_loader & ml) {
    llama_model_qwen3::load_arch_hparams(ml);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH_OUT, hparams.n_embd_out_impl);
    if (hparams.n_embd_out_impl < 2) {
        throw std::runtime_error("qwen3aligner requires at least two timestamp classes");
    }
    hparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
}
