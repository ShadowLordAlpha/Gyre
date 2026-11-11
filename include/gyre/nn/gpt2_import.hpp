#pragma once
// Hugging Face GPT-2 (openai-community/gpt2 and the medium/large/xl siblings) -> Gyre CharLM.
//
// GPT-2 is a pre-LayerNorm decoder with learned positions, tanh-GELU and biases everywhere, which
// is exactly CharLM with `recency_alibi = false`. The conversion is only a repackaging:
//   wte/wpe                      -> wte/wpe
//   h.i.ln_1 / ln_2 / ln_f       -> blocks.i.ln1 / ln2 / ln_f
//   h.i.attn.c_attn  [d, 3d]     -> blocks.i.attn.q/k/v  (split into thirds along the output axis)
//   h.i.attn.c_proj  [d, d]      -> blocks.i.attn.o
//   h.i.mlp.c_fc / c_proj        -> blocks.i.mlp.fc1 / fc2
//   (tied) wte^T                 -> lm_head.weight, lm_head.bias = 0
// GPT-2's Conv1D weights are already [in, out], the layout of gyre::Linear, so nothing else is
// transposed. The causal mask buffers (`h.i.attn.bias`) are ignored. Tensors may be f32/f16/bf16.
#include "gyre/nn/transformer.hpp"

#include <filesystem>

namespace gyre {

// `dir` holds config.json and model.safetensors (single file). Errors on any other architecture
// (activation != gelu_new, layer_norm_epsilon != 1e-5, scale_attn_by_inverse_layer_idx, ...).
Result<CharLM> import_gpt2(const std::filesystem::path& dir, std::shared_ptr<Device> device);

}  // namespace gyre
