#pragma once
// Hugging Face LlamaForCausalLM (SmolLM2, TinyLlama, ...) -> Gyre LlamaLM.
//
//   model.embed_tokens            [V, d]       -> embed.weight            (as is)
//   layers.i.input_layernorm      [d]          -> layers.i.attn_norm.weight
//   layers.i.self_attn.q_proj     [Hq*hd, d]   -> layers.i.attn.q.weight  [d, Hq*hd]  transposed + RoPE permute
//   layers.i.self_attn.k_proj     [Hkv*hd, d]  -> layers.i.attn.k.weight  [d, Hkv*hd] transposed + RoPE permute
//   layers.i.self_attn.v_proj / o_proj         -> layers.i.attn.v / attn.o (transposed)
//   layers.i.post_attention_layernorm          -> layers.i.mlp_norm.weight
//   layers.i.mlp.gate/up/down_proj             -> layers.i.mlp.gate / up / down (transposed)
//   model.norm                                 -> norm.weight
//   lm_head [V, d] (only when not tied)        -> lm_head.weight [d, V]
//
// "RoPE permute": inside every head, HF output index j < hd/2 moves to 2j and j >= hd/2 moves to
// 2(j - hd/2) + 1, turning HF's rotate_half pairs (j, j + hd/2) into gyre::rope's (2i, 2i + 1).
// Tensors may be f32/f16/bf16 (widened exactly to f32). Single-file model.safetensors only.
#include "gyre/nn/llama.hpp"

#include <filesystem>

namespace gyre {

Result<LlamaLM> import_llama_hf(const std::filesystem::path& dir);

// The column permutation above for one projection already in Gyre [in, out] layout (exposed for
// tests): returns the destination column of HF output row `o`.
std::int64_t llama_rope_perm(std::int64_t o, std::int64_t head_dim) noexcept;

}  // namespace gyre
