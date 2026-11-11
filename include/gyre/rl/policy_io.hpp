#pragma once
// Persistence for PolicyNet.
//  * save_policy / load_policy: a training checkpoint in the standard `.gyre` format
//    (arch "policy-net"; config + weights; optional Adam state is not stored here).
//  * export_policy_flat: a small fixed-schema binary for the dependency-free `gyre-infer`
//    runtime (see infer/include/gyre_infer/policy.hpp). Layout, little-endian:
//      char magic[8] = "GPOL1\0\0\0"; u32 version = 1; u32 reserved = 0;
//      u32 feat_dim, global_dim, max_entities, d_model, n_layer, n_head, d_ff, out_dim,
//          gru_hidden, n_heads;
//      u32 head_sizes[n_heads];
//      u64 n_floats;                     // must equal the count implied by the config
//      f32 data[n_floats];               // PolicyNet::parameters() order
//      u32 fnv1a32 of the data bytes.
#include "gyre/rl/policy.hpp"

#include <filesystem>

namespace gyre {

Result<void> save_policy(const std::filesystem::path& path, PolicyNet& net, std::uint64_t train_step = 0);
Result<PolicyNet> load_policy(const std::filesystem::path& path, std::shared_ptr<Device> device);
Result<void> export_policy_flat(const std::filesystem::path& path, PolicyNet& net);

}  // namespace gyre
