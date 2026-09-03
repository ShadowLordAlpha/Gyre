#pragma once

#include "gyre/nn/transformer.hpp"

#include <filesystem>

namespace gyre {

// Writes a loadable ONNX model (opset 17) for CharLM inference.
// Input `tokens`: int64 [1, block_size]. Output `logits`: float [1, block_size, vocab].
// No ONNX Runtime link; consumers load the file elsewhere.
Result<void> export_charlm_onnx(const CharLM& model, const std::filesystem::path& path);

}  // namespace gyre
