#include "gyre/data.hpp"
#include "gyre/nn/tokenize.hpp"
#include "gyre/nn/transformer.hpp"
#include "gyre/train/loop.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

const char* kFallback =
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles\n"
    "And by opposing end them. To die—to sleep,\n"
    "No more; and by a sleep to say we end\n"
    "The heart-ache and the thousand natural shocks\n"
    "That flesh is heir to. To be or not to be.\n";

std::string load_text(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
  using namespace gyre;
  std::filesystem::path data_path = "data/shakespeare.txt";
  std::uint32_t steps = 200;
  std::uint32_t batch = 8;
  std::uint32_t block = 64;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--data" && i + 1 < argc) data_path = argv[++i];
    else if (a == "--steps" && i + 1 < argc) steps = static_cast<std::uint32_t>(std::stoul(argv[++i]));
    else if (a == "--batch" && i + 1 < argc) batch = static_cast<std::uint32_t>(std::stoul(argv[++i]));
    else if (a == "--block" && i + 1 < argc) block = static_cast<std::uint32_t>(std::stoul(argv[++i]));
    else if (a == "--help") {
      std::cout << "gyre-charlm — character LM example (prefer gyre-cli lm train)\n"
                   "  gyre-charlm [--data data/shakespeare.txt] [--steps 200] [--batch 8] [--block 64]\n"
                   "Default data is Karpathy tiny Shakespeare (~1MB):\n"
                   "  curl.exe -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/shakespeare.txt\n";
      return 0;
    }
  }

  auto dev = Device::cpu();
  if (!dev) return 1;
  std::string text = load_text(data_path);
  if (text.size() < 200) {
    std::cerr << "No usable file at " << data_path << " — using a tiny built-in snippet.\n";
    text = kFallback;
  } else {
    std::cout << "loaded " << text.size() << " bytes from " << data_path << '\n';
  }

  auto tok = Tokenizer::chars_from_text(text);
  if (!tok) {
    std::cerr << tok.error().message << '\n';
    return 1;
  }
  auto ids = (*tok)->encode(text);
  if (!ids) return 1;
  auto data = CharDataset::from_ids(*ids, *dev);
  if (!data) return 1;
  Rng rng(1);
  CharLMConfig cfg;
  cfg.vocab = (*tok)->vocab_size();
  cfg.block_size = block;
  cfg.n_layer = 2;
  cfg.n_head = 4;
  cfg.d_model = 64;
  cfg.d_ff = 256;
  auto model = CharLM::create(cfg, *dev, rng);
  if (!model) {
    std::cerr << model.error().message << '\n';
    return 1;
  }
  std::int64_t n = 0;
  for (auto& p : model->parameters()) n += p.value.numel();
  std::cout << "params " << n << " vocab " << cfg.vocab << '\n';
  TrainConfig tc;
  tc.steps = steps;
  tc.batch = batch;
  tc.block = block;
  tc.lr = 3e-4f;
  tc.log_every = 10;
  tc.ckpt_every = 0;
  TrainLoop loop;
  auto r = loop.run(*model, *data, tc, *dev, [](const Metrics& m) {
    std::cout << "step " << m.step << " loss " << m.loss << '\n';
  });
  if (!r) {
    std::cerr << r.error().message << '\n';
    return 1;
  }
  std::vector<std::int32_t> prefix = {ids->front()};
  Rng sample_rng(2);
  auto gen = model->generate(prefix, 200, *dev, &sample_rng, 0.8f);
  if (!gen) {
    std::cerr << gen.error().message << '\n';
    return 1;
  }
  std::cout << (*tok)->decode(*gen) << '\n';
  return 0;
}
