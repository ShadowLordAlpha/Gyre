#include "jobs.hpp"
#include "gyre/nn/tokenize.hpp"

#include <filesystem>
#include <iostream>
#include <string>

static void usage() {
  std::cout <<
      "gyre-cli — Gyre headless runner\n"
      "  gyre-cli ga [--gens 80] [--n 64] [--dim 64]\n"
      "  gyre-cli lm train --data data/shakespeare.txt [--preset medium|tiny|tinygpt|nanogpt]\n"
      "                   [--tokenizer bpe|chars|bytes|unigram] [--tok FILE.gyre.json] [--vocab-size 2000]\n"
      "                   [--holdout 0.1] [--steps 2000] [--batch 4] [--ckpt data/charlm.gyre]\n"
      "                   [--lr 3e-4] [--lr-start 1e-3] [--recency alibi|none]\n"
      "                   [--hf-dir DIR] [--sp FILE.model]\n"
      "  gyre-cli lm generate --data data/shakespeare.txt --ckpt data/charlm.gyre\n"
      "                      [--preset medium|tiny|tinygpt] [--prompt \"To be\"] [--chars 200] [--temp 0.8]\n"
      "  gyre-cli lm export --ckpt data/charlm.gyre --onnx data/charlm.onnx [--preset medium]\n"
      "                    [--data data/shakespeare.txt]\n"
      "  gyre-cli lm eval --ckpt data/charlm.gyre --data data/shakespeare.txt [--preset tinygpt]\n"
      "                  [--split 0.1]\n"
      "  gyre-cli tok train --data data/shakespeare.txt [--tokenizer bpe|chars|bytes|unigram]\n"
      "                    [--vocab-size 2000] --out data/tok.gyre.json\n"
      "  gyre-cli tok export --tok data/tok.gyre.json --hf-dir data/hf-tok\n"
      "  gyre-cli tok import --hf-dir DIR|--sp FILE.model --out data/tok.gyre.json\n"
      "  gyre-cli tok encode --tok FILE [--text \"hello world\"]\n"
      "  gyre-cli grok info [--config data/grok2/config.json]\n"
      "  gyre-cli grok inspect DIR\n"
      "  gyre-cli grok compress-probe DIR [--file SUBSTR] [--max-tensors N] [--chunk-bytes N] [--out FILE]\n"
      "  gyre-cli grok pack DIR --out FILE.wpack [--file SUBSTR] [--max-bytes N]\n"
      "  gyre-cli grok save --preset mini|tiny --out FILE.gyre [--codec identity|alp|zfp]\n"
      "  gyre-cli grok import DIR --out FILE.gyre [--codec identity|alp|zfp]\n"
      "  gyre-cli grok gen --weights FILE.gyre|DIR [--prompt \"To be\"] [--max-new 32] [--tok FILE] [--lora DIR]\n"
      "  gyre-cli ckpt probe FILE.gyre [--max-tensors N]\n"
      "  gyre-cli tui   (build gyre-tui with -DGYRE_ENABLE_TUI=ON)\n"
      "\n"
      "lm is a character-level transformer. Checkpoints use the .gyre extension\n"
      "(format version lives inside the file). Default preset is medium\n"
      "(d=128, 4 layers, T=128). tiny is the 64-wide net. tinygpt is byte-level\n"
      "(V=256, d=192, 6 layers, T=256). nanogpt is 6×384 T=256 (~10.7M at char vocab).\n"
      "Default tokenizer is BPE (vocab 2000). Chars/bytes are BPE with no merges.\n"
      "Pass --tok FILE.gyre.json to reuse a tokenizer (see docs/tokenizer.md).\n"
      "Train holdout (default 0.1) is the last\n"
      "raw-byte fraction; BPE and LM see only the prefix. Match eval --split to it.\n"
      "export writes ONNX (no Runtime link).\n"
      "--data is any UTF-8 text; default path is a Shakespeare corpus.\n";
}

int main(int argc, char** argv) {
  using namespace gyre::app;
  bool bar_on = false;
  auto log = [&](std::string s) {
    if (!s.empty() && s.front() == '\r') {
      bar_on = true;
      std::cout << s << std::flush;
      return;
    }
    if (bar_on) {
      std::cout << '\n';
      bar_on = false;
    }
    std::cout << s << '\n';
  };

  if (argc < 2) {
    usage();
    return 0;
  }
  std::string cmd = argv[1];
  if (cmd == "-h" || cmd == "--help") {
    usage();
    return 0;
  }
  if (cmd == "tui") {
    std::cout << "Build with -DGYRE_ENABLE_TUI=ON and run gyre-tui\n";
    return 0;
  }
  if (cmd == "tok") {
    if (argc < 3) {
      usage();
      return 1;
    }
    std::string sub = argv[2];
    CharLMOpts o;
    std::filesystem::path out;
    std::string enc_text = "hello world";
    for (int i = 3; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--data" && i + 1 < argc) o.data = argv[++i];
      else if (a == "--tokenizer" && i + 1 < argc) o.tokenizer = argv[++i];
      else if (a == "--vocab-size" && i + 1 < argc) o.vocab_size = std::stoi(argv[++i]);
      else if (a == "--out" && i + 1 < argc) out = argv[++i];
      else if (a == "--tok" && i + 1 < argc) o.tok = argv[++i];
      else if (a == "--hf-dir" && i + 1 < argc) o.hf_dir = argv[++i];
      else if (a == "--sp" && i + 1 < argc) o.sp_model = argv[++i];
      else if (a == "--holdout" && i + 1 < argc) o.holdout = std::stod(argv[++i]);
      else if (a == "--text" && i + 1 < argc) enc_text = argv[++i];
    }
    if (sub == "train") {
      auto r = run_tok_train(o, out, log);
      if (!r) {
        std::cerr << r.error().message << '\n';
        return 1;
      }
      return 0;
    }
    if (sub == "export") {
      auto r = run_tok_export_hf(o.tok, o.hf_dir, log);
      if (!r) {
        std::cerr << r.error().message << '\n';
        return 1;
      }
      return 0;
    }
    if (sub == "import") {
      auto r = run_tok_import(o, out, log);
      if (!r) {
        std::cerr << r.error().message << '\n';
        return 1;
      }
      return 0;
    }
    if (sub == "encode") {
      if (o.tok.empty()) {
        std::cerr << "tok encode needs --tok\n";
        return 1;
      }
      auto t = gyre::Tokenizer::load(o.tok);
      if (!t) {
        std::cerr << t.error().message << '\n';
        return 1;
      }
      auto ids = (*t)->encode(enc_text);
      if (!ids) {
        std::cerr << ids.error().message << '\n';
        return 1;
      }
      for (std::size_t i = 0; i < ids->size(); ++i) {
        if (i) std::cout << ' ';
        std::cout << (*ids)[i];
      }
      std::cout << '\n';
      return 0;
    }
    usage();
    return 1;
  }
  if (cmd == "grok") {
    GrokCliOpts go;
    std::string sub = argc > 2 ? argv[2] : "info";
    for (int i = 2; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "info" || a == "inspect" || a == "compress-probe" || a == "pack" ||
          a == "save" || a == "gen" || a == "import") {
        sub = a;
        if ((a == "inspect" || a == "compress-probe" || a == "pack" || a == "import" || a == "gen") &&
            i + 1 < argc && argv[i + 1][0] != '-')
          go.weights = argv[++i];
      } else if (a == "--config" && i + 1 < argc)
        go.config = argv[++i];
      else if (a == "--weights" && i + 1 < argc)
        go.weights = argv[++i];
      else if (a == "--out" && i + 1 < argc)
        go.out = argv[++i];
      else if (a == "--file" && i + 1 < argc)
        go.file_substr = argv[++i];
      else if (a == "--preset" && i + 1 < argc)
        go.preset = argv[++i];
      else if (a == "--prompt" && i + 1 < argc)
        go.prompt = argv[++i];
      else if (a == "--max-new" && i + 1 < argc)
        go.max_new = std::stoi(argv[++i]);
      else if (a == "--tok" && i + 1 < argc)
        go.tok = argv[++i];
      else if (a == "--lora" && i + 1 < argc)
        go.lora = argv[++i];
      else if (a == "--codec" && i + 1 < argc)
        go.codec = argv[++i];
      else if (a == "--max-bytes" && i + 1 < argc)
        go.pack_max = std::stoull(argv[++i]);
      else if (a == "--max-tensors" && i + 1 < argc)
        go.max_tensors = std::stoi(argv[++i]);
      else if (a == "--chunk-bytes" && i + 1 < argc)
        go.chunk_bytes = std::stoull(argv[++i]);
    }
    gyre::Result<void> r;
    if (sub == "info")
      r = run_grok_info(go, log);
    else if (sub == "inspect")
      r = run_grok_inspect(go, log);
    else if (sub == "compress-probe")
      r = run_grok_compress_probe(go, log);
    else if (sub == "pack")
      r = run_grok_pack(go, log);
    else if (sub == "save")
      r = run_grok_save(go, log);
    else if (sub == "gen")
      r = run_grok_gen(go, log);
    else if (sub == "import")
      r = run_grok_import(go, log);
    else {
      usage();
      return 1;
    }
    if (!r) {
      std::cerr << r.error().message << '\n';
      return 1;
    }
    return 0;
  }
  if (cmd == "ckpt") {
    if (argc < 3 || std::string(argv[2]) != "probe") {
      usage();
      return 1;
    }
    std::filesystem::path ckpt;
    int max_t = 32;
    for (int i = 3; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--max-tensors" && i + 1 < argc)
        max_t = std::stoi(argv[++i]);
      else if (!a.empty() && a[0] != '-')
        ckpt = a;
    }
    auto r = run_ckpt_probe(ckpt, max_t, log);
    if (!r) {
      std::cerr << r.error().message << '\n';
      return 1;
    }
    return 0;
  }
  if (cmd == "ga") {
    GaOpts o;
    for (int i = 2; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--gens" && i + 1 < argc) o.generations = static_cast<std::uint32_t>(std::stoul(argv[++i]));
      else if (a == "--n" && i + 1 < argc) o.n = static_cast<std::uint32_t>(std::stoul(argv[++i]));
      else if (a == "--dim" && i + 1 < argc) o.dim = static_cast<std::uint32_t>(std::stoul(argv[++i]));
    }
    auto r = run_onemax(o, log);
    if (!r) {
      std::cerr << r.error().message << '\n';
      return 1;
    }
    return 0;
  }
  if (cmd == "lm" || cmd == "shakespeare") {
    if (cmd == "shakespeare") {
      std::cerr << "note: 'shakespeare' is deprecated; use 'gyre-cli lm …'\n";
    }
    if (argc < 3) {
      usage();
      return 1;
    }
    std::string sub = argv[2];
    CharLMOpts o;
    std::string prompt = "To be";
    int chars = 200;
    std::filesystem::path onnx_path = "data/charlm.onnx";
    double split = -1.0;
    bool split_set = false, holdout_set = false;
    bool block_set = false, d_model_set = false, n_layer_set = false, n_head_set = false, d_ff_set = false;
    std::uint32_t block_ov = 0;
    std::int64_t d_model_ov = 0, n_layer_ov = 0, n_head_ov = 0, d_ff_ov = 0;
    for (int i = 3; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--data" && i + 1 < argc) o.data = argv[++i];
      else if (a == "--ckpt" && i + 1 < argc) o.ckpt = argv[++i];
      else if (a == "--steps" && i + 1 < argc) o.steps = static_cast<std::uint32_t>(std::stoul(argv[++i]));
      else if (a == "--batch" && i + 1 < argc) o.batch = static_cast<std::uint32_t>(std::stoul(argv[++i]));
      else if (a == "--block" && i + 1 < argc) {
        block_ov = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        block_set = true;
      }
      else if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
      else if (a == "--chars" && i + 1 < argc) chars = std::stoi(argv[++i]);
      else if (a == "--temp" && i + 1 < argc) o.temperature = std::stof(argv[++i]);
      else if (a == "--preset" && i + 1 < argc) o.preset = argv[++i];
      else if (a == "--ckpt-every" && i + 1 < argc)
        o.ckpt_every = static_cast<std::uint32_t>(std::stoul(argv[++i]));
      else if (a == "--lr" && i + 1 < argc) o.lr = std::stof(argv[++i]);
      else if (a == "--lr-start" && i + 1 < argc) o.lr_start = std::stof(argv[++i]);
      else if (a == "--lr-decay-steps" && i + 1 < argc)
        o.lr_decay_steps = static_cast<std::uint32_t>(std::stoul(argv[++i]));
      else if (a == "--onnx" && i + 1 < argc) onnx_path = argv[++i];
      else if (a == "--tokenizer" && i + 1 < argc) o.tokenizer = argv[++i];
      else if (a == "--tok" && i + 1 < argc) o.tok = argv[++i];
      else if (a == "--hf-dir" && i + 1 < argc) o.hf_dir = argv[++i];
      else if (a == "--sp" && i + 1 < argc) o.sp_model = argv[++i];
      else if (a == "--vocab-size" && i + 1 < argc) o.vocab_size = std::stoi(argv[++i]);
      else if (a == "--split" && i + 1 < argc) {
        split = std::stod(argv[++i]);
        split_set = true;
      }
      else if (a == "--holdout" && i + 1 < argc) {
        o.holdout = std::stod(argv[++i]);
        holdout_set = true;
      }
      else if (a == "--d-model" && i + 1 < argc) {
        d_model_ov = std::stoll(argv[++i]);
        d_model_set = true;
      }
      else if (a == "--n-layer" && i + 1 < argc) {
        n_layer_ov = std::stoll(argv[++i]);
        n_layer_set = true;
      }
      else if (a == "--n-head" && i + 1 < argc) {
        n_head_ov = std::stoll(argv[++i]);
        n_head_set = true;
      }
      else if (a == "--d-ff" && i + 1 < argc) {
        d_ff_ov = std::stoll(argv[++i]);
        d_ff_set = true;
      }
      else if (a == "--recency" && i + 1 < argc) {
        std::string r = argv[++i];
        o.recency_alibi = (r != "none" && r != "off" && r != "0");
      }
    }
    apply_charlm_preset(o);
    if (block_set) o.block = block_ov;
    if (d_model_set) o.d_model = d_model_ov;
    if (n_layer_set) o.n_layer = n_layer_ov;
    if (n_head_set) o.n_head = n_head_ov;
    if (d_ff_set) o.d_ff = d_ff_ov;
    if (!holdout_set && split_set) o.holdout = split;
    if (sub == "train") {
      auto r = run_charlm_train(o, log);
      if (!r) {
        std::cerr << r.error().message << '\n';
        return 1;
      }
      return 0;
    }
    if (sub == "generate") {
      auto r = run_charlm_generate(o, prompt, chars, log);
      if (!r) {
        std::cerr << r.error().message << '\n';
        return 1;
      }
      return 0;
    }
    if (sub == "export") {
      auto r = run_charlm_export_onnx(o, onnx_path, log);
      if (!r) {
        std::cerr << r.error().message << '\n';
        return 1;
      }
      return 0;
    }
    if (sub == "eval") {
      auto r = run_charlm_eval(o, split, log);
      if (!r) {
        std::cerr << r.error().message << '\n';
        return 1;
      }
      return 0;
    }
  }
  usage();
  return 1;
}
