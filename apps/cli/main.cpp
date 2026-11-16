#include "jobs.hpp"
#include "gyre/nn/tokenize.hpp"

#include <charconv>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

static void usage() {
  std::cout <<
      "gyre-cli — Gyre headless runner\n"
      "  gyre-cli ga [--gens 80] [--n 64] [--dim 64]\n"
      "  gyre-cli lm train --data data/shakespeare.txt [--preset medium|tiny|tinygpt|nanogpt]\n"
      "                   [--tokenizer bpe|chars|bytes|unigram] [--tok FILE.gyre.json] [--vocab-size 2000]\n"
      "                   [--holdout 0.1] [--steps 2000] [--batch 4] [--ckpt data/charlm.gyre]\n"
      "                   [--lr 3e-4] [--lr-start 1e-3] [--grad-clip 0] [--recency alibi|none]\n"
      "                   [--device cpu|vulkan] [--dropout 0] [--decay 0]\n"
      "                   [--prune 0.2] [--prune-gens 1] [--wire FILE.gyre] [--resume]\n"
      "                   [--shuffle-wire]\n"
      "                   [--hf-dir DIR] [--sp FILE.model]\n"
      "  gyre-cli lm generate --data data/shakespeare.txt --ckpt data/charlm.gyre\n"
      "                      [--preset medium|tiny|tinygpt] [--prompt \"To be\"] [--chars 200] [--temp 0.8]\n"
      "                      [--device cpu|vulkan]\n"
      "  gyre-cli lm export --ckpt data/charlm.gyre --onnx data/charlm.onnx [--preset medium]\n"
      "                    [--data data/shakespeare.txt]\n"
      "  gyre-cli lm import-gpt2 --hf-dir data/gpt2 --ckpt data/gpt2.gyre\n"
      "  gyre-cli lm import-llama --hf-dir data/smollm2-360m --ckpt data/smollm2-360m/smollm2-360m.gyre\n"
      "      (arch llama: lm eval adds [--window 2048] [--stride 1024]; lm generate adds\n"
      "       [--no-kv-cache] and --chars counts new tokens)\n"
      "  gyre-cli lm logits --ckpt LLAMA.gyre --prompt \"text\" --out logits.f32   (raw f32 [T, vocab])\n"
      "  gyre-cli lm eval --ckpt data/charlm.gyre --data data/shakespeare.txt [--preset tinygpt]\n"
      "                  [--split 0.1] [--device cpu|vulkan]\n"
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

namespace {

struct ArgError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Whole-string numeric parse: rejects junk ("abc", "12x"), empty, out-of-range, a '-' on an
// unsigned flag (stoul would wrap it to ~4e9) and non-finite floats.
template <class T>
T num(const std::string& flag, const char* text) {
  const std::string_view sv(text);
  T v{};
  const auto [end, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), v);
  if (sv.empty() || ec != std::errc{} || end != sv.data() + sv.size())
    throw ArgError("invalid value '" + std::string(sv) + "' for " + flag);
  if constexpr (std::is_floating_point_v<T>) {
    if (!std::isfinite(v)) throw ArgError("invalid value '" + std::string(sv) + "' for " + flag);
  }
  return v;
}

}  // namespace

static int run(int argc, char** argv);

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const ArgError& e) {
    std::cerr << "error: " << e.what() << "\n(see gyre-cli --help)\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}

static int run(int argc, char** argv) {
  using namespace gyre::app;
  // Value of the flag at argv[i]; advances i. A flag given last without a value is an error.
  auto next = [&](int& i) -> const char* {
    if (i + 1 >= argc) throw ArgError(std::string("missing value for ") + argv[i]);
    return argv[++i];
  };
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
      if (a == "--data") o.data = next(i);
      else if (a == "--tokenizer") o.tokenizer = next(i);
      else if (a == "--vocab-size") o.vocab_size = num<int>(a, next(i));
      else if (a == "--out") out = next(i);
      else if (a == "--tok") o.tok = next(i);
      else if (a == "--hf-dir") o.hf_dir = next(i);
      else if (a == "--sp") o.sp_model = next(i);
      else if (a == "--holdout") o.holdout = num<double>(a, next(i));
      else if (a == "--text") enc_text = next(i);
      else throw ArgError("unknown option " + a);
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
          go.weights = next(i);
      } else if (a == "--config")
        go.config = next(i);
      else if (a == "--weights")
        go.weights = next(i);
      else if (a == "--out")
        go.out = next(i);
      else if (a == "--file")
        go.file_substr = next(i);
      else if (a == "--preset")
        go.preset = next(i);
      else if (a == "--prompt")
        go.prompt = next(i);
      else if (a == "--max-new")
        go.max_new = num<int>(a, next(i));
      else if (a == "--tok")
        go.tok = next(i);
      else if (a == "--lora")
        go.lora = next(i);
      else if (a == "--codec")
        go.codec = next(i);
      else if (a == "--max-bytes")
        go.pack_max = num<std::uint64_t>(a, next(i));
      else if (a == "--max-tensors")
        go.max_tensors = num<int>(a, next(i));
      else if (a == "--chunk-bytes")
        go.chunk_bytes = num<std::uint64_t>(a, next(i));
      else if (!a.empty() && a[0] == '-')
        throw ArgError("unknown option " + a);
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
      if (a == "--max-tensors")
        max_t = num<int>(a, next(i));
      else if (!a.empty() && a[0] != '-')
        ckpt = a;
      else
        throw ArgError("unknown option " + a);
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
      if (a == "--gens") o.generations = num<std::uint32_t>(a, next(i));
      else if (a == "--n") o.n = num<std::uint32_t>(a, next(i));
      else if (a == "--dim") o.dim = num<std::uint32_t>(a, next(i));
      else throw ArgError("unknown option " + a);
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
    std::filesystem::path out_path = "data/logits.f32";
    double split = -1.0;
    bool split_set = false, holdout_set = false;
    bool block_set = false, d_model_set = false, n_layer_set = false, n_head_set = false, d_ff_set = false;
    std::uint32_t block_ov = 0;
    std::int64_t d_model_ov = 0, n_layer_ov = 0, n_head_ov = 0, d_ff_ov = 0;
    for (int i = 3; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--data") o.data = next(i);
      else if (a == "--ckpt") o.ckpt = next(i);
      else if (a == "--steps") o.steps = num<std::uint32_t>(a, next(i));
      else if (a == "--batch") {
        o.batch = num<std::uint32_t>(a, next(i));
        o.batch_set = true;
      }
      else if (a == "--block") {
        block_ov = num<std::uint32_t>(a, next(i));
        block_set = true;
      }
      else if (a == "--prompt") prompt = next(i);
      else if (a == "--chars") chars = num<int>(a, next(i));
      else if (a == "--temp") o.temperature = num<float>(a, next(i));
      else if (a == "--preset") o.preset = next(i);
      else if (a == "--ckpt-every")
        o.ckpt_every = num<std::uint32_t>(a, next(i));
      else if (a == "--lr") o.lr = num<float>(a, next(i));
      else if (a == "--lr-start") o.lr_start = num<float>(a, next(i));
      else if (a == "--lr-decay-steps")
        o.lr_decay_steps = num<std::uint32_t>(a, next(i));
      else if (a == "--grad-clip") o.grad_clip = num<float>(a, next(i));
      else if (a == "--onnx") onnx_path = next(i);
      else if (a == "--out") out_path = next(i);
      else if (a == "--tokenizer") {
        o.tokenizer = next(i);
        o.tokenizer_set = true;
      }
      else if (a == "--tok") o.tok = next(i);
      else if (a == "--hf-dir") o.hf_dir = next(i);
      else if (a == "--sp") o.sp_model = next(i);
      else if (a == "--vocab-size") o.vocab_size = num<int>(a, next(i));
      else if (a == "--split") {
        split = num<double>(a, next(i));
        split_set = true;
      }
      else if (a == "--holdout") {
        o.holdout = num<double>(a, next(i));
        holdout_set = true;
      }
      else if (a == "--d-model") {
        d_model_ov = num<std::int64_t>(a, next(i));
        d_model_set = true;
      }
      else if (a == "--n-layer") {
        n_layer_ov = num<std::int64_t>(a, next(i));
        n_layer_set = true;
      }
      else if (a == "--n-head") {
        n_head_ov = num<std::int64_t>(a, next(i));
        n_head_set = true;
      }
      else if (a == "--d-ff") {
        d_ff_ov = num<std::int64_t>(a, next(i));
        d_ff_set = true;
      }
      else if (a == "--recency") {
        std::string r = next(i);
        if (r == "alibi" || r == "on" || r == "1") o.recency_alibi = true;
        else if (r == "none" || r == "off" || r == "0") o.recency_alibi = false;
        else throw ArgError("--recency must be alibi|none (got '" + r + "')");
      }
      else if (a == "--device") o.device = next(i);
      else if (a == "--prune") o.prune = num<float>(a, next(i));
      else if (a == "--prune-gens")
        o.prune_gens = num<std::uint32_t>(a, next(i));
      else if (a == "--wire") o.wire = next(i);
      else if (a == "--resume") o.resume = true;
      else if (a == "--shuffle-wire") o.shuffle_wire = true;
      else if (a == "--dropout") o.dropout = num<float>(a, next(i));
      else if (a == "--decay") o.decay = num<float>(a, next(i));
      else if (a == "--window") o.eval_window = num<std::int64_t>(a, next(i));
      else if (a == "--stride") o.eval_stride = num<std::int64_t>(a, next(i));
      else if (a == "--no-kv-cache") o.kv_cache = false;
      else throw ArgError("unknown option " + a);
    }
    if (!is_known_preset(o.preset))
      throw ArgError("unknown --preset '" + o.preset + "' (tiny|medium|tinygpt|nanogpt)");
    if (!is_known_tokenizer(o.tokenizer))
      throw ArgError("unknown --tokenizer '" + o.tokenizer + "' (bpe|chars|bytes|unigram)");
    if (o.device != "cpu" && o.device != "vulkan" && o.device != "gpu")
      throw ArgError("unknown --device '" + o.device + "' (cpu|vulkan)");
    apply_charlm_preset(o);
    if (block_set) o.block = block_ov;
    if (d_model_set) o.d_model = d_model_ov;
    if (n_layer_set) o.n_layer = n_layer_ov;
    if (n_head_set) o.n_head = n_head_ov;
    if (d_ff_set) o.d_ff = d_ff_ov;
    if (!holdout_set && split_set) o.holdout = split;
    if (sub == "train") {
      if (o.batch == 0) throw ArgError("--batch must be >= 1");
      if (o.vocab_size < 1) throw ArgError("--vocab-size must be >= 1");
      if (o.prune < 0.f || o.prune >= 1.f) {
        std::cerr << "--prune must be in (0, 1)\n";
        return 1;
      }
      if (o.dropout < 0.f || o.dropout >= 1.f) {
        std::cerr << "--dropout must be in [0, 1)\n";
        return 1;
      }
      if (o.decay < 0.f) {
        std::cerr << "--decay must be >= 0\n";
        return 1;
      }
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
    if (sub == "import-gpt2") {
      auto r = run_gpt2_import(o.hf_dir, o.ckpt, log);
      if (!r) {
        std::cerr << r.error().message << '\n';
        return 1;
      }
      return 0;
    }
    if (sub == "logits") {
      auto r = run_llama_logits(o, prompt, out_path, log);
      if (!r) {
        std::cerr << r.error().message << '\n';
        return 1;
      }
      return 0;
    }
    if (sub == "import-llama") {
      auto r = run_llama_import(o.hf_dir, o.ckpt, log);
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
