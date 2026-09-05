#include "jobs.hpp"

#include "gyre/checkpoint.hpp"
#include "gyre/data.hpp"
#include "json_parse.hpp"
#include "gyre/export/onnx.hpp"
#include "gyre/io/compress_probe.hpp"
#include "gyre/io/safetensors.hpp"
#include "gyre/io/wpack.hpp"
#include "gyre/nn/tokenize.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <memory>
#include <sstream>
#include <string_view>

namespace gyre::app {

void apply_charlm_preset(CharLMOpts& o) {
  CharLMConfig c;
  if (o.preset == "tiny")
    c = CharLMConfig::tiny();
  else if (o.preset == "tinygpt")
    c = CharLMConfig::tinygpt();
  else if (o.preset == "nanogpt")
    c = CharLMConfig::nanogpt();
  else
    c = CharLMConfig::medium();
  o.block = static_cast<std::uint32_t>(c.block_size);
  o.n_layer = c.n_layer;
  o.n_head = c.n_head;
  o.d_model = c.d_model;
  o.d_ff = c.d_ff;
  if ((o.preset == "tinygpt" || o.preset == "nanogpt") && o.batch == 4) o.batch = 2;
}

namespace {

std::string progress_bar(std::string_view phase, std::uint64_t done, std::uint64_t total,
                         std::string_view extra = {}) {
  if (total == 0) total = 1;
  constexpr int w = 28;
  const int fill = static_cast<int>(w * done / total);
  std::string s = "\r";
  s += phase;
  s += " [";
  for (int i = 0; i < w; ++i) s += (i < fill) ? '#' : '-';
  s += "] ";
  s += std::to_string(done);
  s += '/';
  s += std::to_string(total);
  if (!extra.empty()) {
    s += "  ";
    s += extra;
  }
  return s;
}

std::string load_text(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

}  // namespace

Result<void> run_onemax(const GaOpts& opts, LogFn log) {
  auto dev = Device::cpu();
  if (!dev) return std::unexpected(dev.error());
  ga::Config cfg;
  cfg.n = opts.n;
  cfg.dim = opts.dim;
  cfg.elite = 2;
  cfg.gene = ga::GeneDType::u8;
  Rng rng(cfg.seed);
  auto pop = ga::random_population(cfg, *dev, rng);
  if (!pop) return std::unexpected(pop.error());
  float best = 0;
  for (std::uint32_t g = 0; g < opts.generations; ++g) {
    auto e = ga::evaluate(*pop, ga::onemax_fitness);
    if (!e) return e;
    auto f = pop->fitness.host_span<float>();
    if (!f) return std::unexpected(f.error());
    for (auto v : *f) best = std::max(best, v);
    if (log)
      log(progress_bar("ga", g + 1, opts.generations,
                       "best=" + std::to_string(best) + "/" + std::to_string(cfg.dim)));
    auto nxt = ga::step(*pop, cfg, rng);
    if (!nxt) return std::unexpected(nxt.error());
    *pop = std::move(*nxt);
  }
  if (log) log("OneMax done best=" + std::to_string(best));
  return {};
}

static GyreDoc make_charlm_doc(const CharLMOpts& o, std::int64_t vocab, const Tokenizer& tok) {
  nlohmann::json cfg;
  cfg["preset"] = o.preset;
  cfg["n_layer"] = o.n_layer;
  cfg["n_head"] = o.n_head;
  cfg["d_model"] = o.d_model;
  cfg["d_ff"] = o.d_ff;
  cfg["block_size"] = o.block;
  cfg["vocab_size"] = vocab;
  cfg["holdout"] = o.holdout;
  cfg["recency"] = o.recency_alibi ? "alibi" : "none";
  GyreDoc d;
  d.arch = "char-lm";
  d.config_json = cfg.dump();
  if (auto tj = parse_json(tok.to_json()); tj && tj->contains("tokenizer")) {
    d.tokenizer_json = (*tj)["tokenizer"].dump();
  }
  return d;
}

static CharLMConfig charlm_cfg_from_doc(const GyreDoc& d, const CharLMOpts& opts, std::int64_t vocab) {
  CharLMConfig cfg;
  cfg.vocab = vocab;
  cfg.block_size = opts.block;
  cfg.n_layer = opts.n_layer;
  cfg.n_head = opts.n_head;
  cfg.d_model = opts.d_model;
  cfg.d_ff = opts.d_ff;
  cfg.recency_alibi = d.recency_alibi();
  if (auto c = parse_json(d.config_json); c) {
    cfg.block_size = c->value("block_size", cfg.block_size);
    cfg.n_layer = c->value("n_layer", cfg.n_layer);
    cfg.n_head = c->value("n_head", cfg.n_head);
    cfg.d_model = c->value("d_model", cfg.d_model);
    cfg.d_ff = c->value("d_ff", cfg.d_ff);
  }
  return cfg;
}

static Result<std::unique_ptr<Tokenizer>> load_existing_tok(const CharLMOpts& opts) {
  if (!opts.tok.empty()) return Tokenizer::load(opts.tok);
  if (!opts.hf_dir.empty()) return Tokenizer::load_huggingface(opts.hf_dir);
  if (!opts.sp_model.empty()) return Tokenizer::load_sentencepiece(opts.sp_model);
  return std::unexpected(make_error(Errc::io, "no tokenizer path"));
}

static Result<std::unique_ptr<Tokenizer>> make_tokenizer(const CharLMOpts& opts, std::string_view train_text,
                                                         LogFn log) {
  if (!opts.tok.empty() || !opts.hf_dir.empty() || !opts.sp_model.empty()) {
    if (log) log("loading tokenizer …");
    return load_existing_tok(opts);
  }
  if (opts.tokenizer == "unigram") {
    if (log) log("training unigram vocab=" + std::to_string(opts.vocab_size) + " …");
    auto prog = [&](int d, int t) {
      if (log) log(progress_bar("tok", static_cast<std::uint64_t>(d), static_cast<std::uint64_t>(t)));
    };
    return Tokenizer::train_unigram(train_text, opts.vocab_size, prog);
  }
  if (opts.tokenizer == "bytes" || opts.preset == "tinygpt") return Tokenizer::bytes();
  if (opts.tokenizer == "chars") return Tokenizer::chars_from_text(train_text);
  if (log) log("training BPE vocab=" + std::to_string(opts.vocab_size) + " …");
  auto prog = [&](int d, int t) {
    if (log) log(progress_bar("bpe", static_cast<std::uint64_t>(d), static_cast<std::uint64_t>(t)));
  };
  return Tokenizer::train_bpe(train_text, opts.vocab_size, prog);
}

Result<std::string> run_charlm_train(const CharLMOpts& opts, LogFn log) {
  auto dev = Device::cpu();
  if (!dev) return std::unexpected(dev.error());
  auto text = load_text(opts.data);
  if (text.size() < 200) {
    return std::unexpected(make_error(Errc::io, "need a text file at " + opts.data.string()));
  }
  double holdout = opts.holdout;
  if (holdout < 0.0) holdout = 0.0;
  if (holdout >= 1.0) holdout = 0.1;
  const auto split_at =
      static_cast<std::size_t>((1.0 - holdout) * static_cast<double>(text.size()));
  if (holdout > 0.0 && split_at < 200) {
    return std::unexpected(make_error(Errc::io, "train split too short after holdout"));
  }
  std::string train_text = holdout > 0.0 ? text.substr(0, split_at) : text;
  if (log)
    log("loaded " + std::to_string(text.size()) + " bytes  train=" +
        std::to_string(train_text.size()) + " holdout=" + std::to_string(holdout) +
        " val_chars=" + std::to_string(text.size() - train_text.size()));

  auto tok = make_tokenizer(opts, train_text, log);
  if (!tok) return std::unexpected(tok.error());
  if (log) log("encoding train split …");
  auto bprog = [&](int d, int t) {
    if (log) log(progress_bar("tok", static_cast<std::uint64_t>(d), static_cast<std::uint64_t>(t)));
  };
  auto ids = (*tok)->encode(train_text, bprog);
  if (!ids) return std::unexpected(ids.error());
  if (log)
    log(std::string((*tok)->model_name()) + " vocab=" + std::to_string((*tok)->vocab_size()) +
        " seq=" + std::to_string(ids->size()));
  auto data = CharDataset::from_ids(*ids, *dev);
  if (!data) return std::unexpected(data.error());
  Rng rng(1);
  CharLMConfig cfg;
  cfg.vocab = (*tok)->vocab_size();
  cfg.block_size = opts.block;
  cfg.n_layer = opts.n_layer;
  cfg.n_head = opts.n_head;
  cfg.d_model = opts.d_model;
  cfg.d_ff = opts.d_ff;
  cfg.recency_alibi = opts.recency_alibi;
  auto model = CharLM::create(cfg, *dev, rng);
  if (!model) return std::unexpected(model.error());
  std::int64_t n = 0;
  for (auto& p : model->parameters()) n += p.value.numel();
  if (log)
    log("preset " + opts.preset + " tok=" + std::string((*tok)->model_name()) + " params " +
        std::to_string(n) + " vocab " +
        std::to_string(cfg.vocab) + " T=" + std::to_string(opts.block) + " L=" + std::to_string(opts.n_layer) +
        " d=" + std::to_string(opts.d_model) + (cfg.recency_alibi ? " recency=alibi" : " recency=none"));

  auto doc = make_charlm_doc(opts, cfg.vocab, **tok);
  auto meta_json = doc.to_json();
  TrainConfig tc;
  tc.steps = opts.steps;
  tc.batch = opts.batch;
  tc.block = opts.block;
  tc.lr = opts.lr;
  tc.lr_start = opts.lr_start;
  tc.lr_decay_steps = opts.lr_decay_steps;
  tc.log_every = opts.log_every;
  tc.ckpt_every = opts.ckpt_every;
  tc.ckpt_json = meta_json;
  tc.param_names = model->param_names();
  if (!opts.ckpt.empty()) {
    auto dir = opts.ckpt.parent_path();
    std::filesystem::create_directories(dir.empty() ? "." : dir);
    tc.ckpt_dir = dir.empty() ? "." : dir;
    tc.ckpt_path = opts.ckpt;
  }
  TrainLoop loop;
  auto r = loop.run(
      *model, *data, tc, *dev,
      [&](const Metrics& m) {
        if (log)
          log("step " + std::to_string(m.step) + " loss " + std::to_string(m.loss) + " lr " +
              std::to_string(m.lr));
      },
      [&](const Metrics& m) {
        if (log)
          log(progress_bar("train", m.step, opts.steps,
                           "loss=" + std::to_string(m.loss) + " lr=" + std::to_string(m.lr)));
      });
  if (!r) return std::unexpected(r.error());

  if (!opts.ckpt.empty()) {
    CheckpointMeta meta{1, opts.steps, meta_json, model->param_names()};
    auto s = save_gyre1(opts.ckpt, model->parameters(), nullptr, meta);
    if (!s) return std::unexpected(s.error());
    if (log) log("saved " + opts.ckpt.string());
    auto json_path = opts.ckpt;
    json_path.replace_extension(".gyre.json");
    std::uint64_t nbytes = 0;
    for (auto& p : model->parameters()) nbytes += static_cast<std::uint64_t>(p.value.nbytes());
    auto js = save_gyre_json(json_path, model->parameters(), nullptr, doc, model->param_names(),
                             nbytes <= kGyreJsonDataLimit);
    if (js && log) log("saved " + json_path.string());
  }

  std::vector<std::int32_t> prefix = {ids->front()};
  Rng sample_rng(2);
  auto gen = model->generate(prefix, 80, *dev, &sample_rng, opts.temperature);
  if (!gen) return std::unexpected(gen.error());
  std::string sample = (*tok)->decode(*gen);
  if (log) log(sample);
  return sample;
}

Result<std::string> run_charlm_generate(const CharLMOpts& opts_in, std::string prompt, int max_new,
                                        LogFn log) {
  CharLMOpts opts = opts_in;
  apply_charlm_preset(opts);
  auto dev = Device::cpu();
  if (!dev) return std::unexpected(dev.error());
  auto peek = peek_gyre(opts.ckpt);
  if (!peek) return std::unexpected(peek.error());
  std::unique_ptr<Tokenizer> tok;
  if (auto t = Tokenizer::from_json(peek->to_json()); t) {
    tok = std::move(*t);
  } else if (!opts.tok.empty() || !opts.hf_dir.empty() || !opts.sp_model.empty()) {
    auto t2 = load_existing_tok(opts);
    if (!t2) return std::unexpected(t2.error());
    tok = std::move(*t2);
  } else {
    auto text = load_text(opts.data);
    auto t = (opts.preset == "tinygpt") ? Tokenizer::bytes() : Tokenizer::chars_from_text(text);
    if (!t) return std::unexpected(t.error());
    tok = std::move(*t);
  }
  auto cfg = charlm_cfg_from_doc(*peek, opts, tok->vocab_size());
  Rng rng(1);
  auto model = CharLM::create(cfg, *dev, rng);
  if (!model) return std::unexpected(model.error());
  CheckpointMeta meta;
  auto ld = load_gyre1(opts.ckpt, model->parameters(), nullptr, meta);
  if (!ld) return std::unexpected(ld.error());
  if (log) log("loaded " + opts.ckpt.string() + " vocab " + std::to_string(cfg.vocab));
  if (prompt.empty()) prompt = " ";
  auto ids = tok->encode(prompt);
  if (!ids) return std::unexpected(ids.error());
  Rng sample_rng(2);
  auto gen = model->generate(*ids, max_new, *dev, &sample_rng, opts.temperature);
  if (!gen) return std::unexpected(gen.error());
  auto out = tok->decode(*gen);
  if (log) log(out);
  return out;
}

Result<void> run_charlm_export_onnx(const CharLMOpts& opts_in, const std::filesystem::path& onnx_path,
                                   LogFn log) {
  CharLMOpts opts = opts_in;
  apply_charlm_preset(opts);
  auto dev = Device::cpu();
  if (!dev) return std::unexpected(dev.error());
  CharLMConfig cfg;
  cfg.vocab = (opts.preset == "tinygpt") ? 256 : 0;
  if (cfg.vocab == 0) {
    auto peek = peek_gyre1(opts.ckpt);
    if (peek) {
      auto t = Tokenizer::from_json(peek->json);
      if (t) cfg.vocab = (*t)->vocab_size();
    }
    if (cfg.vocab == 0) {
      auto text = load_text(opts.data);
      auto tok = Tokenizer::chars_from_text(text);
      if (!tok) return std::unexpected(tok.error());
      cfg.vocab = (*tok)->vocab_size();
    }
  }
  cfg.block_size = opts.block;
  cfg.n_layer = opts.n_layer;
  cfg.n_head = opts.n_head;
  cfg.d_model = opts.d_model;
  cfg.d_ff = opts.d_ff;
  cfg.recency_alibi = opts.recency_alibi;
  Rng rng(1);
  auto model = CharLM::create(cfg, *dev, rng);
  if (!model) return std::unexpected(model.error());
  CheckpointMeta meta;
  auto ld = load_gyre1(opts.ckpt, model->parameters(), nullptr, meta);
  if (!ld) return std::unexpected(ld.error());
  auto ex = export_charlm_onnx(*model, onnx_path);
  if (!ex) return ex;
  if (log) log("wrote " + onnx_path.string());
  return {};
}

Result<EvalReport> run_charlm_eval(const CharLMOpts& opts_in, double split, LogFn log) {
  CharLMOpts opts = opts_in;
  apply_charlm_preset(opts);
  auto dev = Device::cpu();
  if (!dev) return std::unexpected(dev.error());
  auto peek = peek_gyre(opts.ckpt);
  if (!peek) return std::unexpected(peek.error());
  if (split <= 0.0 || split >= 1.0) {
    split = peek->holdout();
    if (split <= 0.0 || split >= 1.0) split = 0.1;
  }
  std::unique_ptr<Tokenizer> tok;
  if (auto t = Tokenizer::from_json(peek->to_json()); t) {
    tok = std::move(*t);
  } else {
    auto text0 = load_text(opts.data);
    auto t2 = (opts.preset == "tinygpt") ? Tokenizer::bytes() : Tokenizer::chars_from_text(text0);
    if (!t2) return std::unexpected(t2.error());
    tok = std::move(*t2);
  }
  auto cfg = charlm_cfg_from_doc(*peek, opts, tok->vocab_size());
  Rng rng(1);
  auto model = CharLM::create(cfg, *dev, rng);
  if (!model) return std::unexpected(model.error());
  CheckpointMeta meta;
  auto ld = load_gyre1(opts.ckpt, model->parameters(), nullptr, meta);
  if (!ld) return std::unexpected(ld.error());

  auto text = load_text(opts.data);
  if (text.size() < 64) return std::unexpected(make_error(Errc::io, "data too short to eval"));
  const auto split_at = static_cast<std::size_t>((1.0 - split) * static_cast<double>(text.size()));
  std::string val = text.substr(split_at);
  auto ids = tok->encode(val);
  if (!ids) return std::unexpected(ids.error());
  const auto T = cfg.block_size;
  if (static_cast<std::int64_t>(ids->size()) < T + 1) {
    return std::unexpected(make_error(Errc::invalid_shape, "val split shorter than block"));
  }

  double nll = 0;
  std::uint64_t n_pred = 0, n_win = 0;
  const std::int64_t n = static_cast<std::int64_t>(ids->size());
  for (std::int64_t start = 0; start + T + 1 <= n; start += T) {
    std::int64_t xsh[2] = {1, T};
    std::vector<std::byte> xb(static_cast<std::size_t>(T) * 4), yb(static_cast<std::size_t>(T) * 4);
    std::memcpy(xb.data(), ids->data() + start, static_cast<std::size_t>(T) * 4);
    std::memcpy(yb.data(), ids->data() + start + 1, static_cast<std::size_t>(T) * 4);
    auto x = Tensor::from_host(xb, xsh, DType::i32, *dev);
    auto y = Tensor::from_host(yb, xsh, DType::i32, *dev);
    if (!x || !y) return std::unexpected(x ? y.error() : x.error());
    ForwardCtx ctx;
    ctx.train = false;
    auto logits = model->forward(*x, ctx);
    if (!logits) return std::unexpected(logits.error());
    auto loss = softmax_cross_entropy(*logits, *y);
    if (!loss) return std::unexpected(loss.error());
    auto lv = loss->value.host_span<float>();
    if (!lv) return std::unexpected(lv.error());
    nll += static_cast<double>((*lv)[0]) * static_cast<double>(T);
    n_pred += static_cast<std::uint64_t>(T);
    ++n_win;
    if (log && n_win % 4 == 0)
      log(progress_bar("eval", static_cast<std::uint64_t>(start + T), static_cast<std::uint64_t>(n)));
  }
  EvalReport r;
  r.n_pred = n_pred;
  r.n_chars = val.size();
  r.n_windows = n_win;
  r.vocab = cfg.vocab;
  r.block = T;
  r.n_params = 0;
  for (auto& p : model->parameters()) r.n_params += static_cast<std::uint64_t>(p.value.numel());
  r.nats_per_token = nll / static_cast<double>(n_pred);
  r.nats_per_char = nll / static_cast<double>(r.n_chars);
  r.bpc = r.nats_per_char / 0.6931471805599453;  // ln 2
  r.chars_per_token = static_cast<double>(r.n_chars) / static_cast<double>(n_pred);
  if (log) {
    log("eval split=" + std::to_string(split) + " val_chars=" + std::to_string(r.n_chars) +
        " windows=" + std::to_string(r.n_windows));
    log("params " + std::to_string(r.n_params) + " vocab " + std::to_string(r.vocab) + " T=" +
        std::to_string(r.block));
    log("nats/token " + std::to_string(r.nats_per_token) + "  nats/char " +
        std::to_string(r.nats_per_char) + "  BPC " + std::to_string(r.bpc) + "  chars/tok " +
        std::to_string(r.chars_per_token));
    log("compare: nanoGPT shakespeare-char ~10.7M  val ~1.47 nats/char  (~2.12 BPC) on last 10% of the same file");
  }
  return r;
}

Result<void> run_tok_train(const CharLMOpts& opts, const std::filesystem::path& out, LogFn log) {
  auto text = load_text(opts.data);
  if (text.size() < 8) return std::unexpected(make_error(Errc::io, "need --data"));
  double holdout = opts.holdout;
  if (holdout < 0.0) holdout = 0.0;
  if (holdout >= 1.0) holdout = 0.1;
  const auto split_at =
      static_cast<std::size_t>((1.0 - holdout) * static_cast<double>(text.size()));
  std::string train_text = holdout > 0.0 ? text.substr(0, split_at) : text;
  auto tok = make_tokenizer(opts, train_text, log);
  if (!tok) return std::unexpected(tok.error());
  auto path = out;
  if (path.empty()) path = "data/tokenizer.gyre.json";
  auto s = (*tok)->save(path);
  if (!s) return s;
  if (log) log("saved " + path.string() + " vocab=" + std::to_string((*tok)->vocab_size()));
  return {};
}

Result<void> run_tok_export_hf(const std::filesystem::path& tok, const std::filesystem::path& hf_dir,
                               LogFn log) {
  auto t = Tokenizer::load(tok);
  if (!t) return std::unexpected(t.error());
  auto s = (*t)->save_huggingface(hf_dir);
  if (!s) return s;
  if (log) log("wrote HF tokenizer in " + hf_dir.string());
  return {};
}

Result<void> run_tok_import(const CharLMOpts& opts, const std::filesystem::path& out, LogFn log) {
  auto t = load_existing_tok(opts);
  if (!t) {
    if (!opts.hf_dir.empty()) t = Tokenizer::load_huggingface(opts.hf_dir);
    else if (!opts.sp_model.empty()) t = Tokenizer::load_sentencepiece(opts.sp_model);
    else if (!opts.tok.empty()) t = Tokenizer::load(opts.tok);
  }
  if (!t) return std::unexpected(t.error());
  auto path = out.empty() ? std::filesystem::path("data/imported.gyre.json") : out;
  auto s = (*t)->save(path);
  if (!s) return s;
  if (log) log("imported vocab=" + std::to_string((*t)->vocab_size()) + " -> " + path.string());
  return {};
}

Result<void> run_grok_info(const GrokCliOpts& o, LogFn log) {
  auto c = GrokConfig::from_file(o.config);
  if (!c) {
    c = GrokConfig::full();
    if (log) log("(no file " + o.config.string() + ", using built-in full())");
  }
  if (log) {
    log(c->to_string());
    log(std::string("create_ok=") + (c->fits_in_ram_create() ? "yes" : "no (tiny only until bind)"));
  }
  return {};
}

Result<void> run_grok_inspect(const GrokCliOpts& o, LogFn log) {
  auto wdir = o.weights.empty() ? std::filesystem::path("data/grok2") : o.weights;
  if (!std::filesystem::exists(wdir)) return std::unexpected(make_error(Errc::io, "missing " + wdir.string()));
  for (auto& ent : std::filesystem::directory_iterator(wdir)) {
    if (!ent.is_regular_file() || ent.path().extension() != ".safetensors") continue;
    auto f = safetensors_open(ent.path());
    if (!f) {
      if (log) log(ent.path().filename().string() + ": " + f.error().message);
      continue;
    }
    if (log) log(ent.path().filename().string() + " tensors=" + std::to_string(f->tensors.size()));
  }
  return {};
}

Result<void> run_grok_compress_probe(const GrokCliOpts& o, LogFn log) {
  auto wdir = o.weights.empty() ? std::filesystem::path("data/grok2") : o.weights;
  CompressProbeOpts po;
  po.file_substr = o.file_substr;
  po.max_tensors = o.max_tensors;
  po.chunk_bytes = o.chunk_bytes;
  auto rows = compress_probe_dir(wdir, po);
  if (!rows) return std::unexpected(rows.error());
  auto js = compress_probe_json(*rows);
  if (!o.out.empty()) {
    std::ofstream f(o.out, std::ios::binary);
    f << js;
    if (log) log("wrote " + o.out.string() + " rows=" + std::to_string(rows->size()));
  } else if (log) {
    log(js);
  }
  return {};
}

Result<void> run_grok_pack(const GrokCliOpts& o, LogFn log) {
  auto wdir = o.weights.empty() ? std::filesystem::path("data/grok2") : o.weights;
  auto out = o.out.empty() ? std::filesystem::path("data/grok2/norm.gyre.wpack") : o.out;
  WpackFile all;
  all.source = wdir.string();
  for (auto& ent : std::filesystem::directory_iterator(wdir)) {
    if (!ent.is_regular_file() || ent.path().extension() != ".safetensors") continue;
    auto fn = ent.path().filename().string();
    if (!o.file_substr.empty() && fn.find(o.file_substr) == std::string::npos) continue;
    auto st = safetensors_open(ent.path());
    if (!st) return std::unexpected(st.error());
    auto part = wpack_from_safetensors(*st, o.pack_max);
    if (!part) return std::unexpected(part.error());
    for (auto& t : part->tensors) all.tensors.push_back(std::move(t));
  }
  auto s = wpack_save(out, all);
  if (!s) return s;
  if (log) log("packed " + std::to_string(all.tensors.size()) + " tensors -> " + out.string());
  return {};
}

Result<void> run_grok_save(const GrokCliOpts& o, LogFn log) {
  auto out = o.out.empty() ? std::filesystem::path("data/grok-mini.gyre") : o.out;
  auto c = o.preset == "tiny" ? GrokConfig::tiny() : GrokConfig::mini();
  auto d = Device::cpu();
  if (!d) return std::unexpected(d.error());
  Rng rng(1);
  auto m = GrokLM::create(c, *d, rng);
  if (!m) return std::unexpected(m.error());
  auto codec = gyre_codec_from_name(o.codec);
  auto s = m->save_gyre(out, codec);
  if (!s) return s;
  if (log) log("saved " + c.to_string() + " -> " + out.string() + " codec=" + gyre_codec_name(codec));
  return {};
}

Result<void> run_grok_gen(const GrokCliOpts& o, LogFn log) {
  auto wdir = o.weights.empty() ? (o.out.empty() ? std::filesystem::path("data/grok-mini.gyre") : o.out)
                                : o.weights;
  auto d = Device::cpu();
  if (!d) return std::unexpected(d.error());
  auto m = GrokLM::load_weights(wdir, *d);
  if (!m) return std::unexpected(m.error());
  if (!o.lora.empty()) {
    auto lr = m->load_lora(o.lora, *d);
    if (!lr) return lr;
  }
  std::vector<std::int32_t> ids;
  if (!o.tok.empty()) {
    auto t = Tokenizer::load(o.tok);
    if (!t) return std::unexpected(t.error());
    auto e = (*t)->encode(o.prompt);
    if (!e) return std::unexpected(e.error());
    auto g = m->generate(*e, o.max_new, *d, nullptr, 0.f);
    if (!g) return std::unexpected(g.error());
    if (log) log((*t)->decode(*g));
    return {};
  }
  ids = {1, 2};
  auto g = m->generate(ids, o.max_new, *d, nullptr, 0.f);
  if (!g) return std::unexpected(g.error());
  if (log) {
    std::string s;
    for (std::size_t i = 0; i < g->size(); ++i) {
      if (i) s += ' ';
      s += std::to_string((*g)[i]);
    }
    log(s);
  }
  return {};
}

Result<void> run_grok_import(const GrokCliOpts& o, LogFn log) {
  auto wdir = o.weights.empty() ? std::filesystem::path("data/grok2") : o.weights;
  auto out = o.out.empty() ? std::filesystem::path("data/grok.gyre") : o.out;
  auto d = Device::cpu();
  if (!d) return std::unexpected(d.error());
  auto codec = gyre_codec_from_name(o.codec);
  auto s = import_safetensors_to_gyre(wdir, out, *d, codec);
  if (!s) return s;
  if (log) log("imported " + wdir.string() + " -> " + out.string() + " codec=" + gyre_codec_name(codec));
  return {};
}

Result<void> run_ckpt_probe(const std::filesystem::path& ckpt, int max_tensors, LogFn log) {
  auto f = GyreFile::open(ckpt);
  if (!f) return std::unexpected(f.error());
  auto rows = probe_gyre_codecs(*f, max_tensors);
  if (!rows) return std::unexpected(rows.error());
  for (auto& r : *rows) {
    if (!log) continue;
    log(r.name + " raw=" + std::to_string(r.raw_bytes) + " alp=" + std::to_string(r.alp_bytes) +
        (r.alp_ok ? " ok" : " FAIL") + " zfp=" +
        (r.zfp_bytes ? std::to_string(r.zfp_bytes) : std::string("n/a")) +
        (r.zfp_ok ? " ok" : ""));
  }
  return {};
}

}  // namespace gyre::app
