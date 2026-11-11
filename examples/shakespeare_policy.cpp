// Next-character prediction on text, expressed with the *RL stack* (no CharLM involved):
//
//   observation at step t = the last K characters as a SET of entities
//                           (feature row = one-hot char | one-hot recency slot)
//   action               = the next character (one head with V options)
//   training             = ImitationTrainer (cross-entropy on demonstrated actions)
//   memory               = the GRU core carried across the 128-step segment
//   deployment           = export_policy_flat -> gyre-infer (also used here to sample text)
//
// It exists to compare the new stack against CharLM on the same data and the same
// held-out slice (last --holdout fraction of the file), in nats/char.
//
//   gyre-shakespeare-policy --data data/shakespeare.txt --window 1 --hidden 256 --steps 3000
#include "gyre/gyre.hpp"
#include "gyre_infer/policy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

using namespace gyre;

namespace {

struct Opt {
  std::string data = "data/shakespeare.txt";
  std::string out = "shakespeare-policy";
  int window = 1;        // K: how many recent characters are shown as entities
  int hidden = 256;      // GRU state size
  int d_model = 64;      // set-encoder width
  int layers = 1;        // set-encoder blocks
  int steps = 3000;      // optimiser steps
  int streams = 32;      // E: parallel segments per step
  int seg = 128;         // T: segment length (BPTT length, matches CharLM block)
  float lr = 2e-3f;
  float lr_end = 2e-4f;
  float holdout = 0.1f;
  float max_minutes = 0.f;  // > 0: stop training after this long
  int eval_every = 500;
  std::uint64_t seed = 1;
};

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "error: %s\n", m.c_str());
  std::exit(1);
}
template <class T>
T ok(Result<T> r, const char* what) {
  if (!r) die(std::string(what) + ": " + r.error().message);
  return std::move(*r);
}
void ok(Result<void> r, const char* what) {
  if (!r) die(std::string(what) + ": " + r.error().message);
}

Tensor mk(const std::vector<float>& v, std::vector<std::int64_t> sh) {
  auto d = Device::cpu();
  return ok(Tensor::from_host(std::as_bytes(std::span<const float>(v)), sh, DType::f32, *d), "tensor");
}
Tensor mk_u8(const std::vector<std::uint8_t>& v, std::vector<std::int64_t> sh) {
  auto d = Device::cpu();
  return ok(Tensor::from_host(std::as_bytes(std::span<const std::uint8_t>(v)), sh, DType::u8, *d), "tensor");
}
Tensor mk_i32(const std::vector<std::int32_t>& v, std::vector<std::int64_t> sh) {
  auto d = Device::cpu();
  return ok(Tensor::from_host(std::as_bytes(std::span<const std::int32_t>(v)), sh, DType::i32, *d), "tensor");
}

struct Corpus {
  std::vector<std::uint8_t> ids;  // vocabulary indices
  std::vector<unsigned char> chars;  // id -> byte
  std::size_t train_end{0};
  int V() const { return static_cast<int>(chars.size()); }
};

Corpus load_corpus(const Opt& o) {
  std::ifstream f(o.data, std::ios::binary);
  if (!f) die("cannot open " + o.data);
  std::string raw((std::istreambuf_iterator<char>(f)), {});
  Corpus c;
  bool seen[256] = {false};
  for (unsigned char ch : raw) seen[ch] = true;
  int map[256];
  for (int i = 0; i < 256; ++i) {
    map[i] = -1;
    if (seen[i]) {
      map[i] = static_cast<int>(c.chars.size());
      c.chars.push_back(static_cast<unsigned char>(i));
    }
  }
  c.ids.reserve(raw.size());
  for (unsigned char ch : raw) c.ids.push_back(static_cast<std::uint8_t>(map[ch]));
  c.train_end = static_cast<std::size_t>(double(raw.size()) * (1.0 - o.holdout));
  return c;
}

PolicyConfig make_config(const Opt& o, int V) {
  PolicyConfig c;
  c.encoder.feat_dim = V + o.window;
  c.encoder.global_dim = 1;
  c.encoder.max_entities = o.window;
  c.encoder.d_model = o.d_model;
  c.encoder.n_layer = o.layers;
  c.encoder.n_head = 4;
  c.encoder.d_ff = 2 * o.d_model;
  c.encoder.out_dim = o.hidden;
  c.head_sizes = {V};
  c.gru_hidden = o.hidden;
  return c;
}

// Fill `buf` (E streams x T steps) with demonstrations: at step t the entities are the last K
// characters (current first); the expert action is the next character.
void fill(RolloutBuffer& buf, const Corpus& c, const Opt& o, const std::vector<std::size_t>& starts) {
  const int E = static_cast<int>(starts.size()), T = o.seg, K = o.window, V = c.V(), F = V + K, H = o.hidden;
  buf.clear();
  const auto legal = mk_u8(std::vector<std::uint8_t>(static_cast<std::size_t>(E) * V, 1), {E, V});
  const auto active = mk_u8(std::vector<std::uint8_t>(static_cast<std::size_t>(E), 1), {E, 1});
  const auto h0 = mk(std::vector<float>(static_cast<std::size_t>(E) * H, 0.f), {E, H});
  const auto glob = mk(std::vector<float>(static_cast<std::size_t>(E), 0.f), {E, 1});
  for (int t = 0; t < T; ++t) {
    std::vector<float> ent(static_cast<std::size_t>(E) * K * F, 0.f);
    std::vector<std::uint8_t> mask(static_cast<std::size_t>(E) * K, 0);
    std::vector<std::int32_t> act(static_cast<std::size_t>(E));
    for (int e = 0; e < E; ++e) {
      for (int j = 0; j < K && t - j >= 0; ++j) {
        const std::size_t row = (static_cast<std::size_t>(e) * K + j);
        ent[row * F + c.ids[starts[e] + t - j]] = 1.f;
        ent[row * F + V + j] = 1.f;
        mask[row] = 1;
      }
      act[e] = c.ids[starts[e] + t + 1];
    }
    Obs obs{mk(ent, {E, K, F}), mk_u8(mask, {E, K}), glob};
    auto rec = ok(make_demo_step(obs, legal, active, mk_i32(act, {E, 1}),
                                 std::vector<std::uint8_t>(static_cast<std::size_t>(E), t == 0 ? 1 : 0), h0),
                  "demo step");
    ok(buf.push(rec), "push");
  }
}

std::vector<std::vector<std::size_t>> heldout_chunks(const Corpus& c, const Opt& o, int E) {
  std::vector<std::size_t> all;
  for (std::size_t s = c.train_end; s + o.seg + 1 <= c.ids.size(); s += o.seg) all.push_back(s);
  std::vector<std::vector<std::size_t>> out;
  for (std::size_t i = 0; i + E <= all.size(); i += E) out.emplace_back(all.begin() + i, all.begin() + i + E);
  return out;
}

double eval_heldout(ImitationTrainer& tr, RolloutBuffer& buf, const Corpus& c, const Opt& o, int E,
                    double* acc = nullptr) {
  double loss = 0, a = 0;
  const auto chunks = heldout_chunks(c, o, E);
  for (const auto& starts : chunks) {
    fill(buf, c, o, starts);
    auto st = ok(tr.evaluate(buf), "evaluate");
    loss += st.loss;
    a += st.accuracy;
  }
  if (acc) *acc = a / double(chunks.size());
  return loss / double(chunks.size());
}

}  // namespace

int main(int argc, char** argv) {
  Opt o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) die("missing value for " + a);
      return argv[++i];
    };
    if (a == "--data") o.data = next();
    else if (a == "--out") o.out = next();
    else if (a == "--window") o.window = std::atoi(next());
    else if (a == "--hidden") o.hidden = std::atoi(next());
    else if (a == "--d-model") o.d_model = std::atoi(next());
    else if (a == "--layers") o.layers = std::atoi(next());
    else if (a == "--steps") o.steps = std::atoi(next());
    else if (a == "--streams") o.streams = std::atoi(next());
    else if (a == "--seg") o.seg = std::atoi(next());
    else if (a == "--lr") o.lr = static_cast<float>(std::atof(next()));
    else if (a == "--lr-end") o.lr_end = static_cast<float>(std::atof(next()));
    else if (a == "--holdout") o.holdout = static_cast<float>(std::atof(next()));
    else if (a == "--max-minutes") o.max_minutes = static_cast<float>(std::atof(next()));
    else if (a == "--eval-every") o.eval_every = std::atoi(next());
    else if (a == "--seed") o.seed = static_cast<std::uint64_t>(std::atoll(next()));
    else die("unknown flag " + a);
  }
  auto corpus = load_corpus(o);
  const int V = corpus.V();
  std::printf("corpus: %zu chars, V=%d, train %zu / held-out %zu\n", corpus.ids.size(), V, corpus.train_end,
              corpus.ids.size() - corpus.train_end);

  Rng rng(o.seed);
  auto dev = ok(Device::cpu(), "device");
  auto cfg = make_config(o, V);
  auto net = ok(PolicyNet::create(cfg, dev, rng), "net");
  std::int64_t nparams = 0;
  for (auto& p : net.parameters()) nparams += p.value.numel();
  std::printf("PolicyNet: window=%d d_model=%d layers=%d gru=%d -> %lld params\n", o.window, o.d_model, o.layers,
              o.hidden, static_cast<long long>(nparams));

  RolloutSpec spec;
  spec.steps = o.seg;
  spec.envs = o.streams;
  spec.max_entities = o.window;
  spec.feat_dim = V + o.window;
  spec.global_dim = 1;
  spec.hidden_dim = o.hidden;
  spec.layout = net.layout();
  auto buf = ok(RolloutBuffer::create(spec), "buffer");
  RolloutSpec espec = spec;
  espec.envs = 64;
  auto ebuf = ok(RolloutBuffer::create(espec), "eval buffer");

  ImitationConfig ic;
  ic.lr = o.lr;
  ic.minibatches = 1;
  ic.max_grad_norm = 1.f;
  auto trainer = ok(ImitationTrainer::create(net, ic), "trainer");

  const auto t_start = std::chrono::steady_clock::now();
  auto minutes = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() / 60.0;
  };
  std::printf("untrained held-out: %.4f nats/char (uniform = %.4f)\n",
              eval_heldout(trainer, ebuf, corpus, o, 64), std::log(double(V)));

  double ema = 0;
  const std::size_t max_start = corpus.train_end - static_cast<std::size_t>(o.seg) - 2;
  int done_steps = 0;
  for (int step = 1; step <= o.steps; ++step) {
    const float frac = float(step - 1) / float(std::max(1, o.steps - 1));
    trainer.set_lr(o.lr_end + 0.5f * (o.lr - o.lr_end) * (1.f + std::cos(3.14159265f * frac)));
    std::vector<std::size_t> starts(static_cast<std::size_t>(o.streams));
    for (auto& s : starts) s = rng.u64() % max_start;
    fill(buf, corpus, o, starts);
    auto st = ok(trainer.epoch(buf), "train");
    ema = step == 1 ? st.loss : 0.98 * ema + 0.02 * st.loss;
    done_steps = step;
    if (step % 50 == 0) std::printf("step %5d  train %.4f (ema %.4f)  acc %.3f  %.1f min\n", step, st.loss, ema, st.accuracy, minutes());
    if (o.eval_every > 0 && step % o.eval_every == 0)
      std::printf("  >> held-out %.4f nats/char\n", eval_heldout(trainer, ebuf, corpus, o, 64));
    if (o.max_minutes > 0 && minutes() > o.max_minutes) {
      std::printf("time budget reached after %d steps\n", step);
      break;
    }
  }
  double acc = 0;
  const double final_loss = eval_heldout(trainer, ebuf, corpus, o, 64, &acc);
  std::printf("FINAL held-out: %.4f nats/char  (%.4f bits/char, top-1 acc %.3f) after %d steps, %.1f min\n",
              final_loss, final_loss / std::log(2.0), acc, done_steps, minutes());

  // ---- deployment path: checkpoint, flat export, gyre-infer
  std::filesystem::create_directories(o.out);
  ok(save_policy(o.out + "/model.gyre", net, static_cast<std::uint64_t>(done_steps)), "save_policy");
  ok(export_policy_flat(o.out + "/model.gpol", net), "export");
  std::string err;
  auto pol = gyre_infer::Policy::load(o.out + "/model.gpol", &err);
  if (!pol) die("gyre-infer load: " + err);

  // held-out NLL through gyre-infer on the first 64 held-out segments vs the trainer's number
  {
    const auto chunks = heldout_chunks(corpus, o, 64);
    fill(ebuf, corpus, o, chunks[0]);
    const double train_side = ok(trainer.evaluate(ebuf), "evaluate").loss;
    double nll = 0;
    long n = 0;
    std::vector<float> logits(static_cast<std::size_t>(V)), ent;
    for (int e = 0; e < 64; ++e) {
      auto sess = pol->make_session();
      for (int t = 0; t < o.seg; ++t) {
        const int K = o.window, F = V + K;
        ent.assign(static_cast<std::size_t>(K) * F, 0.f);
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(K), 0);
        for (int j = 0; j < K && t - j >= 0; ++j) {
          ent[static_cast<std::size_t>(j) * F + corpus.ids[chunks[0][e] + t - j]] = 1.f;
          ent[static_cast<std::size_t>(j) * F + V + j] = 1.f;
          mask[j] = 1;
        }
        float glob = 0.f, value = 0.f;
        gyre_infer::Input in;
        in.entities = ent.data();
        in.mask = mask.data();
        in.n_slots = K;
        in.global = &glob;
        in.reset = t == 0;
        if (!pol->step(sess, in, logits.data(), &value)) die("infer step: " + sess.error());
        const int target = corpus.ids[chunks[0][e] + t + 1];
        float mx = *std::max_element(logits.begin(), logits.end());
        double z = 0;
        for (float l : logits) z += std::exp(double(l - mx));
        nll += -(double(logits[target] - mx) - std::log(z));
        ++n;
      }
    }
    std::printf("gyre-infer parity on 64 held-out segments: infer %.4f vs trainer %.4f nats/char\n", nll / double(n),
                train_side);
  }

  // ---- sample text with gyre-infer
  for (float temp : {0.7f}) {
    const std::string prompt = "ROMEO:\n";
    std::vector<int> hist;
    auto sess = pol->make_session();
    Rng srng(o.seed + 99);
    std::vector<float> logits(static_cast<std::size_t>(V)), ent;
    std::string text = prompt;
    auto feed = [&](int t) {
      const int K = o.window, F = V + K;
      ent.assign(static_cast<std::size_t>(K) * F, 0.f);
      std::vector<std::uint8_t> mask(static_cast<std::size_t>(K), 0);
      for (int j = 0; j < K && j < static_cast<int>(hist.size()); ++j) {
        ent[static_cast<std::size_t>(j) * F + hist[hist.size() - 1 - j]] = 1.f;
        ent[static_cast<std::size_t>(j) * F + V + j] = 1.f;
        mask[j] = 1;
      }
      float glob = 0.f, value = 0.f;
      gyre_infer::Input in;
      in.entities = ent.data();
      in.mask = mask.data();
      in.n_slots = K;
      in.global = &glob;
      in.reset = t == 0;
      if (!pol->step(sess, in, logits.data(), &value)) die("infer step: " + sess.error());
    };
    int t = 0;
    int id_of[256];
    for (int i = 0; i < 256; ++i) id_of[i] = 0;
    for (int i = 0; i < V; ++i) id_of[corpus.chars[i]] = i;
    for (unsigned char ch : prompt) {
      hist.push_back(id_of[ch]);
      feed(t++);
    }
    for (int i = 0; i < 500; ++i) {
      for (auto& l : logits) l /= temp;
      const int pick = gyre_infer::pick_masked(logits.data(), nullptr, V, srng.uniform01());
      hist.push_back(pick);
      text.push_back(static_cast<char>(corpus.chars[pick]));
      feed(t++);
    }
    std::printf("\n--- sample (temp %.1f, via gyre-infer) ---\n%s\n", temp, text.c_str());
  }
  return 0;
}
