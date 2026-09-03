#include "jobs.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

int main() {
  using namespace ftxui;
  using gyre::app::GaOpts;
  using gyre::app::CharLMOpts;

  auto screen = ScreenInteractive::Fullscreen();

  int tab = 0;
  std::vector<std::string> tab_entries = {"Character LM", "Genetic algorithm"};

  std::string data_path = "data/shakespeare.txt";
  std::string ckpt_path = "data/charlm.gyre";
  std::string preset_s = "medium";
  std::string tok_s = "bpe";
  std::string tok_path = "";
  std::string vocab_s = "2000";
  std::string holdout_s = "0.1";
  std::string steps_s = "2000";
  std::string batch_s = "4";
  std::string block_s = "128";
  std::string prompt = "To be";
  std::string chars_s = "200";
  std::string temp_s = "0.8";
  std::string gens_s = "80";
  std::string n_s = "64";
  std::string dim_s = "64";

  std::mutex log_mu;
  std::vector<std::string> lines;
  std::string progress;
  std::atomic<bool> busy{false};

  auto push = [&](std::string s) {
    std::lock_guard<std::mutex> g(log_mu);
    if (!s.empty() && s.front() == '\r') {
      progress = s.substr(1);
    } else {
      lines.push_back(std::move(s));
      if (lines.size() > 200) lines.erase(lines.begin(), lines.begin() + 50);
    }
    screen.Post(Event::Custom);
  };

  auto start_job = [&](auto fn) {
    if (busy.exchange(true)) {
      push("already running");
      return;
    }
    std::thread([&, fn = std::move(fn)]() mutable {
      fn();
      busy = false;
      screen.Post(Event::Custom);
    }).detach();
  };

  auto train_sh = [&] {
    CharLMOpts o;
    o.data = data_path;
    o.ckpt = ckpt_path;
    o.preset = preset_s;
    o.tokenizer = tok_s;
    if (!tok_path.empty()) o.tok = tok_path;
    try {
      o.vocab_size = std::stoi(vocab_s);
      o.holdout = std::stod(holdout_s);
      o.steps = static_cast<std::uint32_t>(std::stoul(steps_s));
      o.batch = static_cast<std::uint32_t>(std::stoul(batch_s));
      o.block = static_cast<std::uint32_t>(std::stoul(block_s));
      o.temperature = std::stof(temp_s);
      gyre::app::apply_charlm_preset(o);
      o.steps = static_cast<std::uint32_t>(std::stoul(steps_s));
      o.batch = static_cast<std::uint32_t>(std::stoul(batch_s));
      o.block = static_cast<std::uint32_t>(std::stoul(block_s));
    } catch (...) {
      push("invalid numeric field");
      return;
    }
    start_job([&, o] {
      push("char-LM train…");
      auto r = gyre::app::run_charlm_train(o, [&](std::string s) { push(std::move(s)); });
      if (!r) push("error: " + r.error().message);
      else push("train finished");
    });
  };

  auto gen_sh = [&] {
    CharLMOpts o;
    o.data = data_path;
    o.ckpt = ckpt_path;
    o.preset = preset_s;
    try {
      gyre::app::apply_charlm_preset(o);
      o.block = static_cast<std::uint32_t>(std::stoul(block_s));
      o.temperature = std::stof(temp_s);
    } catch (...) {
      push("invalid block/temp");
      return;
    }
    int n = 200;
    try {
      n = std::stoi(chars_s);
    } catch (...) {
    }
    start_job([&, o, n] {
      push("generate…");
      auto r = gyre::app::run_charlm_generate(o, prompt, n, [&](std::string s) { push(std::move(s)); });
      if (!r) push("error: " + r.error().message);
    });
  };

  auto run_ga = [&] {
    GaOpts o;
    try {
      o.generations = static_cast<std::uint32_t>(std::stoul(gens_s));
      o.n = static_cast<std::uint32_t>(std::stoul(n_s));
      o.dim = static_cast<std::uint32_t>(std::stoul(dim_s));
    } catch (...) {
      push("invalid GA field");
      return;
    }
    start_job([&, o] {
      push("OneMax…");
      auto r = gyre::app::run_onemax(o, [&](std::string s) { push(std::move(s)); });
      if (!r) push("error: " + r.error().message);
    });
  };

  auto in_preset = Input(&preset_s, "tiny|medium|tinygpt");
  auto in_tok = Input(&tok_s, "bpe|chars|bytes|unigram");
  auto in_tok_path = Input(&tok_path, "tok .gyre.json");
  auto in_vocab = Input(&vocab_s, "bpe vocab");
  auto in_holdout = Input(&holdout_s, "holdout");
  auto in_data = Input(&data_path, "data path");
  auto in_ckpt = Input(&ckpt_path, "checkpoint");
  auto in_steps = Input(&steps_s, "steps");
  auto in_batch = Input(&batch_s, "batch");
  auto in_block = Input(&block_s, "block");
  auto in_prompt = Input(&prompt, "prompt");
  auto in_chars = Input(&chars_s, "chars");
  auto in_temp = Input(&temp_s, "temp");
  auto in_gens = Input(&gens_s, "generations");
  auto in_n = Input(&n_s, "pop size");
  auto in_dim = Input(&dim_s, "gene dim");

  auto btn_train = Button("Train char-LM", train_sh);
  auto btn_gen = Button("Generate from ckpt", gen_sh);
  auto btn_ga = Button("Run OneMax", run_ga);
  auto btn_quit = Button("Quit", screen.ExitLoopClosure());

  auto tabs = Toggle(&tab_entries, &tab);

  auto sh_form = Container::Vertical({
      in_preset,
      in_tok,
      in_tok_path,
      in_vocab,
      in_holdout,
      in_data,
      in_ckpt,
      in_steps,
      in_batch,
      in_block,
      in_prompt,
      in_chars,
      in_temp,
      Container::Horizontal({btn_train, btn_gen}),
  });
  auto ga_form = Container::Vertical({
      in_gens,
      in_n,
      in_dim,
      btn_ga,
  });
  auto root = Container::Vertical({
      tabs,
      Container::Tab({sh_form, ga_form}, &tab),
      btn_quit,
  });

  auto renderer = Renderer(root, [&] {
    std::vector<std::string> snap;
    std::string prog;
    {
      std::lock_guard<std::mutex> g(log_mu);
      snap = lines;
      prog = progress;
    }
    Elements log_el;
    const int keep = 18;
    int start = static_cast<int>(snap.size()) > keep ? static_cast<int>(snap.size()) - keep : 0;
    for (int i = start; i < static_cast<int>(snap.size()); ++i) log_el.push_back(text(snap[static_cast<std::size_t>(i)]));
    if (log_el.empty()) log_el.push_back(text("(log empty)"));

    Element form;
    if (tab == 0) {
      form = vbox({
          hbox({text("preset "), in_preset->Render(), text("  tok "), in_tok->Render(), text("  V "),
                in_vocab->Render(), text("  holdout "), in_holdout->Render()}),
          hbox({text("data   "), in_data->Render() | flex}),
          hbox({text("tok    "), in_tok_path->Render() | flex}),
          hbox({text("ckpt   "), in_ckpt->Render() | flex}),
          hbox({text("steps  "), in_steps->Render(), text("  batch "), in_batch->Render(),
                text("  block "), in_block->Render()}),
          hbox({text("prompt "), in_prompt->Render() | flex, text("  chars "), in_chars->Render(),
                text("  temp "), in_temp->Render()}),
          hbox({btn_train->Render(), separator(), btn_gen->Render()}),
          paragraph("Tiny character transformer. --data is any text (default: Shakespeare corpus). "
                    "Holdout is the last raw-byte fraction (default 0.1). Train writes GYRE1."),
      });
    } else {
      form = vbox({
          hbox({text("generations "), in_gens->Render(), text("  N "), in_n->Render(), text("  dim "),
                in_dim->Render()}),
          btn_ga->Render(),
          paragraph("OneMax: count 1-bits in a u8 bitstring. Fitness should climb toward dim."),
      });
    }

    return vbox({
               text("Gyre TUI 0.1") | bold,
               text(busy ? (prog.empty() ? "status: running" : prog) : "status: idle"),
               tabs->Render(),
               separator(),
               form,
               separator(),
               text("log") | bold,
               vbox(std::move(log_el)) | yframe | flex,
               separator(),
               hbox({btn_quit->Render(), filler(),
                     text("Tab: switch  Enter: activate  q not bound — use Quit")}),
           }) |
           border;
  });

  screen.Loop(renderer);
  return 0;
}
