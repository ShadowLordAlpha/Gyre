#include "gyre/log.hpp"

namespace gyre {
namespace {
LogSink g_sink;
}

void set_log_sink(LogSink sink) { g_sink = std::move(sink); }

void log(LogLevel level, std::string_view msg) {
  if (g_sink) g_sink(level, msg);
}

}  // namespace gyre
