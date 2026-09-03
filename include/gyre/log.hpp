#pragma once

#include <functional>
#include <string_view>

namespace gyre {

enum class LogLevel : int { debug = 0, info = 1, warn = 2, error = 3 };

using LogSink = std::function<void(LogLevel, std::string_view)>;

void set_log_sink(LogSink sink);
void log(LogLevel level, std::string_view msg);

}  // namespace gyre
