#pragma once

#include "gyre/error.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace gyre::detail {

// Writes `path` through a sibling "<path>.tmp": `write(out)` fills the stream, which is then
// flushed, closed and checked before the temp file is renamed over `path`. A crash, a kill or a
// full disk therefore leaves the previous file intact (and never reports success for a truncated
// file). If the platform refuses to replace the destination (Windows, file open elsewhere) the
// temp file is copied over it instead, which is what an in-place write would have done.
template <class WriteFn>
Result<void> write_file_atomic(const std::filesystem::path& path, WriteFn&& write) {
  std::error_code ec;
  const auto parent = path.parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent, ec);
  auto tmp = path;
  tmp += ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) return std::unexpected(make_error(Errc::io, "open write " + tmp.string()));
    Result<void> r = write(out);
    if (r) {
      out.flush();
      if (!out) r = std::unexpected(make_error(Errc::io, "write " + tmp.string()));
    }
    out.close();
    if (r && out.fail()) r = std::unexpected(make_error(Errc::io, "close " + tmp.string()));
    if (!r) {
      std::filesystem::remove(tmp, ec);
      return r;
    }
  }
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    std::error_code ec2;
    std::filesystem::copy_file(tmp, path, std::filesystem::copy_options::overwrite_existing, ec2);
    std::error_code ec3;
    std::filesystem::remove(tmp, ec3);
    if (ec2) return std::unexpected(make_error(Errc::io, "replace " + path.string() + ": " + ec.message()));
  }
  return {};
}

}  // namespace gyre::detail
