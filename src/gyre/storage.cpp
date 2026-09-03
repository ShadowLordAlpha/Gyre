#include "gyre/detail/storage.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace gyre {

Storage::~Storage() {
#ifdef _WIN32
  if (mapped) UnmapViewOfFile(mapped);
  if (win_map) CloseHandle(static_cast<HANDLE>(win_map));
  if (win_file) CloseHandle(static_cast<HANDLE>(win_file));
  mapped = nullptr;
  win_map = nullptr;
  win_file = nullptr;
#else
  if (mapped && mapped_len) munmap(mapped, mapped_len);
  if (posix_fd >= 0) close(posix_fd);
  mapped = nullptr;
  posix_fd = -1;
#endif
}

Result<std::shared_ptr<Storage>> Storage::mmap_file(const std::filesystem::path& path) {
  auto st = std::make_shared<Storage>();
#ifdef _WIN32
  HANDLE f = CreateFileW(path.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (f == INVALID_HANDLE_VALUE) {
    return std::unexpected(make_error(Errc::io, "mmap open " + path.string()));
  }
  LARGE_INTEGER sz{};
  if (!GetFileSizeEx(f, &sz) || sz.QuadPart <= 0) {
    CloseHandle(f);
    return std::unexpected(make_error(Errc::io, "mmap size"));
  }
  HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!m) {
    CloseHandle(f);
    return std::unexpected(make_error(Errc::io, "mmap mapping"));
  }
  void* p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
  if (!p) {
    CloseHandle(m);
    CloseHandle(f);
    return std::unexpected(make_error(Errc::io, "mmap view"));
  }
  st->win_file = f;
  st->win_map = m;
  st->mapped = static_cast<std::byte*>(p);
  st->mapped_len = static_cast<std::size_t>(sz.QuadPart);
#else
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return std::unexpected(make_error(Errc::io, "mmap open " + path.string()));
  struct stat s {};
  if (fstat(fd, &s) != 0 || s.st_size <= 0) {
    close(fd);
    return std::unexpected(make_error(Errc::io, "mmap size"));
  }
  void* p = mmap(nullptr, static_cast<std::size_t>(s.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
  if (p == MAP_FAILED) {
    close(fd);
    return std::unexpected(make_error(Errc::io, "mmap view"));
  }
  st->posix_fd = fd;
  st->mapped = static_cast<std::byte*>(p);
  st->mapped_len = static_cast<std::size_t>(s.st_size);
#endif
  return st;
}

}  // namespace gyre
