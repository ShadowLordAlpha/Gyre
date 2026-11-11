#include "gyre/detail/storage.hpp"

#include <cstring>
#include <map>
#include <mutex>

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

namespace {
constexpr std::size_t kPoolMin = std::size_t{1} << 20;    // pool only buffers >= 1 MiB
constexpr std::size_t kPoolCap = std::size_t{768} << 20;  // total bytes kept
struct Pool {
  std::mutex mu;
  std::multimap<std::size_t, std::vector<std::byte>> free;  // key = capacity
  std::size_t held{0};
};
Pool& pool() {
  static Pool* p = new Pool();  // intentionally leaked: Storage may be destroyed during exit
  return *p;
}
}  // namespace

std::vector<std::byte> pool_acquire(std::size_t bytes, bool zero) {
  if (bytes >= kPoolMin) {
    auto& P = pool();
    std::vector<std::byte> got;
    {
      std::lock_guard<std::mutex> lk(P.mu);
      auto it = P.free.lower_bound(bytes);
      if (it != P.free.end() && it->first <= bytes + bytes / 2) {  // at most 50% waste
        got = std::move(it->second);
        P.held -= it->first;
        P.free.erase(it);
      }
    }
    if (got.capacity() >= bytes) {
      got.resize(bytes);
      if (zero) std::memset(got.data(), 0, bytes);
      return got;
    }
  }
  std::vector<std::byte> v;
  v.resize(bytes);  // value-initialised: zero
  return v;
}

void pool_release(std::vector<std::byte>&& buf) {
  const std::size_t cap = buf.capacity();
  if (cap < kPoolMin) return;
  auto& P = pool();
  std::lock_guard<std::mutex> lk(P.mu);
  if (P.held + cap > kPoolCap) return;  // drop: let the vector free itself
  P.held += cap;
  P.free.emplace(cap, std::move(buf));
}

Storage::~Storage() {
  if (!mapped && !gpu && heap.capacity() >= kPoolMin) pool_release(std::move(heap));
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
