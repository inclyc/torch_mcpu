#include "OpenRegSharedAllocator.h"

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>

#if !TORCH_MCPU_SHARED_USE_PREMAPPED
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <system_error>
#endif

namespace c10::mcpu {
namespace {

constexpr size_t kAlignment = 256;

uintptr_t read_address_or_size(const char* name, bool required) {
  const char* value = std::getenv(name);
  if (!value) {
    if (required) {
      throw std::invalid_argument(std::string("xmalloc: missing ") + name);
    }
    return 0;
  }
  std::string_view text(value);
  int base = 10;
  if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
    text.remove_prefix(2);
    base = 16;
  }
  uintptr_t result = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), result, base);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
    throw std::invalid_argument(
        std::string("xmalloc: invalid unsigned integer in ") + name);
  }
  return result;
}

#if !TORCH_MCPU_SHARED_USE_PREMAPPED
struct FileDescriptor {
  int fd;
  ~FileDescriptor() {
    ::close(fd);
  }
};

void check_system_call(bool success, const char* operation) {
  if (!success) {
    throw std::system_error(
        errno, std::generic_category(), std::string("xmalloc: ") + operation);
  }
}
#endif

// Only this function depends on how the hardware exposes shared memory.
uintptr_t open_region(uintptr_t requested_base, size_t size) {
#if TORCH_MCPU_SHARED_USE_PREMAPPED
  (void)size;
  // Device startup is responsible for mapping the same physical region into
  // each rank. TORCH_MCPU_SHARED_BASE is a process-accessible address, not an
  // unmapped PA.
  if (requested_base == 0) {
    throw std::invalid_argument(
        "xmalloc: TORCH_MCPU_SHARED_BASE must be nonzero for pre-mapped regions");
  }
  return requested_base;
#else
  const char* path = std::getenv("TORCH_MCPU_SHARED_SIM_FILE");
  if (!path || !*path) {
    throw std::invalid_argument("xmalloc: missing TORCH_MCPU_SHARED_SIM_FILE");
  }
  if (size > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
    throw std::invalid_argument("xmalloc: region size exceeds off_t range");
  }
  if (requested_base != 0 && requested_base % ::sysconf(_SC_PAGESIZE) != 0) {
    throw std::invalid_argument(
        "xmalloc: simulated TORCH_MCPU_SHARED_BASE must be page aligned");
  }
  const int fd = ::open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0600);
  check_system_call(fd >= 0, "open shared file");
  FileDescriptor owner{fd};
  // Serialize first creation against another rank opening the same file.
  check_system_call(::flock(fd, LOCK_EX) == 0, "lock shared file");
  struct stat status {};
  check_system_call(::fstat(fd, &status) == 0, "stat shared file");
  if (!S_ISREG(status.st_mode)) {
    throw std::invalid_argument(
        "xmalloc: TORCH_MCPU_SHARED_SIM_FILE must be a regular file");
  }
  if (status.st_size == 0) {
    check_system_call(::ftruncate(fd, size) == 0, "size shared file");
  } else if (status.st_size != static_cast<off_t>(size)) {
    throw std::invalid_argument(
        "xmalloc: shared file size differs from TORCH_MCPU_SHARED_SIZE");
  }
  // mmap retains the open file description. Release the initialization lock
  // explicitly so another process can attach while this mapping stays alive.
  check_system_call(::flock(fd, LOCK_UN) == 0, "unlock shared file");
  int flags = MAP_SHARED;
  if (requested_base != 0) {
#ifdef MAP_FIXED_NOREPLACE
    flags |= MAP_FIXED_NOREPLACE;
#else
    throw std::runtime_error(
        "xmalloc: fixed simulation mapping is unsupported");
#endif
  }
  void* mapping = ::mmap(
      reinterpret_cast<void*>(requested_base),
      size,
      PROT_READ | PROT_WRITE,
      flags,
      fd,
      0);
  check_system_call(mapping != MAP_FAILED, "map shared file");
  const auto base = reinterpret_cast<uintptr_t>(mapping);
  if (requested_base != 0 && base != requested_base) {
    ::munmap(mapping, size);
    throw std::runtime_error("xmalloc: kernel did not honor fixed mapping");
  }
  // Mapping survives fd close. The caller removes the file after all ranks exit.
  return base;
#endif
}

struct Arena {
  uintptr_t base;
  size_t capacity;
  size_t used = 0;
  std::mutex mutex;

  Arena() {
    const auto requested_base = read_address_or_size(
        "TORCH_MCPU_SHARED_BASE", TORCH_MCPU_SHARED_USE_PREMAPPED);
    const auto size = read_address_or_size("TORCH_MCPU_SHARED_SIZE", true);
    if (size == 0 ||
        requested_base > std::numeric_limits<uintptr_t>::max() - size) {
      throw std::invalid_argument("xmalloc: invalid shared region range");
    }
    // Alignment and capacity accounting are common to both backends.
    const size_t padding =
        (kAlignment - requested_base % kAlignment) % kAlignment;
    if (size <= padding || size - padding < kAlignment) {
      throw std::invalid_argument(
          "xmalloc: shared region has no aligned capacity");
    }
    base = open_region(requested_base, size) + padding;
    capacity = size - padding;
  }

  void* allocate(size_t bytes) {
    std::lock_guard<std::mutex> guard(mutex);
    const size_t reserved = (bytes + kAlignment - 1) & ~(kAlignment - 1);
    if (reserved > capacity - used) {
      throw std::runtime_error("xmalloc: shared region exhausted");
    }
    void* result = reinterpret_cast<void*>(base + used);
    used += reserved;
    return result;
  }
};

} // namespace

void* xmalloc(size_t size) {
  if (size == 0 ||
      size > std::numeric_limits<size_t>::max() - (kAlignment - 1)) {
    throw std::invalid_argument(
        "xmalloc: size must be positive and align without overflow");
  }
  // Deliberately live until process exit, including during runtime teardown.
  // This arena is not the ordinary caching allocator and is never recycled.
  static Arena* arena = new Arena();
  return arena->allocate(size);
}

} // namespace c10::mcpu
