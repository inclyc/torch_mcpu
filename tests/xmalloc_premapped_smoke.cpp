// Compile with TORCH_MCPU_SHARED_USE_PREMAPPED=1; supplies a mapped address
// without hardware.
#include "runtime/OpenRegSharedAllocator.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Function>
void require_failure(Function function) {
  try {
    function();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("expected allocation failure");
}

int main() {
  try {
    using c10::mcpu::xmalloc;
    ::unsetenv("TORCH_MCPU_SHARED_SIM_FILE");
    ::unsetenv("TORCH_MCPU_SHARED_BASE");
    ::setenv("TORCH_MCPU_SHARED_SIZE", "1024", 1);
    require_failure([] { xmalloc(1); });
    ::setenv("TORCH_MCPU_SHARED_BASE", "0", 1);
    require_failure([] { xmalloc(1); });
    ::setenv("TORCH_MCPU_SHARED_BASE", "0xffffffffffffffff", 1);
    require_failure([] { xmalloc(1); });

    alignas(256) static unsigned char backing[1024] = {};
    backing[0] = 91;
    // The common layer must align the base as well as each allocation size.
    const auto start = reinterpret_cast<uintptr_t>(backing) + 1;
    ::setenv("TORCH_MCPU_SHARED_BASE", std::to_string(start).c_str(), 1);
    ::setenv("TORCH_MCPU_SHARED_SIZE", "1023", 1);
    require_failure([] { xmalloc(0); });
    require_failure([] { xmalloc(std::numeric_limits<size_t>::max()); });
    auto* a = static_cast<unsigned char*>(xmalloc(1));
    auto* b = static_cast<unsigned char*>(xmalloc(257));
    require(a == backing + 256, "base was not aligned to 256 bytes");
    require(b == backing + 512, "allocation size was not rounded up");
    a[0] = 12;
    b[256] = 34;
    require(backing[0] == 91, "alignment prefix was overwritten");
    require(backing[256] == 12 && backing[768] == 34, "wrong mapped backing");
    require_failure([] { xmalloc(1); });
    std::cout << "PASS: pre-mapped branch uses supplied base and bounds.\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
