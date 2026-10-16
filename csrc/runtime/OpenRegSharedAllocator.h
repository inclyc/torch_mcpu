#pragma once

#include <Macros.h>
#include <cstddef>

namespace c10::mcpu {

// Process-lifetime KV arena. Each rank keeps its own cursor: ranks sharing an
// arena MUST make identical allocations in identical order during startup.
// Returns a 256-byte-aligned local address; never pass it to free/orFree.
// No implicit initialization, device synchronization, or per-allocation free.
// Throws on invalid configuration, zero size, overflow, or arena exhaustion.
MCPU_EXPORT void* xmalloc(size_t size);

} // namespace c10::mcpu
