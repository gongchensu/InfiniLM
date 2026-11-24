#pragma once

#include <cstddef>
#include <memory>

#include <infini/rt/memory.h>

#include "runtime/device.hpp"

namespace infinilm::runtime {

class Runtime;

using Memory = infini::rt::Memory;

// RT stores allocation identity as an opaque owner token. LM created that
// token from shared_ptr<Runtime>, so this boundary can restore the typed owner.
inline std::shared_ptr<Runtime> allocationOwner(
    const std::shared_ptr<Memory> &memory) {
    if (!memory) {
        return nullptr;
    }
    return std::static_pointer_cast<Runtime>(memory->allocation_owner());
}

} // namespace infinilm::runtime
