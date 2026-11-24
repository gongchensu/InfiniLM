#include "runtime/runtime.hpp"

#include <infini/rt.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string_view>
#include <vector>

namespace {

using namespace infinilm::runtime;

void require(bool condition) {
    if (!condition) {
        throw std::runtime_error{"tensor allocator bridge assertion failed"};
    }
}

} // namespace

int main() {
    const char *requested_device = std::getenv("INFINILM_TENSOR_ALLOCATOR_DEVICE");
    const bool use_nvidia = requested_device != nullptr && std::string_view{requested_device} == "nvidia";
    const Device device{use_nvidia ? Device::Type::kNvidia : Device::Type::kCpu, 0};
    auto runtime = Runtime::create(device);
    runtime->activate();

    infini::rt::Tensor tensor = infini::rt::Tensor::empty(
        *runtime, {2, 3}, DataType::kFloat32,
        infini::rt::TensorStorageKind::kGeneral);
    require(tensor && tensor.is_contiguous() && tensor.numel() == 6);
    require(tensor.device() == runtime->device());
    require(tensor.data() != nullptr);

    std::vector<float> host_values(tensor.numel(), 0.0F);
    for (std::size_t index = 0; index < host_values.size(); ++index) {
        host_values[index] = static_cast<float>(index);
    }
    runtime->memcpyH2D(tensor.data(), host_values.data(), tensor.nbytes());
    runtime->syncStream();
    if (use_nvidia) {
        std::vector<float> checked_values(host_values.size(), 0.0F);
        runtime->memcpyD2H(checked_values.data(), tensor.data(), tensor.nbytes());
        require(checked_values.back() == static_cast<float>(host_values.size() - 1));
    } else {
        auto *values = static_cast<float *>(tensor.data());
        require(values[5] == 5.0F);
    }

    const auto owner = runtime->allocation_owner().lock();
    require(owner == runtime);

    tensor.reset();
    runtime->syncStream();
    return 0;
}
