#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace npu::tile_fwk::gpu_vk {

class ShapeCache {
public:
    bool Find(const std::string &key, std::vector<std::int64_t> &shape) const;
    void Insert(const std::string &key, const std::vector<std::int64_t> &shape);
    std::size_t EntryCount() const;

private:
    std::unordered_map<std::string, std::vector<std::int64_t>> entries_;
};

} // namespace npu::tile_fwk::gpu_vk
