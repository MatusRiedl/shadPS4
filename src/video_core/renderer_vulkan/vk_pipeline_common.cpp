// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <boost/container/static_vector.hpp>

#include "common/assert.h"

#include "shader_recompiler/resource.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

Pipeline::Pipeline(const Instance& instance_, Scheduler& scheduler_, DescriptorHeap& desc_heap_,
                   const Shader::Profile& profile_, vk::PipelineCache pipeline_cache,
                   bool is_compute_ /*= false*/)
    : instance{instance_}, scheduler{scheduler_}, desc_heap{desc_heap_}, profile{profile_},
      is_compute{is_compute_} {}

Pipeline::~Pipeline() = default;

void Pipeline::BindResources(DescriptorWrites& set_writes,
                             const Shader::PushData& push_data) const {
    const auto cmdbuf = scheduler.CommandBuffer();
    const auto bind_point =
        IsCompute() ? vk::PipelineBindPoint::eCompute : vk::PipelineBindPoint::eGraphics;

    const auto stage_flags = IsCompute() ? vk::ShaderStageFlagBits::eCompute : AllGraphicsStageBits;
    cmdbuf.pushConstants(*pipeline_layout, stage_flags, 0u, sizeof(push_data), &push_data);

    // Bind descriptor set.
    if (set_writes.empty()) {
        return;
    }

    if (uses_push_descriptors) {
        cmdbuf.pushDescriptorSetKHR(bind_point, *pipeline_layout, 0, set_writes);
        return;
    }

    const auto desc_set = desc_heap.Commit(*desc_layout);
    for (auto& set_write : set_writes) {
        set_write.dstSet = desc_set;
    }
    instance.GetDevice().updateDescriptorSets(set_writes, {});
    cmdbuf.bindDescriptorSets(bind_point, *pipeline_layout, 0, desc_set, {});
}

void Pipeline::BindResources(DescriptorWrites& set_writes, const Shader::PushData& push_data,
                             DrawPacket& packet) const {
    ASSERT(!IsCompute());
    static_assert(sizeof(push_data) <= sizeof(packet.push_data));
    packet.layout = *pipeline_layout;
    packet.push_stages = AllGraphicsStageBits;
    packet.push_size = sizeof(push_data);
    std::memcpy(packet.push_data.data(), &push_data, sizeof(push_data));

    if (set_writes.empty()) {
        return;
    }

    if (!uses_push_descriptors) {
        const auto desc_set = desc_heap.Commit(*desc_layout);
        for (auto& set_write : set_writes) {
            set_write.dstSet = desc_set;
        }
        instance.GetDevice().updateDescriptorSets(set_writes, {});
        packet.bind_set = true;
        packet.desc_set = desc_set;
        return;
    }

    // Push descriptors are recorded later: copy the writes with what they point to.
    size_t num_buffers = 0;
    size_t num_images = 0;
    for (const auto& set_write : set_writes) {
        ASSERT(!set_write.pNext && !set_write.pTexelBufferView);
        if (set_write.pBufferInfo) {
            num_buffers += set_write.descriptorCount;
        } else if (set_write.pImageInfo) {
            num_images += set_write.descriptorCount;
        }
    }
    packet.writes.assign(set_writes.begin(), set_writes.end());
    packet.buffer_infos.resize(num_buffers);
    packet.image_infos.resize(num_images);
    size_t buffer_index = 0;
    size_t image_index = 0;
    for (auto& set_write : packet.writes) {
        if (set_write.pBufferInfo) {
            auto* dst = packet.buffer_infos.data() + buffer_index;
            std::copy_n(set_write.pBufferInfo, set_write.descriptorCount, dst);
            set_write.pBufferInfo = dst;
            buffer_index += set_write.descriptorCount;
        } else if (set_write.pImageInfo) {
            auto* dst = packet.image_infos.data() + image_index;
            std::copy_n(set_write.pImageInfo, set_write.descriptorCount, dst);
            set_write.pImageInfo = dst;
            image_index += set_write.descriptorCount;
        }
    }
    packet.push_descriptors = true;
}

std::string Pipeline::GetDebugString() const {
    std::string stage_desc;
    for (const auto& stage : stages) {
        if (stage) {
            const auto shader_name = PipelineCache::GetShaderName(stage->hw_stage, stage->pgm_hash);
            if (stage_desc.empty()) {
                stage_desc = shader_name;
            } else {
                stage_desc = fmt::format("{},{}", stage_desc, shader_name);
            }
        }
    }
    return stage_desc;
}

} // namespace Vulkan
