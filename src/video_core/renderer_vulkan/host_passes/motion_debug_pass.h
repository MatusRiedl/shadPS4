// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/host_passes/xfb_velocity_pass.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image.h"

namespace VideoCore {
struct Image;
}

namespace Vulkan {
class Instance;
class Runtime;
class Scheduler;
} // namespace Vulkan

namespace Vulkan::HostPasses {

// Overtime Native v3 (GPU.motion_debug): draws the velocity pass's motion vectors and their source
// over the presented image, to check the motion inputs DLSS reads (camera pans, players, HUD).
class MotionDebugPass {
public:
    MotionDebugPass(const Instance& instance, Scheduler& scheduler, Runtime& runtime);
    ~MotionDebugPass();

    void Render(VideoCore::Image& color, const XfbVelocityPass::Outputs& inputs);

private:
    void CreatePipeline();
    void ResizeStaging(vk::Extent2D extent);

    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    vk::ShaderModule module{};
    vk::UniqueDescriptorSetLayout set_layout;
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
    VideoCore::UniqueImage staging;
    vk::UniqueImageView staging_view;
    vk::Extent2D size{};
    bool size_logged{};
    u64 frames{};
};

} // namespace Vulkan::HostPasses
