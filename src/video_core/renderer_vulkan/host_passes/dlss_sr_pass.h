// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include "video_core/renderer_vulkan/host_passes/xfb_velocity_pass.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {
struct Image;
}

namespace Vulkan {
class Instance;
class Runtime;
class Scheduler;
} // namespace Vulkan

namespace Vulkan::HostPasses {

// Overtime Native v3 (GPU.upscaler = "dlss"): DLSS Super Resolution through the driver's NGX core
// (no NVIDIA library linked), fed the presented image with depth and motion vectors from the
// velocity pass. The player's own nvngx_dlss.dll sits beside the executable. No jitter exists in
// a PS4 game, so it upscales (or anti-aliases at the same size) from motion alone. Windows only.
class DlssSrPass {
public:
    DlssSrPass(const Instance& instance, Scheduler& scheduler, Runtime& runtime);
    ~DlssSrPass();

    // The upscaled image at output_size (R16G16B16A16, shader read layout), or a null view when
    // DLSS cannot run this frame (the caller uses FSR 1 instead). color_view is the presenter's
    // view of the guest image (its display format, alpha forced to one). May finish the scheduler's
    // command buffer when the output size changes.
    vk::ImageView Render(VideoCore::Image& color, vk::ImageView color_view, vk::Format color_format,
                         const XfbVelocityPass::Outputs& inputs, vk::Extent2D output_size);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan::HostPasses
