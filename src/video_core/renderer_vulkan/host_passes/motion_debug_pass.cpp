// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/host_shaders/motion_debug_comp.h"
#include "video_core/renderer_vulkan/host_passes/motion_debug_pass.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan::HostPasses {

namespace {

constexpr vk::Format StagingFormat = vk::Format::eR16G16B16A16Sfloat;
// Motion of this many pixels per frame shows at full colour.
constexpr float FullPixels = 16.0f;

struct PushConstants {
    u32 srgb_target;
    float full_pixels;
};

bool IsSrgb(vk::Format format) {
    switch (format) {
    case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eA8B8G8R8SrgbPack32:
        return true;
    default:
        return false;
    }
}

} // namespace

MotionDebugPass::MotionDebugPass(const Instance& instance_, Scheduler& scheduler_,
                                 Runtime& runtime_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging{instance.GetDevice(), instance.GetAllocator()} {
    CreatePipeline();
    LOG_INFO(Render_Vulkan, "Motion view on (GPU.motion_debug)");
}

MotionDebugPass::~MotionDebugPass() {
    pipeline.reset();
    layout.reset();
    set_layout.reset();
    instance.GetDevice().destroyShaderModule(module);
}

void MotionDebugPass::CreatePipeline() {
    const vk::Device device = instance.GetDevice();
    module = CompileSPV(MOTION_DEBUG_COMP, device);
    ASSERT(module);
    SetObjectName(device, module, "motion_debug.comp");
    const std::array bindings = {
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 3,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    };
    set_layout = Check<"create motion view descriptor set layout">(
        device.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo{
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        }));
    const vk::PushConstantRange constants{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(PushConstants),
    };
    const vk::DescriptorSetLayout set = *set_layout;
    layout = Check<"create motion view pipeline layout">(
        device.createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &set,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &constants,
        }));
    pipeline = Check<"create motion view pipeline">(device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{
                .stage =
                    {
                        .stage = vk::ShaderStageFlagBits::eCompute,
                        .module = module,
                        .pName = "main",
                    },
                .layout = *layout,
            }));
}

void MotionDebugPass::ResizeStaging(vk::Extent2D extent) {
    if (size == extent) {
        return;
    }
    scheduler.Finish();
    staging_view.reset();
    staging.Destroy();
    size = extent;
    staging.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = StagingFormat,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    const vk::Device device = instance.GetDevice();
    SetObjectName(device, static_cast<vk::Image>(staging), "Motion View Staging");
    staging_view = Check<"create motion view staging view">(
        device.createImageViewUnique(vk::ImageViewCreateInfo{
            .image = staging,
            .viewType = vk::ImageViewType::e2D,
            .format = StagingFormat,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .levelCount = 1,
                .layerCount = 1,
            },
        }));
}

void MotionDebugPass::Render(VideoCore::Image& color, const XfbVelocityPass::Outputs& inputs) {
    const vk::Extent2D extent{color.info.size.width, color.info.size.height};
    if (!size_logged) {
        LOG_INFO(Render_Vulkan, "Motion view: presented image {}x{}, motion targets {}x{}{}",
                 extent.width, extent.height, inputs.size.width, inputs.size.height,
                 extent == inputs.size ? "" : " (sizes differ: DLSS would skip these frames)");
        size_logged = true;
    }
    ResizeStaging(extent);

    const vk::CommandBuffer cmdbuf = scheduler.CommandBuffer();
    scheduler.EndRendering();
    runtime.Transit(&color, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderSampledRead);
    runtime.FlushBarriers();

    const vk::ImageSubresourceRange range{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = 1,
    };
    const auto all = vk::PipelineStageFlagBits2::eAllCommands;
    const auto any_access = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
    const std::array start = {
        vk::ImageMemoryBarrier2{
            .srcStageMask = all,
            .srcAccessMask = any_access,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = staging,
            .subresourceRange = range,
        },
        vk::ImageMemoryBarrier2{
            .srcStageMask = all,
            .srcAccessMask = any_access,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = inputs.motion,
            .subresourceRange = range,
        },
        vk::ImageMemoryBarrier2{
            .srcStageMask = all,
            .srcAccessMask = any_access,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = inputs.mask,
            .subresourceRange = range,
        },
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<u32>(start.size()),
        .pImageMemoryBarriers = start.data(),
    });

    VideoCore::ImageViewInfo color_view_info{};
    color_view_info.format = color.info.pixel_format;
    const vk::ImageView color_view = *color.FindView(color_view_info, false).image_view;
    const vk::DescriptorImageInfo color_info{
        .imageView = color_view, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo motion_info{.imageView = inputs.motion_view,
                                              .imageLayout = vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo mask_info{.imageView = inputs.mask_view,
                                            .imageLayout = vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo staging_info{.imageView = *staging_view,
                                               .imageLayout = vk::ImageLayout::eGeneral};
    const std::array writes = {
        vk::WriteDescriptorSet{.dstBinding = 0,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eSampledImage,
                               .pImageInfo = &color_info},
        vk::WriteDescriptorSet{.dstBinding = 1,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eStorageImage,
                               .pImageInfo = &motion_info},
        vk::WriteDescriptorSet{.dstBinding = 2,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eStorageImage,
                               .pImageInfo = &mask_info},
        vk::WriteDescriptorSet{.dstBinding = 3,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eStorageImage,
                               .pImageInfo = &staging_info},
    };
    const PushConstants constants{
        .srgb_target = IsSrgb(color.info.pixel_format) ? 1u : 0u,
        .full_pixels = FullPixels,
    };
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *layout, 0, writes);
    cmdbuf.pushConstants(*layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(constants),
                         &constants);
    cmdbuf.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);

    const vk::ImageMemoryBarrier2 to_blit{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eTransferSrcOptimal,
        .image = staging,
        .subresourceRange = range,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &to_blit,
    });
    runtime.Transit(&color, vk::ImageLayout::eTransferDstOptimal,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
    runtime.FlushBarriers();

    const vk::ImageSubresourceLayers layers{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    const std::array<vk::Offset3D, 2> bounds = {
        vk::Offset3D{0, 0, 0},
        vk::Offset3D{static_cast<s32>(size.width), static_cast<s32>(size.height), 1},
    };
    const vk::ImageBlit region{
        .srcSubresource = layers,
        .srcOffsets = bounds,
        .dstSubresource = layers,
        .dstOffsets = bounds,
    };
    cmdbuf.blitImage(staging, vk::ImageLayout::eTransferSrcOptimal, color.GetImage(),
                     vk::ImageLayout::eTransferDstOptimal, region, vk::Filter::eNearest);
    if (frames++ == 0) {
        LOG_INFO(Render_Vulkan, "Motion view: first frame drawn");
    }
}

} // namespace Vulkan::HostPasses
