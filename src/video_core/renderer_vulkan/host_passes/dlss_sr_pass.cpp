// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/host_passes/dlss_sr_pass.h"

#if defined(_WIN32) && defined(_MSC_VER)

#include <array>
#include <filesystem>
#include <string>
#include <string_view>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include "common/logging/log.h"
#include "common/path_util.h"
#include "video_core/renderer_vulkan/host_passes/ngx_abi.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan::HostPasses {

namespace {

constexpr vk::Format OutputFormat = vk::Format::eR16G16B16A16Sfloat;
// NGX feature and value ids (public DLSS SDK 310.9.16, nvsdk_ngx_defs.h).
constexpr int FeatureSuperSampling = 1;
constexpr int PerfQualityMaxQuality = 2;
constexpr int PerfQualityDlaa = 5;
constexpr int FlagMvLowRes = 1 << 1;

bool Succeeded(Ngx::Result result) {
    return (static_cast<unsigned>(result) & 0xFFF00000u) != 0xBAD00000u;
}

// The driver core's entry points (the SDK's loader passes the FeatureCommonInfo last).
using PfnInitExt = Ngx::Result (*)(unsigned long long, const wchar_t*, VkInstance,
                                   VkPhysicalDevice, VkDevice, int, const Ngx::FeatureCommonInfo*);
using PfnInitProjectId = Ngx::Result (*)(const char*, int, const char*, const wchar_t*, VkInstance,
                                         VkPhysicalDevice, VkDevice, int,
                                         const Ngx::FeatureCommonInfo*);
using Ngx::ProjectId;

void NgxLog(const char* message, int, int) {
    std::string_view line{message ? message : ""};
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
        line.remove_suffix(1);
    }
    LOG_INFO(Render_Vulkan, "NGX: {}", line);
}

std::wstring ExecutableDir() {
    std::array<wchar_t, 32768> path{};
    const DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    return std::filesystem::path{std::wstring{path.data(), length}}.parent_path().wstring();
}
using PfnParameters = Ngx::Result (*)(Ngx::Parameter**);
using PfnDestroy = Ngx::Result (*)(Ngx::Parameter*);
using PfnCreate1 = Ngx::Result (*)(VkDevice, VkCommandBuffer, int, Ngx::Parameter*, void**);
using PfnEvaluate = Ngx::Result (*)(VkCommandBuffer, const void*, const Ngx::Parameter*, void*);
using PfnRelease = Ngx::Result (*)(void*);

std::filesystem::path NgxCoreDir() {
    std::array<wchar_t, MAX_PATH> path{};
    DWORD size = static_cast<DWORD>(path.size() * sizeof(wchar_t));
    if (RegGetValueW(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Services\\nvlddmkm\\NGXCore",
                     L"NGXPath", RRF_RT_REG_SZ, nullptr, path.data(), &size) != ERROR_SUCCESS) {
        return {};
    }
    return std::filesystem::path{path.data()};
}

} // namespace

struct DlssSrPass::Impl {
    Impl(const Instance& instance_, Scheduler& scheduler_, Runtime& runtime_)
        : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
          output{instance.GetDevice(), instance.GetAllocator()} {}

    ~Impl() {
        const vk::Device device = instance.GetDevice();
        if (feature && release) {
            (void)device.waitIdle();
            release(feature);
        }
        if (parameters && destroy) {
            destroy(parameters);
        }
        // NGX itself is left up: the DLSS 5 pass may share the driver's core on this device.
    }

    void Fail(const char* reason) {
        failed = true;
        LOG_ERROR(Render_Vulkan, "DLSS: {}; FSR 1 is used for this session", reason);
    }

    bool Initialise();
    void ResizeOutput(vk::Extent2D extent);
    vk::ImageView Render(VideoCore::Image& color, vk::ImageView color_view, vk::Format color_format,
                         const XfbVelocityPass::Outputs& inputs, vk::Extent2D output_size);

    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;

    HMODULE core{};
    std::wstring exe_dir;
    const wchar_t* exe_dir_ptr{};
    Ngx::FeatureCommonInfo common{};
    PfnDestroy destroy{};
    PfnCreate1 create{};
    PfnEvaluate evaluate{};
    PfnRelease release{};
    Ngx::Parameter* parameters{};
    void* feature{};
    bool initialised{};
    bool failed{};
    bool reset{true};
    bool mismatch_logged{};
    u64 frames{};

    vk::Extent2D render_size{};
    vk::Extent2D out_size{};
    VideoCore::UniqueImage output;
    vk::UniqueImageView output_view;
};

bool DlssSrPass::Impl::Initialise() {
    initialised = true;
    const auto core_dir = NgxCoreDir();
    if (core_dir.empty()) {
        Fail("the driver's NGX core is not registered (NGXCore\\NGXPath)");
        return false;
    }
    core = LoadLibraryW((core_dir / L"_nvngx.dll").c_str());
    if (!core) {
        core = LoadLibraryW((core_dir / L"nvngx.dll").c_str());
    }
    if (!core) {
        Fail("the driver's NGX core did not load");
        return false;
    }
    const auto init = reinterpret_cast<PfnInitExt>(GetProcAddress(core, "NVSDK_NGX_VULKAN_Init_Ext"));
    const auto capability = reinterpret_cast<PfnParameters>(
        GetProcAddress(core, "NVSDK_NGX_VULKAN_GetCapabilityParameters"));
    destroy = reinterpret_cast<PfnDestroy>(
        GetProcAddress(core, "NVSDK_NGX_VULKAN_DestroyParameters"));
    create = reinterpret_cast<PfnCreate1>(GetProcAddress(core, "NVSDK_NGX_VULKAN_CreateFeature1"));
    evaluate =
        reinterpret_cast<PfnEvaluate>(GetProcAddress(core, "NVSDK_NGX_VULKAN_EvaluateFeature"));
    release =
        reinterpret_cast<PfnRelease>(GetProcAddress(core, "NVSDK_NGX_VULKAN_ReleaseFeature"));
    if (!init || !capability || !create || !evaluate || !release) {
        Fail("the NGX core is missing Vulkan entry points");
        return false;
    }
    const auto data_path = Common::FS::GetUserPath(Common::FS::PathType::LogDir).wstring();
    // The player's nvngx_dlss.dll sits beside the executable; NGX's log lines go to ours.
    exe_dir = ExecutableDir();
    exe_dir_ptr = exe_dir.c_str();
    common = Ngx::FeatureCommonInfo{
        .paths = &exe_dir_ptr,
        .path_count = 1,
        .internal_data = nullptr,
        .log_callback = NgxLog,
        .minimum_log_level = 1,
        .disable_other_sinks = false,
    };
    const auto vk_instance = static_cast<VkInstance>(instance.GetInstance());
    const auto physical = static_cast<VkPhysicalDevice>(instance.GetPhysicalDevice());
    const auto device = static_cast<VkDevice>(instance.GetDevice());
    Ngx::Result init_result = 0;
    const auto init_project =
        reinterpret_cast<PfnInitProjectId>(GetProcAddress(core, "NVSDK_NGX_VULKAN_Init_ProjectID"));
    if (init_project) {
        init_result = init_project(ProjectId, Ngx::EngineTypeCustom, "1.0", data_path.c_str(),
                                   vk_instance, physical, device, Ngx::VersionApi, &common);
        LOG_INFO(Render_Vulkan, "DLSS: NGX init (project id) {:#x}",
                 static_cast<u32>(init_result));
    }
    if (!init_project || !Succeeded(init_result)) {
        init_result = init(0, data_path.c_str(), vk_instance, physical, device, Ngx::VersionApi,
                           &common);
        LOG_INFO(Render_Vulkan, "DLSS: NGX init {:#x}", static_cast<u32>(init_result));
    }
    if (!Succeeded(init_result)) {
        Fail("NGX would not initialise on this device");
        return false;
    }
    const Ngx::Result caps = capability(&parameters);
    if (!Succeeded(caps) || !parameters) {
        parameters = nullptr;
        Fail("NGX gave no capability parameters");
        return false;
    }
    unsigned int available = 0;
    int needs_driver = 0;
    parameters->Get("SuperSampling.Available", &available);
    parameters->Get("SuperSampling.NeedsUpdatedDriver", &needs_driver);
    int feature_init = 0;
    parameters->Get("SuperSampling.FeatureInitResult", &feature_init);
    LOG_INFO(Render_Vulkan,
             "DLSS: Super Resolution available {}, needs a newer driver {}, feature init {:#x}",
             available, needs_driver, static_cast<u32>(feature_init));
    if (!available) {
        Fail("DLSS Super Resolution is not available (an RTX GPU and nvngx_dlss.dll beside the "
             "executable are needed)");
        return false;
    }
    return true;
}

void DlssSrPass::Impl::ResizeOutput(vk::Extent2D extent) {
    if (out_size == extent && output_view) {
        return;
    }
    if (output_view) {
        scheduler.Finish();
    }
    output_view.reset();
    output.Destroy();
    out_size = extent;
    output.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = OutputFormat,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                 vk::ImageUsageFlagBits::eTransferSrc,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    const vk::Device device = instance.GetDevice();
    SetObjectName(device, static_cast<vk::Image>(output), "DLSS Output");
    output_view = Check<"create dlss output view">(
        device.createImageViewUnique(vk::ImageViewCreateInfo{
            .image = output,
            .viewType = vk::ImageViewType::e2D,
            .format = OutputFormat,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .levelCount = 1,
                .layerCount = 1,
            },
        }));
}

vk::ImageView DlssSrPass::Impl::Render(VideoCore::Image& color, vk::ImageView color_view,
                                       vk::Format color_format,
                                       const XfbVelocityPass::Outputs& inputs,
                                       vk::Extent2D output_size) {
    if (failed || (!initialised && !Initialise())) {
        return {};
    }
    const vk::Extent2D extent{color.info.size.width, color.info.size.height};
    if (extent != inputs.size || extent.width > output_size.width ||
        extent.height > output_size.height) {
        if (!mismatch_logged) {
            LOG_WARNING(Render_Vulkan,
                        "DLSS: presented image {}x{}, motion targets {}x{}, output {}x{}: not run",
                        extent.width, extent.height, inputs.size.width, inputs.size.height,
                        output_size.width, output_size.height);
            mismatch_logged = true;
        }
        return {};
    }

    // Sized work first: it may finish the command buffer.
    if (feature && (render_size != extent || out_size != output_size)) {
        scheduler.Finish();
        release(feature);
        feature = nullptr;
    }
    ResizeOutput(output_size);

    const vk::CommandBuffer cmdbuf = scheduler.CommandBuffer();
    scheduler.EndRendering();
    const auto command_buffer = static_cast<VkCommandBuffer>(cmdbuf);

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
            .dstStageMask = all,
            .dstAccessMask = any_access,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = output,
            .subresourceRange = range,
        },
        vk::ImageMemoryBarrier2{
            .srcStageMask = all,
            .srcAccessMask = any_access,
            .dstStageMask = all,
            .dstAccessMask = any_access,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = inputs.motion,
            .subresourceRange = range,
        },
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<u32>(start.size()),
        .pImageMemoryBarriers = start.data(),
    });

    if (!feature) {
        const bool same_size = extent == output_size;
        parameters->Set("Width", extent.width);
        parameters->Set("Height", extent.height);
        parameters->Set("OutWidth", output_size.width);
        parameters->Set("OutHeight", output_size.height);
        parameters->Set("PerfQualityValue", same_size ? PerfQualityDlaa : PerfQualityMaxQuality);
        parameters->Set("RTXValue", 0);
        parameters->Set("FreeMemOnReleaseFeature", 0);
        parameters->Set("CreationNodeMask", 1u);
        parameters->Set("VisibilityNodeMask", 1u);
        parameters->Set("DLSS.Enable.Output.Subrects", 0);
        // LDR input, depth not inverted (clears at 1.0 far), motion at the presented image's size.
        parameters->Set("DLSS.Feature.Create.Flags", FlagMvLowRes);
        const Ngx::Result result = create(static_cast<VkDevice>(instance.GetDevice()),
                                          command_buffer, FeatureSuperSampling, parameters,
                                          &feature);
        LOG_INFO(Render_Vulkan, "DLSS: feature create {:#x}, {}x{} -> {}x{} ({})",
                 static_cast<u32>(result), extent.width, extent.height, output_size.width,
                 output_size.height, same_size ? "DLAA" : "quality");
        if (!Succeeded(result) || !feature) {
            feature = nullptr;
            Fail("the Super Resolution feature would not build (is nvngx_dlss.dll beside the "
                 "executable?)");
            return {};
        }
        render_size = extent;
        reset = true;
    }

    const auto resource = [](vk::Image image, vk::ImageView view, vk::Format format,
                             vk::Extent2D size, bool read_write) {
        Ngx::ResourceVk out{};
        out.resource.image_view = {
            .image_view = static_cast<VkImageView>(view),
            .image = static_cast<VkImage>(image),
            .subresource_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
            .format = static_cast<VkFormat>(format),
            .width = size.width,
            .height = size.height,
        };
        out.type = Ngx::ResourceTypeVk::ImageView;
        out.read_write = read_write;
        return out;
    };
    Ngx::ResourceVk color_resource =
        resource(color.GetImage(), color_view, color_format, extent, false);
    Ngx::ResourceVk depth_resource =
        resource(inputs.depth, inputs.depth_view, vk::Format::eR32Sfloat, extent, false);
    Ngx::ResourceVk motion_resource =
        resource(inputs.motion, inputs.motion_view, vk::Format::eR16G16Sfloat, extent, false);
    Ngx::ResourceVk output_resource =
        resource(output, *output_view, OutputFormat, output_size, true);
    parameters->Set("Color", static_cast<void*>(&color_resource));
    parameters->Set("Depth", static_cast<void*>(&depth_resource));
    parameters->Set("MotionVectors", static_cast<void*>(&motion_resource));
    parameters->Set("Output", static_cast<void*>(&output_resource));
    parameters->Set("Jitter.Offset.X", 0.0f);
    parameters->Set("Jitter.Offset.Y", 0.0f);
    // Vectors are in pixels, current to previous, as DLSS reads them.
    parameters->Set("MV.Scale.X", 1.0f);
    parameters->Set("MV.Scale.Y", 1.0f);
    parameters->Set("Reset", reset ? 1 : 0);
    parameters->Set("Sharpness", 0.0f);
    parameters->Set("DLSS.Render.Subrect.Dimensions.Width", extent.width);
    parameters->Set("DLSS.Render.Subrect.Dimensions.Height", extent.height);
    const Ngx::Result evaluated = evaluate(command_buffer, feature, parameters, nullptr);
    reset = false;
    if (!Succeeded(evaluated)) {
        LOG_ERROR(Render_Vulkan, "DLSS: evaluate {:#x}", static_cast<u32>(evaluated));
        Fail("the Super Resolution feature refused to evaluate");
        return {};
    }
    if (frames++ == 0) {
        LOG_INFO(Render_Vulkan, "DLSS: first frame upscaled");
    }

    const vk::ImageMemoryBarrier2 done{
        .srcStageMask = all,
        .srcAccessMask = any_access,
        .dstStageMask = all,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .image = output,
        .subresourceRange = range,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &done,
    });
    return *output_view;
}

DlssSrPass::DlssSrPass(const Instance& instance, Scheduler& scheduler, Runtime& runtime)
    : impl{std::make_unique<Impl>(instance, scheduler, runtime)} {}

DlssSrPass::~DlssSrPass() = default;

vk::ImageView DlssSrPass::Render(VideoCore::Image& color, vk::ImageView color_view,
                                 vk::Format color_format, const XfbVelocityPass::Outputs& inputs,
                                 vk::Extent2D output_size) {
    return impl->Render(color, color_view, color_format, inputs, output_size);
}

} // namespace Vulkan::HostPasses

#else

namespace Vulkan::HostPasses {

struct DlssSrPass::Impl {};

DlssSrPass::DlssSrPass(const Instance&, Scheduler&, Runtime&) {}

DlssSrPass::~DlssSrPass() = default;

vk::ImageView DlssSrPass::Render(VideoCore::Image&, vk::ImageView, vk::Format,
                                 const XfbVelocityPass::Outputs&, vk::Extent2D) {
    return {};
}

} // namespace Vulkan::HostPasses

#endif
