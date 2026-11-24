// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/host_passes/frame_gen.h"

// DLSS frame generation through Streamline 2.14 (see frame_gen.h), ported from the owner's KytyPS5
// frame generation. The decisions live in frame_gen_policy.h. Nothing NVIDIA is linked:
// sl.interposer.dll is loaded from beside the exe and its exports are resolved by name.

#if OVERTIME_ENABLE_DLSSG

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "common/logging/log.h"
#include "common/process_exit.h"
#include "core/emulator_settings.h"
#include "video_core/renderer_vulkan/host_passes/frame_gen_policy.h"
#include "video_core/renderer_vulkan/host_passes/picture.h"

#include "sl.h"
#include "sl_consts.h"
#include "sl_dlss_g.h"
#include "sl_pcl.h"
#include "sl_reflex.h"
// Non-inline definitions: included by this file only.
#include "sl_security.h"

namespace Vulkan::HostPasses::FrameGen {

namespace {

// Overtime's own project id for Streamline (a self-generated GUID, as the KytyPS5 port uses).
constexpr const char* ProjectId = "bca9f413-41aa-46c2-b12b-f3bfc9e2130e";

// Streamline's own log (it then does not use OutputDebugString). Any thread.
void SlLog(sl::LogType type, const char* message) {
    if (message == nullptr) {
        return;
    }
    std::string_view text(message);
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.remove_suffix(1);
    }
    switch (type) {
    case sl::LogType::eError:
        LOG_ERROR(Render_Vulkan, "OVERTIME_FG: sl {}", text);
        break;
    case sl::LogType::eWarn:
        LOG_WARNING(Render_Vulkan, "OVERTIME_FG: sl {}", text);
        break;
    default:
        LOG_DEBUG(Render_Vulkan, "OVERTIME_FG: sl {}", text);
        break;
    }
}

const char* ResultName(sl::Result result) {
    switch (result) {
    case sl::Result::eOk:
        return "ok";
    case sl::Result::eErrorIO:
        return "io";
    case sl::Result::eErrorDriverOutOfDate:
        return "driver-out-of-date";
    case sl::Result::eErrorOSOutOfDate:
        return "os-out-of-date";
    case sl::Result::eErrorOSDisabledHWS:
        return "hw-scheduling-off";
    case sl::Result::eErrorDeviceNotCreated:
        return "device-not-created";
    case sl::Result::eErrorNoSupportedAdapterFound:
        return "no-supported-adapter";
    case sl::Result::eErrorAdapterNotSupported:
        return "adapter-not-supported";
    case sl::Result::eErrorNoPlugins:
        return "no-plugins";
    case sl::Result::eErrorVulkanAPI:
        return "vulkan-api";
    case sl::Result::eErrorReflexAPI:
        return "reflex-api";
    case sl::Result::eErrorNGXFailed:
        return "ngx-failed";
    case sl::Result::eErrorMissingInputParameter:
        return "missing-input";
    case sl::Result::eErrorNotInitialized:
        return "not-initialized";
    case sl::Result::eErrorInitNotCalled:
        return "init-not-called";
    case sl::Result::eErrorInvalidParameter:
        return "invalid-parameter";
    case sl::Result::eErrorMissingConstants:
        return "missing-constants";
    case sl::Result::eErrorInvalidIntegration:
        return "invalid-integration";
    case sl::Result::eErrorFeatureMissing:
        return "feature-missing";
    case sl::Result::eErrorFeatureNotSupported:
        return "feature-not-supported";
    case sl::Result::eErrorFeatureFailedToLoad:
        return "feature-failed-to-load";
    case sl::Result::eErrorInvalidState:
        return "invalid-state";
    case sl::Result::eWarnOutOfVRAM:
        return "out-of-vram";
    default:
        return "error";
    }
}

// The first failure bit of DLSSGState::status (a literal: ModeGate keeps it).
const char* StatusName(sl::DLSSGStatus status) {
    const auto bits = static_cast<u32>(status);
    const auto has = [bits](sl::DLSSGStatus bit) { return (bits & static_cast<u32>(bit)) != 0; };
    if (has(sl::DLSSGStatus::eFailResolutionTooLow)) {
        return "status-resolution-too-low";
    }
    if (has(sl::DLSSGStatus::eFailReflexNotDetectedAtRuntime)) {
        return "status-reflex";
    }
    if (has(sl::DLSSGStatus::eFailHDRFormatNotSupported)) {
        return "status-hdr-format";
    }
    if (has(sl::DLSSGStatus::eFailCommonConstantsInvalid)) {
        return "status-constants";
    }
    return "status-other";
}

bool EnvIs(const char* name, const char* value) {
    const char* v = std::getenv(name);
    return v != nullptr && std::strcmp(v, value) == 0;
}

// The settings, plus test switches read once before the instance:
//   OVERTIME_DLSSG_HOLD=1              Streamline loaded and fed, frame generation never on
//   OVERTIME_DLSSG_TAG_LIFETIME=present  tags valid until present (default: copied at tag time)
//   OVERTIME_DLSSG_REFLEX_SLEEP=n      0 never, 1 on the guest's flip submit (default), 2 also
//                                      on flips the emulator submits itself
//   OVERTIME_DLSSG_ALLOW_UNSIGNED=1    skip the NVIDIA signature check of sl.interposer.dll
struct Config {
    bool switch_on = false;
    bool hold = false;
    bool until_present = false;
    bool allow_unsigned = false;
    u32 frames = 1;
    int reflex_sleep = 1;
};

Config ReadConfig() {
    Config c;
    c.switch_on = Picture::DlssFg();
    c.frames = std::clamp<u32>(EmulatorSettings.GetFrameGenFrames(), 1, 3);
    c.hold = EnvIs("OVERTIME_DLSSG_HOLD", "1");
    c.until_present = EnvIs("OVERTIME_DLSSG_TAG_LIFETIME", "present");
    c.allow_unsigned = EnvIs("OVERTIME_DLSSG_ALLOW_UNSIGNED", "1");
    if (const char* v = std::getenv("OVERTIME_DLSSG_REFLEX_SLEEP")) {
        c.reflex_sleep = std::clamp(std::atoi(v), 0, 2);
    }
    return c;
}

struct Api {
    HMODULE module = nullptr;
    PFun_slInit* init = nullptr;
    PFun_slIsFeatureSupported* is_feature_supported = nullptr;
    PFun_slGetFeatureFunction* get_feature_function = nullptr;
    PFun_slGetNewFrameToken* new_frame_token = nullptr;
    PFun_slSetTagForFrame* set_tag_for_frame = nullptr;
    PFun_slSetConstants* set_constants = nullptr;
    PFN_vkGetInstanceProcAddr get_instance_proc = nullptr;
    PFN_vkGetDeviceProcAddr get_device_proc = nullptr;
    PFun_slDLSSGSetOptions* dlssg_set_options = nullptr;
    PFun_slDLSSGGetState* dlssg_get_state = nullptr;
    PFun_slPCLSetMarker* pcl_set_marker = nullptr;
    PFun_slReflexSleep* reflex_sleep = nullptr;
    PFun_slReflexSetOptions* reflex_set_options = nullptr;
};

struct State {
    Config cfg;
    Api api;

    std::atomic<bool> active{false};   // Streamline is the Vulkan loader
    std::atomic<bool> ready{false};    // device side set up, feature functions resolved
    std::atomic<bool> tags_off{false}; // a tag call failed: no more tags this session
    std::atomic<bool> reflex_on{false};
    std::atomic<const char*> pending_fail{nullptr}; // a failure seen off the present thread
    std::atomic<u32> api_errors{0};

    // Ledger, frame tokens, gate, sequencer, last size. Never held across a present or a sleep.
    std::mutex mutex;
    FrameLedger ledger;
    ModeGate gate;
    ModeSequencer sequencer;
    u64 started = 0; // index the flip-submit side started (simulation start marker)
    sl::FrameToken* present_token = nullptr;
    u32 frames_max = 0;
    u32 min_size = 0;
    u32 bad_status = 0;
    vk::Extent2D last_size{};
    u64 tagged = 0;
    u64 presents = 0;
    u64 new_frame_presents = 0;
    u64 multi_presents = 0; // presents where Streamline showed more than one frame
};

State& G() {
    static State state;
    return state;
}

const sl::ViewportHandle& Viewport() {
    static const sl::ViewportHandle viewport{0u};
    return viewport;
}

std::wstring ExeDirectory() {
    std::wstring path(MAX_PATH, L'\0');
    for (;;) {
        const DWORD length =
            GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
        if (length == 0) {
            return {};
        }
        if (length < path.size()) {
            path.resize(length);
            break;
        }
        path.resize(path.size() * 2);
    }
    const auto slash = path.find_last_of(L"\\/");
    return slash == std::wstring::npos ? std::wstring{} : path.substr(0, slash + 1);
}

std::wstring Widen(std::string_view ascii) {
    return {ascii.begin(), ascii.end()};
}

// The DLL a function pointer points into (for the dispatcher log).
std::string ModuleOf(const void* address) {
    if (address == nullptr) {
        return "null";
    }
    HMODULE module = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCWSTR>(address), &module) == 0) {
        return "unknown";
    }
    char path[MAX_PATH]{};
    if (GetModuleFileNameA(module, path, MAX_PATH) == 0) {
        return "unknown";
    }
    const std::string_view full(path);
    const auto slash = full.find_last_of("\\/");
    return std::string(slash == std::string_view::npos ? full : full.substr(slash + 1));
}

template <typename Function>
bool ResolveExport(HMODULE module, const char* name, Function*& out) {
    out = reinterpret_cast<Function*>(reinterpret_cast<void*>(GetProcAddress(module, name)));
    return out != nullptr;
}

template <typename Function>
bool ResolveFeature(sl::Feature feature, const char* name, Function*& out) {
    void* function = nullptr;
    if (G().api.get_feature_function(feature, name, function) != sl::Result::eOk ||
        function == nullptr) {
        out = nullptr;
        LOG_WARNING(Render_Vulkan, "OVERTIME_FG: feature-function name={} missing", name);
        return false;
    }
    out = reinterpret_cast<Function*>(function);
    return true;
}

// Points one dispatcher entry at the interposer's function and logs where it pointed before.
template <typename Pfn>
void Repoint(Pfn& entry, PFN_vkVoidFunction from_streamline, const char* name) {
    const std::string before = ModuleOf(reinterpret_cast<const void*>(entry));
    if (from_streamline != nullptr) {
        entry = reinterpret_cast<Pfn>(from_streamline);
    }
    LOG_INFO(Render_Vulkan, "OVERTIME_FG: dispatch {} before={} after={}", name, before,
             ModuleOf(reinterpret_cast<const void*>(entry)));
}

// Streamline's error callback runs on the present thread inside a failing present or acquire and
// must not block: count only.
void OnApiError(const sl::APIError& /*error*/) {
    G().api_errors.fetch_add(1, std::memory_order_relaxed);
}

PresentMode ToMode(vk::PresentModeKHR mode) {
    switch (mode) {
    case vk::PresentModeKHR::eMailbox:
        return PresentMode::Mailbox;
    case vk::PresentModeKHR::eImmediate:
        return PresentMode::Immediate;
    case vk::PresentModeKHR::eFifoRelaxed:
        return PresentMode::FifoRelaxed;
    default:
        return PresentMode::Fifo;
    }
}

// Caller holds the mutex. Streamline applies the options at the next present.
void SetModeLocked(bool on) {
    auto& g = G();
    sl::DLSSGOptions options{};
    options.mode = on ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
    options.numFramesToGenerate = FramesToGenerate(g.cfg.frames, g.frames_max);
    options.onErrorCallback = &OnApiError;
    const sl::Result result = g.api.dlssg_set_options(Viewport(), options);
    // DLSS-G needs Reflex on while it runs (sl_dlss_g.h).
    sl::ReflexOptions reflex{};
    reflex.mode = on ? sl::ReflexMode::eLowLatency : sl::ReflexMode::eOff;
    const sl::Result reflex_result = g.api.reflex_set_options(reflex);
    g.reflex_on.store(on && reflex_result == sl::Result::eOk, std::memory_order_relaxed);
    LOG_INFO(Render_Vulkan, "OVERTIME_FG: mode={} frames={} max={} result={} reflex={}",
             on ? "on" : "off", options.numFramesToGenerate, g.frames_max, ResultName(result),
             ResultName(reflex_result));
    if (on && result != sl::Result::eOk) {
        g.gate.Fail("set-options");
    }
}

sl::FrameToken* TokenLocked(u64 index) {
    sl::FrameToken* token = nullptr;
    const auto value = static_cast<u32>(index);
    if (G().api.new_frame_token(token, &value) != sl::Result::eOk) {
        return nullptr;
    }
    return token;
}

void Marker(sl::PCLMarker marker, const sl::FrameToken* token) {
    if (token != nullptr) {
        G().api.pcl_set_marker(marker, *token);
    }
}

sl::Resource MakeResource(vk::Image image, vk::ImageView view, vk::Format format,
                          vk::ImageUsageFlags usage, vk::Extent2D size) {
    sl::Resource resource(sl::ResourceType::eTex2d, static_cast<VkImage>(image), nullptr,
                          static_cast<VkImageView>(view),
                          static_cast<u32>(VK_IMAGE_LAYOUT_GENERAL));
    resource.width = size.width;
    resource.height = size.height;
    resource.nativeFormat = static_cast<u32>(format);
    resource.mipLevels = 1;
    resource.arrayLayers = 1;
    resource.flags = 0; // VkImageCreateFlags (must be set)
    resource.usage = static_cast<u32>(static_cast<VkImageUsageFlags>(usage));
    return resource;
}

sl::float4x4 Identity() {
    sl::float4x4 m;
    m.setRow(0, sl::float4(1.0f, 0.0f, 0.0f, 0.0f));
    m.setRow(1, sl::float4(0.0f, 1.0f, 0.0f, 0.0f));
    m.setRow(2, sl::float4(0.0f, 0.0f, 1.0f, 0.0f));
    m.setRow(3, sl::float4(0.0f, 0.0f, 0.0f, 1.0f));
    return m;
}

// The guest's camera is not known: a fixed perspective (fov_y 60 degrees, near 0.1, far 1000) at
// the render aspect, row-vector convention (sl_consts.h), depth 0 near and 1 far.
struct Projection {
    sl::float4x4 view_to_clip;
    sl::float4x4 clip_to_view;
    float near_plane;
    float far_plane;
    float fov_y;
    float aspect;
};

Projection FixedProjection(vk::Extent2D size) {
    Projection p{};
    p.near_plane = 0.1f;
    p.far_plane = 1000.0f;
    p.fov_y = 1.04719755f; // 60 degrees
    p.aspect =
        size.height == 0 ? 1.0f : static_cast<float>(size.width) / static_cast<float>(size.height);
    const float y = 1.0f / std::tan(p.fov_y * 0.5f);
    const float x = y / p.aspect;
    const float a = p.far_plane / (p.far_plane - p.near_plane);
    const float b = -p.near_plane * p.far_plane / (p.far_plane - p.near_plane);
    // clip = view * M: x' = x*X, y' = y*Y, z' = z*a + b, w' = z.
    p.view_to_clip.setRow(0, sl::float4(x, 0.0f, 0.0f, 0.0f));
    p.view_to_clip.setRow(1, sl::float4(0.0f, y, 0.0f, 0.0f));
    p.view_to_clip.setRow(2, sl::float4(0.0f, 0.0f, a, 1.0f));
    p.view_to_clip.setRow(3, sl::float4(0.0f, 0.0f, b, 0.0f));
    // The inverse: X = x'/x, Y = y'/y, Z = w', W = (z' - a*w')/b.
    p.clip_to_view.setRow(0, sl::float4(1.0f / x, 0.0f, 0.0f, 0.0f));
    p.clip_to_view.setRow(1, sl::float4(0.0f, 1.0f / y, 0.0f, 0.0f));
    p.clip_to_view.setRow(2, sl::float4(0.0f, 0.0f, 0.0f, 1.0f / b));
    p.clip_to_view.setRow(3, sl::float4(0.0f, 0.0f, 1.0f, -a / b));
    return p;
}

void LogExit() {
    auto& g = G();
    LOG_INFO(Render_Vulkan,
             "OVERTIME_FG: exit presents={} new_frames={} tagged={} multi={} api_errors={} "
             "gate={} failed={}",
             g.presents, g.new_frame_presents, g.tagged, g.multi_presents, g.api_errors.load(),
             g.gate.On() ? 1 : 0, g.gate.Failed() ? 1 : 0);
}

} // namespace

PFN_vkGetInstanceProcAddr LoaderBeforeInstance(PFN_vkGetInstanceProcAddr plain) {
    auto& g = G();
    g.cfg = ReadConfig();
    if (!g.cfg.switch_on) {
        return plain;
    }
    const std::wstring directory = ExeDirectory();
    std::string_view missing;
    for (const auto name : RequiredDlls) {
        const std::wstring path = directory + Widen(name);
        if (GetFileAttributesW(path.c_str()) == INVALID_FILE_ATTRIBUTES) {
            missing = name;
            break;
        }
    }
    const auto decision = DecideLoader({true, true, true, missing});
    if (decision.loader == Loader::Plain) {
        LOG_WARNING(Render_Vulkan, "OVERTIME_FG: loader=plain reason={}{}{}", decision.reason,
                    missing.empty() ? "" : " dll=", missing);
        return plain;
    }
    const std::wstring interposer = directory + L"sl.interposer.dll";
    if (!g.cfg.allow_unsigned && !sl::security::verifyEmbeddedSignature(interposer.c_str())) {
        LOG_WARNING(Render_Vulkan,
                    "OVERTIME_FG: loader=plain reason=unsigned dll=sl.interposer.dll");
        return plain;
    }
    auto& api = g.api;
    api.module = LoadLibraryW(interposer.c_str());
    if (api.module == nullptr) {
        LOG_WARNING(Render_Vulkan, "OVERTIME_FG: loader=plain reason=load-failed error={}",
                    GetLastError());
        return plain;
    }
    // From here Streamline is in the process (it stays loaded): end it without DLL detach.
    Common::SetTerminateOnExit(&LogExit);
    const bool resolved =
        ResolveExport(api.module, "slInit", api.init) &&
        ResolveExport(api.module, "slIsFeatureSupported", api.is_feature_supported) &&
        ResolveExport(api.module, "slGetFeatureFunction", api.get_feature_function) &&
        ResolveExport(api.module, "slGetNewFrameToken", api.new_frame_token) &&
        ResolveExport(api.module, "slSetTagForFrame", api.set_tag_for_frame) &&
        ResolveExport(api.module, "slSetConstants", api.set_constants);
    api.get_instance_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        reinterpret_cast<void*>(GetProcAddress(api.module, "vkGetInstanceProcAddr")));
    api.get_device_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
        reinterpret_cast<void*>(GetProcAddress(api.module, "vkGetDeviceProcAddr")));
    if (!resolved || api.get_instance_proc == nullptr || api.get_device_proc == nullptr) {
        // Loaded but never initialized: harmless, the plain loader runs as before.
        LOG_WARNING(Render_Vulkan, "OVERTIME_FG: loader=plain reason=missing-export");
        return plain;
    }

    static const sl::Feature features[] = {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL};
    static std::wstring plugin_directory;
    static const wchar_t* plugin_paths[1] = {};
    plugin_directory = directory;
    plugin_paths[0] = plugin_directory.c_str();
    sl::Preferences preferences{};
    preferences.showConsole = false;
    preferences.logLevel = sl::LogLevel::eDefault;
    preferences.pathsToPlugins = plugin_paths;
    preferences.numPathsToPlugins = 1;
    preferences.logMessageCallback = &SlLog;
    // No downloads at run time (no eAllowOTA / eLoadDownloadedPlugins); tags per frame token.
    preferences.flags = sl::PreferenceFlags::eDisableCLStateTracking |
                        sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    preferences.featuresToLoad = features;
    preferences.numFeaturesToLoad = static_cast<u32>(std::size(features));
    preferences.engine = sl::EngineType::eCustom;
    preferences.engineVersion = "1.0";
    preferences.projectId = ProjectId;
    preferences.renderAPI = sl::RenderAPI::eVulkan;
    const sl::Result result = api.init(preferences, sl::kSDKVersion);
    if (result != sl::Result::eOk) {
        LOG_WARNING(Render_Vulkan, "OVERTIME_FG: loader=plain reason=slInit result={} ({})",
                    ResultName(result), static_cast<int>(result));
        return plain;
    }
    {
        std::scoped_lock lock{g.mutex};
        g.gate = ModeGate(ModeGate::Config{60, 30, g.cfg.hold});
    }
    g.active.store(true);
    LOG_INFO(Render_Vulkan,
             "OVERTIME_FG: loader=streamline frames={} hold={} tag_lifetime={} reflex_sleep={} "
             "signature={}",
             g.cfg.frames, g.cfg.hold ? 1 : 0, g.cfg.until_present ? "present" : "now",
             g.cfg.reflex_sleep, g.cfg.allow_unsigned ? "skipped" : "ok");
    return api.get_instance_proc;
}

void AfterDevice(vk::Instance instance_handle, vk::PhysicalDevice physical_device,
                 vk::Device device_handle) {
    auto& g = G();
    if (!g.active.load()) {
        return;
    }
    const auto device = static_cast<VkDevice>(device_handle);
    const auto instance = static_cast<VkInstance>(instance_handle);

    // The functions Streamline hooks, from the interposer itself, whatever init(device) resolved.
    auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
    const auto gdp = g.api.get_device_proc;
    const auto gip = g.api.get_instance_proc;
    Repoint(d.vkQueuePresentKHR, gdp(device, "vkQueuePresentKHR"), "vkQueuePresentKHR");
    Repoint(d.vkCreateSwapchainKHR, gdp(device, "vkCreateSwapchainKHR"), "vkCreateSwapchainKHR");
    Repoint(d.vkDestroySwapchainKHR, gdp(device, "vkDestroySwapchainKHR"), "vkDestroySwapchainKHR");
    Repoint(d.vkGetSwapchainImagesKHR, gdp(device, "vkGetSwapchainImagesKHR"),
            "vkGetSwapchainImagesKHR");
    Repoint(d.vkAcquireNextImageKHR, gdp(device, "vkAcquireNextImageKHR"), "vkAcquireNextImageKHR");
    Repoint(d.vkDeviceWaitIdle, gdp(device, "vkDeviceWaitIdle"), "vkDeviceWaitIdle");
    Repoint(d.vkDestroySurfaceKHR, gip(instance, "vkDestroySurfaceKHR"), "vkDestroySurfaceKHR");

    sl::AdapterInfo adapter{};
    adapter.vkPhysicalDevice = static_cast<VkPhysicalDevice>(physical_device);
    const sl::Result supported = g.api.is_feature_supported(sl::kFeatureDLSS_G, adapter);
    LOG_INFO(Render_Vulkan, "OVERTIME_FG: adapter dlss_g={}", ResultName(supported));
    if (supported != sl::Result::eOk) {
        std::scoped_lock lock{g.mutex};
        g.gate.Fail("adapter");
        LOG_WARNING(Render_Vulkan,
                    "OVERTIME_FG: this card has no DLSS frame generation ({}): it stays off",
                    ResultName(supported));
        return;
    }
    auto& api = g.api;
    const bool resolved =
        ResolveFeature(sl::kFeatureDLSS_G, "slDLSSGSetOptions", api.dlssg_set_options) &&
        ResolveFeature(sl::kFeatureDLSS_G, "slDLSSGGetState", api.dlssg_get_state) &&
        ResolveFeature(sl::kFeaturePCL, "slPCLSetMarker", api.pcl_set_marker) &&
        ResolveFeature(sl::kFeatureReflex, "slReflexSleep", api.reflex_sleep) &&
        ResolveFeature(sl::kFeatureReflex, "slReflexSetOptions", api.reflex_set_options);
    if (!resolved) {
        std::scoped_lock lock{g.mutex};
        g.gate.Fail("feature-functions");
        return;
    }
    // Reflex options must be set at least once, even off.
    const sl::ReflexOptions reflex{};
    const sl::Result reflex_result = api.reflex_set_options(reflex);
    g.ready.store(true);
    LOG_INFO(Render_Vulkan, "OVERTIME_FG: ready reflex={}", ResultName(reflex_result));
}

bool Ready() {
    return G().ready.load(std::memory_order_relaxed);
}

vk::PresentModeKHR AdjustPresentMode(vk::PresentModeKHR requested, bool mailbox_available,
                                     bool immediate_available) {
    auto& g = G();
    if (!g.ready.load()) {
        return requested;
    }
    bool frame_gen = false;
    {
        std::scoped_lock lock{g.mutex};
        frame_gen = g.sequencer.FrameGenPresentMode();
    }
    if (!frame_gen) {
        return requested;
    }
    vk::PresentModeKHR result = requested;
    switch (PresentModeFor(true, ToMode(requested), mailbox_available, immediate_available)) {
    case PresentMode::Mailbox:
        result = vk::PresentModeKHR::eMailbox;
        break;
    case PresentMode::Immediate:
        result = vk::PresentModeKHR::eImmediate;
        break;
    default:
        break;
    }
    LOG_INFO(Render_Vulkan, "OVERTIME_FG: present-mode requested={} frame_gen={}",
             vk::to_string(requested), vk::to_string(result));
    return result;
}

bool BeforePresent(u32 width, u32 height, u64 index) {
    auto& g = G();
    if (!g.ready.load(std::memory_order_relaxed)) {
        return false;
    }
    std::scoped_lock lock{g.mutex};
    const bool new_frame = index != 0;
    g.presents++;
    if (new_frame) {
        g.new_frame_presents++;
    }
    if (const char* reason = g.pending_fail.exchange(nullptr)) {
        g.gate.Fail(reason);
    }
    sl::DLSSGState state{};
    const sl::Result state_result = g.api.dlssg_get_state(Viewport(), state, nullptr);
    if (state_result == sl::Result::eOk) {
        g.frames_max = state.numFramesToGenerateMax;
        g.min_size = state.minWidthOrHeight;
        if (state.numFramesActuallyPresented > 1) {
            g.multi_presents++;
        }
        // Recovery (DLSS-G guide 14.3): a status other than OK turns the mode off. A few
        // presents of grace after turning on, while Streamline sees the first markers.
        if (g.sequencer.Current() == ModeSequencer::State::On &&
            state.status != sl::DLSSGStatus::eOk) {
            if (++g.bad_status >= 10) {
                const char* reason = StatusName(state.status);
                g.gate.Fail(reason);
                LOG_WARNING(Render_Vulkan,
                            "OVERTIME_FG: status={:#x} {}: frame generation off for this session",
                            static_cast<u32>(state.status), reason);
            }
        } else {
            g.bad_status = 0;
        }
    }
    const bool window_ok = width != 0 && height != 0 && std::min(width, height) >= g.min_size;
    const auto step = g.gate.Update({new_frame, window_ok});
    const auto out = g.sequencer.BeforePresent(step.on);
    if (out.set_options) {
        SetModeLocked(out.options_on);
    }
    if (step.changed) {
        LOG_INFO(Render_Vulkan, "OVERTIME_FG: gate on={} reason={} present={}", step.on ? 1 : 0,
                 step.reason, g.presents);
    }
    g.present_token = new_frame ? TokenLocked(index) : nullptr;
    if (g.presents % 600 == 0) {
        LOG_INFO(Render_Vulkan,
                 "OVERTIME_FG: stats presents={} new_frames={} tagged={} multi={} state={} "
                 "gate={} status={:#x} min={} max={} api_errors={}",
                 g.presents, g.new_frame_presents, g.tagged, g.multi_presents,
                 static_cast<int>(g.sequencer.Current()), g.gate.On() ? 1 : 0,
                 static_cast<u32>(state.status), g.min_size, g.frames_max,
                 g.api_errors.load(std::memory_order_relaxed));
    }
    return out.request_recreate;
}

bool AfterFrameGenRecreate() {
    auto& g = G();
    if (!g.ready.load()) {
        return false;
    }
    std::scoped_lock lock{g.mutex};
    const auto out = g.sequencer.Recreated(g.gate.On());
    if (out.set_options) {
        SetModeLocked(out.options_on);
    }
    LOG_INFO(Render_Vulkan, "OVERTIME_FG: recreated state={} again={}",
             static_cast<int>(g.sequencer.Current()), out.request_recreate ? 1 : 0);
    return out.request_recreate;
}

void BeforeOutsideRecreate() {
    auto& g = G();
    if (!g.ready.load()) {
        return;
    }
    std::scoped_lock lock{g.mutex};
    const auto out = g.sequencer.OutsideRecreate();
    if (out.set_options) {
        SetModeLocked(false);
    }
}

void PresentStart() {
    auto& g = G();
    if (!g.ready.load(std::memory_order_relaxed)) {
        return;
    }
    sl::FrameToken* token = nullptr;
    {
        std::scoped_lock lock{g.mutex};
        token = g.present_token;
    }
    Marker(sl::PCLMarker::ePresentStart, token);
}

void PresentEnd() {
    auto& g = G();
    if (!g.ready.load(std::memory_order_relaxed)) {
        return;
    }
    sl::FrameToken* token = nullptr;
    {
        std::scoped_lock lock{g.mutex};
        token = g.present_token;
    }
    Marker(sl::PCLMarker::ePresentEnd, token);
}

void FlipSubmitted(bool guest_thread) {
    auto& g = G();
    if (!g.ready.load(std::memory_order_relaxed)) {
        return;
    }
    sl::FrameToken* ending = nullptr;
    sl::FrameToken* next = nullptr;
    {
        std::scoped_lock lock{g.mutex};
        if (g.started != 0) {
            ending = TokenLocked(g.started);
        }
        g.started = std::max(g.started + 1, g.ledger.LastTagged() + 1);
        next = TokenLocked(g.started);
    }
    Marker(sl::PCLMarker::eSimulationEnd, ending);
    Marker(sl::PCLMarker::eRenderSubmitStart, ending);
    Marker(sl::PCLMarker::eRenderSubmitEnd, ending);
    const bool may_sleep = g.cfg.reflex_sleep == 2 || (g.cfg.reflex_sleep == 1 && guest_thread);
    if (next != nullptr && may_sleep && g.reflex_on.load(std::memory_order_relaxed)) {
        g.api.reflex_sleep(*next);
    }
    Marker(sl::PCLMarker::eSimulationStart, next);
}

u64 TagFrame(vk::CommandBuffer cmdbuf, const TagInputs& inputs) {
    auto& g = G();
    if (!g.ready.load(std::memory_order_relaxed) || g.tags_off.load(std::memory_order_relaxed) ||
        !cmdbuf || !inputs.depth || !inputs.motion || !inputs.depth_view || !inputs.motion_view ||
        inputs.size.width == 0 || inputs.size.height == 0) {
        return 0;
    }
    std::scoped_lock lock{g.mutex};
    const u64 index = g.ledger.Tag(g.started);
    sl::FrameToken* const token = TokenLocked(index);
    if (token == nullptr) {
        g.tags_off.store(true);
        g.pending_fail.store("tag-token");
        LOG_WARNING(Render_Vulkan, "OVERTIME_FG: tag failed index={}: frame token", index);
        return 0;
    }

    // Constants: matrices without jitter (the velocity pass has none), motion in pixels from
    // current to previous, so mvecScale takes them to [-1, 1] screen units (sl_consts.h). The
    // camera is unknown: a fixed perspective, an identity view and no camera motion between
    // frames (the motion vectors carry all of it).
    const bool reset = g.last_size != inputs.size;
    const Projection projection = FixedProjection(inputs.size);
    sl::Constants c{};
    c.cameraViewToClip = projection.view_to_clip;
    c.clipToCameraView = projection.clip_to_view;
    c.clipToLensClip = Identity();
    c.clipToPrevClip = Identity();
    c.prevClipToClip = Identity();
    c.jitterOffset = {0.0f, 0.0f};
    c.mvecScale = {1.0f / static_cast<float>(inputs.size.width),
                   1.0f / static_cast<float>(inputs.size.height)};
    c.cameraPinholeOffset = {0.0f, 0.0f};
    c.cameraPos = {0.0f, 0.0f, 0.0f};
    c.cameraUp = {0.0f, 1.0f, 0.0f};
    c.cameraRight = {1.0f, 0.0f, 0.0f};
    c.cameraFwd = {0.0f, 0.0f, 1.0f};
    c.cameraNear = projection.near_plane;
    c.cameraFar = projection.far_plane;
    c.cameraFOV = projection.fov_y;
    c.cameraAspectRatio = projection.aspect;
    c.depthInverted = sl::Boolean::eFalse;
    c.cameraMotionIncluded = sl::Boolean::eTrue;
    c.motionVectors3D = sl::Boolean::eFalse;
    c.reset = reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    c.orthographicProjection = sl::Boolean::eFalse;
    c.motionVectorsDilated = sl::Boolean::eFalse;
    c.motionVectorsJittered = sl::Boolean::eFalse;
    const sl::Result constants_result = g.api.set_constants(c, *token, Viewport());

    sl::Resource depth = MakeResource(inputs.depth, inputs.depth_view, vk::Format::eR32Sfloat,
                                      inputs.depth_usage, inputs.size);
    sl::Resource motion = MakeResource(inputs.motion, inputs.motion_view, vk::Format::eR16G16Sfloat,
                                       inputs.motion_usage, inputs.size);
    const sl::ResourceLifecycle lifecycle = g.cfg.until_present
                                                ? sl::ResourceLifecycle::eValidUntilPresent
                                                : sl::ResourceLifecycle::eOnlyValidNow;
    const sl::Extent extent{0, 0, inputs.size.width, inputs.size.height};
    const sl::ResourceTag tags[] = {
        sl::ResourceTag(&depth, sl::kBufferTypeDepth, lifecycle, &extent),
        sl::ResourceTag(&motion, sl::kBufferTypeMotionVectors, lifecycle, &extent),
    };
    const sl::Result tag_result =
        g.api.set_tag_for_frame(*token, Viewport(), tags, static_cast<u32>(std::size(tags)),
                                static_cast<VkCommandBuffer>(cmdbuf));
    if (constants_result != sl::Result::eOk || tag_result != sl::Result::eOk) {
        g.tags_off.store(true);
        g.pending_fail.store("tag");
        LOG_WARNING(Render_Vulkan,
                    "OVERTIME_FG: tag failed index={} constants={} tags={}: no more tags this "
                    "session",
                    index, ResultName(constants_result), ResultName(tag_result));
        return 0;
    }
    g.last_size = inputs.size;
    if (++g.tagged <= 3) {
        LOG_INFO(Render_Vulkan, "OVERTIME_FG: tag index={} size={}x{} lifetime={} reset={}", index,
                 inputs.size.width, inputs.size.height, g.cfg.until_present ? "present" : "now",
                 reset ? 1 : 0);
    }
    return index;
}

} // namespace Vulkan::HostPasses::FrameGen

#endif // OVERTIME_ENABLE_DLSSG
