// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

// Overtime Native v3: DLSS frame generation (DLSS-G) through NVIDIA Streamline 2.14, experimental.
// GPU.frame_gen = "dlss" with sl.interposer.dll and the rest of the DLL set beside the exe: the
// interposer becomes the Vulkan loader, the velocity pass's depth and motion vectors are tagged
// per frame, and Streamline adds GPU.frame_gen_frames (1..3) generated frames at each present.
// A missing DLL, an unsigned interposer or a card without DLSS-G gives one log line and the plain
// path. Log lines start with "OVERTIME_FG:". Built only on Windows with OVERTIME_ENABLE_DLSSG;
// elsewhere every call below does nothing.
namespace Vulkan::HostPasses::FrameGen {

/// The velocity pass's outputs of one frame, in eGeneral.
struct TagInputs {
    vk::Image depth; // R32F, not inverted (1.0 far)
    vk::ImageView depth_view;
    vk::ImageUsageFlags depth_usage;
    vk::Image motion; // RG16F, pixels, current to previous
    vk::ImageView motion_view;
    vk::ImageUsageFlags motion_usage;
    vk::Extent2D size;
};

#if OVERTIME_ENABLE_DLSSG

/// Before the instance (windowed only): Streamline's vkGetInstanceProcAddr, or `plain`.
PFN_vkGetInstanceProcAddr LoaderBeforeInstance(PFN_vkGetInstanceProcAddr plain);

/// After vkCreateDevice and the dispatcher's device init: the adapter check and feature functions.
void AfterDevice(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device);

/// Streamline is set up and frame generation may turn on.
bool Ready();

/// The present mode for the next swapchain (mailbox or immediate while frame generation is on).
vk::PresentModeKHR AdjustPresentMode(vk::PresentModeKHR requested, bool mailbox_available,
                                     bool immediate_available);

/// Present thread, before acquiring. index: the number of the frame about to be shown (its flip's,
/// from FlipSubmitted), 0 for a repeat or blank frame. Returns true when the swapchain must be
/// recreated first (RecreateForFrameGen, then AfterFrameGenRecreate until it returns false).
bool BeforePresent(u32 width, u32 height, u64 index);
bool AfterFrameGenRecreate();

/// Present thread, before showing the last frame again: true while frame generation is on, and it
/// is not shown again. Every present must carry its frame's Reflex markers (DLSS-G turns itself
/// off otherwise); the generated frames fill the gap.
bool SkipRepeat();

/// Before any other swapchain recreate (resize, out of date, HDR): frame generation off first.
void BeforeOutsideRecreate();

/// Around vkQueuePresentKHR (present thread).
void PresentStart();
void PresentEnd();

/// A guest flip was submitted: the simulation markers, and Reflex's sleep while frame generation
/// is on (on the guest's thread, by default). Returns the flip's frame number (0 before Streamline
/// is set up), which rides with the flip to FlipExecuting.
u64 FlipSubmitted(bool guest_thread);

/// GPU thread, before the flip's frame is prepared: `frame` is FlipSubmitted's number for it. The
/// guest may run frames ahead of the GPU; its tags and its present use this number, so every call
/// for one frame carries the same frame token.
void FlipExecuting(u64 frame);

/// The frame the GPU thread prepares now (FlipExecuting's), 0 before Streamline is set up. A frame
/// without tags still carries it to its present (the present markers).
u64 FrameIndex();

/// GPU thread, after the frame's host passes: tags depth and motion vectors with the constants.
/// Returns the frame's number for its present (also when the tags failed), 0 before set up.
u64 TagFrame(vk::CommandBuffer cmdbuf, const TagInputs& inputs);

#else

inline PFN_vkGetInstanceProcAddr LoaderBeforeInstance(PFN_vkGetInstanceProcAddr plain) {
    return plain;
}
inline void AfterDevice(vk::Instance, vk::PhysicalDevice, vk::Device) {}
inline bool Ready() {
    return false;
}
inline vk::PresentModeKHR AdjustPresentMode(vk::PresentModeKHR requested, bool, bool) {
    return requested;
}
inline bool BeforePresent(u32, u32, u64) {
    return false;
}
inline bool AfterFrameGenRecreate() {
    return false;
}
inline bool SkipRepeat() {
    return false;
}
inline void BeforeOutsideRecreate() {}
inline void PresentStart() {}
inline void PresentEnd() {}
inline u64 FlipSubmitted(bool) {
    return 0;
}
inline void FlipExecuting(u64) {}
inline u64 FrameIndex() {
    return 0;
}
inline u64 TagFrame(vk::CommandBuffer, const TagInputs&) {
    return 0;
}

#endif

} // namespace Vulkan::HostPasses::FrameGen
