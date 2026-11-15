// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

#include "common/assert.h"
#include "common/debug.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

std::mutex Scheduler::submit_mutex;

Scheduler::Scheduler(const Instance& instance)
    : instance{instance}, work_semaphore{instance}, command_pool{instance, &work_semaphore} {
#if TRACY_GPU_ENABLED
    profiler_scope = reinterpret_cast<tracy::VkCtxScope*>(std::malloc(sizeof(tracy::VkCtxScope)));
#endif
    BeginSession();
    priority_pending_ops_thread =
        std::jthread(std::bind_front(&Scheduler::PriorityPendingOpsThread, this));
}

Scheduler::~Scheduler() {
    StopRecorder();
#if TRACY_GPU_ENABLED
    std::free(profiler_scope);
#endif
}

void Scheduler::SubmitOutOfBand(vk::CommandBuffer cmdbuf, vk::Fence fence, u64 wait_tick) const {
    // The submission signalling wait_tick makes the GPU writes before it available, the wait
    // makes them visible to the copies.
    const vk::Semaphore timeline = work_semaphore.Handle();
    const vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eTransfer;
    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = 1,
        .pWaitSemaphoreValues = &wait_tick,
    };
    const bool own_queue = instance.HasReadbackQueue();
    const vk::SubmitInfo submit_info = {
        .pNext = own_queue ? &timeline_si : nullptr,
        .waitSemaphoreCount = own_queue ? 1U : 0U,
        .pWaitSemaphores = &timeline,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmdbuf,
    };
    // Both queues are externally synchronized; without a readback queue it is the graphics one.
    std::scoped_lock lk{submit_mutex};
    const auto submit_result = instance.GetReadbackQueue().submit(submit_info, fence);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");
}

static void RecordBeginRendering(vk::CommandBuffer cmdbuf, const RenderState& render_state) {
    std::array<vk::RenderingAttachmentInfo, 8> color_attachments;
    for (u32 i = 0; i < render_state.num_color_attachments; ++i) {
        const auto& cb = render_state.color_attachments[i];
        color_attachments[i] = vk::RenderingAttachmentInfo{
            .imageView = cb.image_view,
            .imageLayout = cb.image_layout,
            .loadOp = cb.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.uint32 = cb.clear_value}},
        };
    }

    const auto& db = render_state.depth_stencil_attachment;
    const vk::RenderingAttachmentInfo depth_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue =
            vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = std::bit_cast<float>(
                                                                          db.clear_value[0])}},
    };
    const vk::RenderingAttachmentInfo stencil_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.depthStencil =
                                         vk::ClearDepthStencilValue{.stencil = db.clear_value[1]}},
    };

    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .offset = {0, 0},
                .extent = {render_state.width, render_state.height},
            },
        .layerCount = render_state.num_layers,
        .colorAttachmentCount = render_state.num_color_attachments,
        .pColorAttachments = color_attachments.data(),
        .pDepthAttachment = db.has_depth ? &depth_attachment : nullptr,
        .pStencilAttachment = db.has_stencil ? &stencil_attachment : nullptr,
    };

    cmdbuf.beginRendering(rendering_info);
}

void Scheduler::BeginRendering(const RenderState& new_state) {
    if (is_rendering && render_state == new_state) {
        return;
    }
    EndRendering();
    is_rendering = true;
    render_state = new_state;
    RecordBeginRendering(CommandBuffer(), render_state);
}

void Scheduler::BeginRendering(const RenderState& new_state, DrawPacket& packet) {
    if (is_rendering && render_state == new_state) {
        return;
    }
    packet.end_rendering = is_rendering;
    packet.begin_rendering = true;
    packet.render_state = new_state;
    is_rendering = true;
    render_state = new_state;
}

void Scheduler::EndRendering() {
    if (!is_rendering) {
        return;
    }
    is_rendering = false;
    CommandBuffer().endRendering();
}

vk::CommandBuffer Scheduler::UploadCommandBuffer() {
    auto& upload_cmdbuf = sessions.back().upload;
    if (upload_cmdbuf) {
        return upload_cmdbuf;
    }
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    upload_cmdbuf = command_pool.Commit();
    Check(upload_cmdbuf.begin(begin_info));
    return upload_cmdbuf;
}

void Scheduler::Flush(SubmitInfo& info) {
    // When flushing, we only send data to the driver; no waiting is necessary.
    SubmitExecution(info);
}

void Scheduler::Flush() {
    SubmitInfo info{};
    Flush(info);
}

void Scheduler::Finish() {
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    SubmitInfo info{};
    SubmitExecution(info);
    Wait(presubmit_tick);
}

void Scheduler::Wait(u64 tick) {
    if (tick >= work_semaphore.CurrentTick()) {
        // Make sure we are not waiting for the current tick without signalling
        SubmitInfo info{};
        Flush(info);
    }
    work_semaphore.Wait(tick);
}

void Scheduler::PopPendingOperations() {
    if (pending_ops_empty.load(std::memory_order_relaxed)) {
        return;
    }
    std::unique_lock lk(pending_ops_mutex);
    work_semaphore.Refresh();
    while (!pending_ops.empty() && work_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        pending_ops.front().callback();
        pending_ops.pop();
    }
    pending_ops_empty.store(pending_ops.empty(), std::memory_order_relaxed);
}

void Scheduler::BeginSession() {
    EndSession();

    auto& session = sessions.emplace_back();

    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    session.primary = primary_pool ? primary_pool->Commit() : command_pool.Commit();
    Check(session.primary.begin(begin_info));

    // Invalidate dynamic state so it gets applied to the new command buffer.
    dynamic_state.Invalidate();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        static const auto scope_loc =
            GPU_SCOPE_LOCATION("Guest Frame", MarkersPalette::GpuMarkerColor);
        new (profiler_scope) tracy::VkCtxScope{profiler_ctx, &scope_loc, current_cmdbuf, true};
    }
#endif
}

void Scheduler::EndSession() {
    if (sessions.empty()) {
        return;
    }
    if (recorder_on) {
        DrainRecorder();
    }

    if (on_session) {
        on_session();
    }

    const auto& session = sessions.back();
    if (session.upload) {
        Check(session.upload.end());
    }

    EndRendering();
    Check(session.primary.end());
}

void Scheduler::SubmitExecution(SubmitInfo& info) {
    std::scoped_lock lk{submit_mutex};
    const u64 signal_value = work_semaphore.NextTick();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    if (on_submit) {
        on_submit(info);
    }

    EndSession();

    std::vector<vk::CommandBuffer> cmd_buffers;
    cmd_buffers.reserve(sessions.size() * 2);

    for (const auto& session : sessions) {
        if (session.upload) {
            cmd_buffers.push_back(session.upload);
        }
        cmd_buffers.push_back(session.primary);
    }
    sessions.clear();

    const vk::Semaphore timeline = work_semaphore.Handle();
    info.AddSignal(timeline, signal_value);

    std::array<vk::PipelineStageFlags, std::tuple_size_v<decltype(info.wait_semas)>>
        wait_stage_masks;
    wait_stage_masks.fill(vk::PipelineStageFlagBits::eAllCommands);

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = info.num_wait_semas,
        .pWaitSemaphoreValues = info.wait_ticks.data(),
        .signalSemaphoreValueCount = info.num_signal_semas,
        .pSignalSemaphoreValues = info.signal_ticks.data(),
    };

    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = info.num_wait_semas,
        .pWaitSemaphores = info.wait_semas.data(),
        .pWaitDstStageMask = wait_stage_masks.data(),
        .commandBufferCount = static_cast<u32>(cmd_buffers.size()),
        .pCommandBuffers = cmd_buffers.data(),
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };

    ImGui::Core::TextureManager::Submit();
    auto submit_result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    work_semaphore.Refresh();
    BeginSession();

    // Apply pending operations
    PopPendingOperations();
}

void Scheduler::PriorityPendingOpsThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuSchedPriorityPendingOpsRunner");

    while (!stoken.stop_requested()) {
        PendingOp op;
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops_cv.wait(lk, stoken,
                                         [this] { return !priority_pending_ops.empty(); });
            if (stoken.stop_requested()) {
                break;
            }

            op = std::move(priority_pending_ops.front());
            priority_pending_ops.pop();
        }

        work_semaphore.Wait(op.gpu_tick);
        if (stoken.stop_requested()) {
            break;
        }

        op.callback();
    }
}

void DynamicState::Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf) {
    if (dirty_state.viewports) {
        dirty_state.viewports = false;
        cmdbuf.setViewportWithCount(viewports);
    }
    if (dirty_state.scissors) {
        dirty_state.scissors = false;
        cmdbuf.setScissorWithCount(scissors);
    }
    if (dirty_state.depth_test_enabled) {
        dirty_state.depth_test_enabled = false;
        cmdbuf.setDepthTestEnable(depth_test_enabled);
    }
    if (dirty_state.depth_write_enabled) {
        dirty_state.depth_write_enabled = false;
        // Note that this must be set in a command buffer even if depth test is disabled.
        cmdbuf.setDepthWriteEnable(depth_write_enabled);
    }
    if (depth_test_enabled && dirty_state.depth_compare_op) {
        dirty_state.depth_compare_op = false;
        cmdbuf.setDepthCompareOp(depth_compare_op);
    }
    if (dirty_state.depth_bounds_test_enabled) {
        dirty_state.depth_bounds_test_enabled = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBoundsTestEnable(depth_bounds_test_enabled);
        }
    }
    if (depth_bounds_test_enabled && dirty_state.depth_bounds) {
        dirty_state.depth_bounds = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBounds(depth_bounds_min, depth_bounds_max);
        }
    }
    if (dirty_state.depth_bias_enabled) {
        dirty_state.depth_bias_enabled = false;
        cmdbuf.setDepthBiasEnable(depth_bias_enabled);
    }
    if (depth_bias_enabled && dirty_state.depth_bias) {
        dirty_state.depth_bias = false;
        cmdbuf.setDepthBias(depth_bias_constant, depth_bias_clamp, depth_bias_slope);
    }
    if (dirty_state.stencil_test_enabled) {
        dirty_state.stencil_test_enabled = false;
        cmdbuf.setStencilTestEnable(stencil_test_enabled);
    }
    if (stencil_test_enabled) {
        if (dirty_state.stencil_front_ops && dirty_state.stencil_back_ops &&
            stencil_front_ops == stencil_back_ops) {
            dirty_state.stencil_front_ops = false;
            dirty_state.stencil_back_ops = false;
            cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack, stencil_front_ops.fail_op,
                                stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                stencil_front_ops.compare_op);
        } else {
            if (dirty_state.stencil_front_ops) {
                dirty_state.stencil_front_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFront, stencil_front_ops.fail_op,
                                    stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                    stencil_front_ops.compare_op);
            }
            if (dirty_state.stencil_back_ops) {
                dirty_state.stencil_back_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eBack, stencil_back_ops.fail_op,
                                    stencil_back_ops.pass_op, stencil_back_ops.depth_fail_op,
                                    stencil_back_ops.compare_op);
            }
        }
        if (dirty_state.stencil_front_reference && dirty_state.stencil_back_reference &&
            stencil_front_reference == stencil_back_reference) {
            dirty_state.stencil_front_reference = false;
            dirty_state.stencil_back_reference = false;
            cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_reference);
        } else {
            if (dirty_state.stencil_front_reference) {
                dirty_state.stencil_front_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_reference);
            }
            if (dirty_state.stencil_back_reference) {
                dirty_state.stencil_back_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eBack, stencil_back_reference);
            }
        }
        if (dirty_state.stencil_front_write_mask && dirty_state.stencil_back_write_mask &&
            stencil_front_write_mask == stencil_back_write_mask) {
            dirty_state.stencil_front_write_mask = false;
            dirty_state.stencil_back_write_mask = false;
            cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_write_mask);
        } else {
            if (dirty_state.stencil_front_write_mask) {
                dirty_state.stencil_front_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_write_mask);
            }
            if (dirty_state.stencil_back_write_mask) {
                dirty_state.stencil_back_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, stencil_back_write_mask);
            }
        }
        if (dirty_state.stencil_front_compare_mask && dirty_state.stencil_back_compare_mask &&
            stencil_front_compare_mask == stencil_back_compare_mask) {
            dirty_state.stencil_front_compare_mask = false;
            dirty_state.stencil_back_compare_mask = false;
            cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                         stencil_front_compare_mask);
        } else {
            if (dirty_state.stencil_front_compare_mask) {
                dirty_state.stencil_front_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFront,
                                             stencil_front_compare_mask);
            }
            if (dirty_state.stencil_back_compare_mask) {
                dirty_state.stencil_back_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eBack,
                                             stencil_back_compare_mask);
            }
        }
    }
    if (dirty_state.primitive_restart_enable) {
        dirty_state.primitive_restart_enable = false;
        cmdbuf.setPrimitiveRestartEnable(primitive_restart_enable);
    }
    if (dirty_state.rasterizer_discard_enable) {
        dirty_state.rasterizer_discard_enable = false;
        cmdbuf.setRasterizerDiscardEnable(rasterizer_discard_enable);
    }
    if (dirty_state.cull_mode) {
        dirty_state.cull_mode = false;
        cmdbuf.setCullMode(cull_mode);
    }
    if (dirty_state.front_face) {
        dirty_state.front_face = false;
        cmdbuf.setFrontFace(front_face);
    }
    if (dirty_state.blend_constants) {
        dirty_state.blend_constants = false;
        cmdbuf.setBlendConstants(blend_constants.data());
    }
    if (dirty_state.color_write_masks) {
        dirty_state.color_write_masks = false;
        if (instance.IsDynamicColorWriteMaskSupported()) {
            cmdbuf.setColorWriteMaskEXT(0, color_write_masks);
        }
    }
    if (dirty_state.line_width) {
        dirty_state.line_width = false;
        cmdbuf.setLineWidth(line_width);
    }
    if (dirty_state.feedback_loop_enabled && instance.IsAttachmentFeedbackLoopLayoutSupported()) {
        dirty_state.feedback_loop_enabled = false;
        cmdbuf.setAttachmentFeedbackLoopEnableEXT(feedback_loop_enabled
                                                      ? vk::ImageAspectFlagBits::eColor
                                                      : vk::ImageAspectFlagBits::eNone);
    }
}

// Overtime recorder (OVERTIME_VK_RECORDER). The GPU thread fills a DrawPacket per direct draw and
// the recorder thread records it into the command buffer. Every direct use of the command buffer
// (CommandBuffer(), EndSession) waits for the recorder first, so the order of commands is kept.

namespace {

void CpuRelax() {
#if defined(_M_X64) || defined(__x86_64__)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

template <typename Flags>
void OrFlags(Flags& dst, const Flags& src) {
    auto* d = reinterpret_cast<u8*>(&dst);
    const auto* s = reinterpret_cast<const u8*>(&src);
    for (size_t i = 0; i < sizeof(Flags); ++i) {
        d[i] |= s[i];
    }
}

} // Anonymous namespace

void DrawPacket::Reset() {
    set_vertex_input = false;
    vertex_bindings.clear();
    vertex_attributes.clear();
    bind_vertex_buffers2 = false;
    vertex_buffers.clear();
    vertex_offsets.clear();
    vertex_sizes.clear();
    vertex_strides.clear();
    bind_index = false;
    bind_index2 = false;
    push_size = 0;
    push_descriptors = false;
    bind_set = false;
    writes.clear();
    buffer_infos.clear();
    image_infos.clear();
    texel_views.clear();
    end_rendering = false;
    begin_rendering = false;
    indexed = false;
}

void Scheduler::EnableRecorder() {
    if (recorder_on) {
        return;
    }
    primary_pool = std::make_unique<CommandPool>(instance, &work_semaphore);
    recorder_ring = std::make_unique<DrawPacket[]>(RecorderRing);
    // The current primary comes from the shared pool: continue in a session of the own pool.
    BeginSession();
    recorder_stat_time = std::chrono::steady_clock::now();
    recorder_on = true;
    recorder_thread = std::thread(&Scheduler::RecorderThread, this);
}

void Scheduler::StopRecorder() {
    if (!recorder_thread.joinable()) {
        return;
    }
    {
        std::scoped_lock lock{recorder_mutex};
        recorder_stop = true;
    }
    recorder_cv.notify_all();
    recorder_thread.join();
}

DrawPacket& Scheduler::BeginPacket() {
    if (recorder_owner == std::thread::id{}) {
        recorder_owner = std::this_thread::get_id();
    }
    const u64 index = recorder_submitted.load(std::memory_order_relaxed);
    if (index - recorder_done.load(std::memory_order_acquire) >= RecorderRing) {
        ++recorder_stat_full;
        WaitRecorder(index - RecorderRing + 1);
    }
    auto& packet = recorder_ring[index % RecorderRing];
    packet.Reset();
    return packet;
}

void Scheduler::SubmitPacket() {
    const u64 index = recorder_submitted.load(std::memory_order_relaxed);
    recorder_ring[index % RecorderRing].cmdbuf = sessions.back().primary;
    recorder_submitted.store(index + 1); // seq_cst: pairs with recorder_sleeping
    if (recorder_sleeping.load()) {
        std::scoped_lock lock{recorder_mutex};
        recorder_cv.notify_one();
    }

    if ((++recorder_stat_packets & 1023) == 0) {
        const auto now = std::chrono::steady_clock::now();
        if (now - recorder_stat_time >= std::chrono::seconds{10}) {
            LOG_INFO(Core,
                     "Overtime recorder (last 10 s): {} draws recorded on the recorder thread, "
                     "{} waits for it ({:.1f} ms in all), {} full-ring waits",
                     recorder_stat_packets, recorder_stat_waits,
                     double(recorder_stat_wait_ns) / 1'000'000.0, recorder_stat_full);
            recorder_stat_packets = recorder_stat_waits = recorder_stat_full = 0;
            recorder_stat_wait_ns = 0;
            recorder_stat_time = now;
        }
    }
}

void Scheduler::DrainRecorder() {
    const u64 submitted = recorder_submitted.load(std::memory_order_relaxed);
    if (submitted == recorder_merged) {
        return;
    }
    if (recorder_done.load(std::memory_order_acquire) != submitted) {
        WaitRecorder(submitted);
    }
    // The recorder is idle: take back the dirty flags its last Commit left set, so a draw
    // committed here emits them.
    OrFlags(dynamic_state.dirty_state, recorder_carry);
    std::memset(&recorder_carry, 0, sizeof(recorder_carry));
    recorder_merged = submitted;
}

void Scheduler::WaitRecorder(u64 packets) {
    if (std::this_thread::get_id() != recorder_owner && !recorder_warned_thread) {
        recorder_warned_thread = true;
        LOG_ERROR(Render_Vulkan, "Overtime recorder: command buffer used from a second thread");
    }
    const auto start = std::chrono::steady_clock::now();
    for (u32 spin = 0; spin < 16384; ++spin) {
        if (recorder_done.load(std::memory_order_acquire) >= packets) {
            break;
        }
        CpuRelax();
    }
    if (recorder_done.load(std::memory_order_acquire) < packets) {
        std::unique_lock lock{drain_mutex};
        drain_waiting.store(true); // seq_cst: pairs with recorder_done
        u32 slices = 0;
        while (!drain_cv.wait_for(lock, std::chrono::seconds{2},
                                  [&] { return recorder_done.load() >= packets; })) {
            LOG_WARNING(Render_Vulkan, "Overtime recorder still busy after {} s ({} of {} draws)",
                        ++slices * 2, recorder_done.load(), packets);
        }
        drain_waiting.store(false);
    }
    ++recorder_stat_waits;
    recorder_stat_wait_ns += static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                             start)
            .count());
}

void Scheduler::RecorderThread() {
    Common::SetCurrentThreadName("shadPS4:VkRecorder");
    u64 next = 0;
    for (;;) {
        u64 available = recorder_submitted.load(std::memory_order_acquire);
        for (u32 spin = 0; spin < 2048 && available == next; ++spin) {
            CpuRelax();
            available = recorder_submitted.load(std::memory_order_acquire);
        }
        if (available == next) {
            std::unique_lock lock{recorder_mutex};
            recorder_sleeping.store(true); // seq_cst: pairs with recorder_submitted
            recorder_cv.wait(lock,
                             [&] { return recorder_stop || recorder_submitted.load() != next; });
            recorder_sleeping.store(false);
            if (recorder_stop) {
                return;
            }
            continue;
        }
        for (; next < available; ++next) {
            RecordPacket(recorder_ring[next % RecorderRing]);
            recorder_done.store(next + 1); // seq_cst: pairs with drain_waiting
            if (drain_waiting.load()) {
                std::scoped_lock lock{drain_mutex};
                drain_cv.notify_all();
            }
        }
    }
}

void Scheduler::RecordPacket(DrawPacket& p) {
    const vk::CommandBuffer cmdbuf = p.cmdbuf;
    if (p.set_vertex_input) {
        cmdbuf.setVertexInputEXT(p.vertex_bindings, p.vertex_attributes);
    }
    if (!p.vertex_buffers.empty()) {
        const auto num_buffers = static_cast<u32>(p.vertex_buffers.size());
        if (p.bind_vertex_buffers2) {
            cmdbuf.bindVertexBuffers2(0, num_buffers, p.vertex_buffers.data(),
                                      p.vertex_offsets.data(), p.vertex_sizes.data(),
                                      p.vertex_strides.data());
        } else {
            cmdbuf.bindVertexBuffers(0, num_buffers, p.vertex_buffers.data(),
                                     p.vertex_offsets.data());
        }
    }
    if (p.bind_index) {
        if (p.bind_index2) {
            cmdbuf.bindIndexBuffer2KHR(p.index_buffer, p.index_offset, p.index_size,
                                       p.index_type);
        } else {
            cmdbuf.bindIndexBuffer(p.index_buffer, p.index_offset, p.index_type);
        }
    }
    if (p.push_size) {
        cmdbuf.pushConstants(p.layout, p.push_stages, 0U, p.push_size, p.push_data.data());
    }
    if (p.push_descriptors) {
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, p.layout, 0, p.writes);
    } else if (p.bind_set) {
        cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, p.layout, 0, p.desc_set, {});
    }

    // The GPU thread cleared its flags when it made the packet; what this Commit leaves set
    // carries over to the next packet (or back to the GPU thread when it drains).
    OrFlags(p.dynamic_state.dirty_state, recorder_carry);
    p.dynamic_state.Commit(instance, cmdbuf);
    recorder_carry = p.dynamic_state.dirty_state;

    if (p.begin_rendering) {
        if (p.end_rendering) {
            cmdbuf.endRendering();
        }
        RecordBeginRendering(cmdbuf, p.render_state);
    }
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, p.pipeline);
    if (p.indexed) {
        cmdbuf.drawIndexed(p.count, p.instances, 0, static_cast<s32>(p.vertex_offset),
                           p.instance_offset);
    } else {
        cmdbuf.draw(p.count, p.instances, p.vertex_offset, p.instance_offset);
    }
}

} // namespace Vulkan
