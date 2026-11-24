// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// Overtime Native v3: DLSS frame generation (GPU.frame_gen = "dlss", experimental). The decisions,
// free of Vulkan and Streamline; frame_gen.cpp feeds them and acts on the result. Ported from the
// owner's KytyPS5 frame generation.

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>

namespace Vulkan::HostPasses::FrameGen {

// The production DLLs of Streamline 2.14.16 (bin/x64) that frame generation loads, next to the exe.
// NvLowLatencyVk.dll is left out: sl.common only falls back to it without VK_NV_low_latency2, and
// its licence does not list it as distributable.
inline constexpr std::array<std::string_view, 6> RequiredDlls = {
    "sl.interposer.dll", "sl.common.dll", "sl.dlss_g.dll",
    "nvngx_dlssg.dll",   "sl.reflex.dll", "sl.pcl.dll",
};

enum class Loader : uint8_t { Plain, Streamline };

struct LoaderInputs {
    bool frame_gen_switch = false; // GPU.frame_gen = "dlss"
    bool built_with_dlssg = false; // OVERTIME_ENABLE_DLSSG
    bool windowed = false;         // a real window (not the headless instance)
    std::string_view missing_dll;  // the first of RequiredDlls not found, empty when all are
};

struct LoaderDecision {
    Loader loader = Loader::Plain;
    std::string_view reason;
};

// Before the Vulkan instance exists: Streamline's loader or the plain one (exactly as before). The
// adapter check comes later (slIsFeatureSupported needs the physical device); a failure there keeps
// the Streamline loader with frame generation off.
[[nodiscard]] constexpr LoaderDecision DecideLoader(const LoaderInputs& in) {
    if (!in.frame_gen_switch) {
        return {Loader::Plain, "off"};
    }
    if (!in.built_with_dlssg) {
        return {Loader::Plain, "not-built"};
    }
    if (!in.windowed) {
        return {Loader::Plain, "headless"};
    }
    if (!in.missing_dll.empty()) {
        return {Loader::Plain, "dll-missing"};
    }
    return {Loader::Streamline, "ok"};
}

// GPU.frame_gen_frames (1..3 = 2x..4x, default 1): generated frames per game frame, within what
// the driver reports (DLSSGState::numFramesToGenerateMax; 0 = not reported yet, then 1).
[[nodiscard]] constexpr uint32_t FramesToGenerate(uint32_t requested, uint32_t reported_max) {
    const uint32_t max = reported_max == 0 ? 1u : reported_max;
    return std::clamp(requested == 0 ? 1u : requested, 1u, std::min(max, 3u));
}

// Frame generation turns on and off with a swapchain recreate, so never per frame: it turns on
// after on_after eligible game frames in a row and off after off_after ineligible ones. A failure
// reported by Streamline turns it off for the rest of the session.
class ModeGate {
public:
    struct Config {
        uint32_t on_after = 60;
        uint32_t off_after = 30;
        bool hold = false; // OVERTIME_DLSSG_HOLD=1: Streamline loaded, frame generation never on
    };
    struct Inputs {
        bool tags_ready = false; // this present shows a frame tagged with depth and motion
        bool window_ok = false;  // not minimized, at least DLSSGState::minWidthOrHeight
    };
    struct Step {
        bool on = false;
        bool changed = false;
        std::string_view reason;
    };

    ModeGate() = default;
    explicit ModeGate(Config config) : m_config(config) {}

    // Once per present: in play a present is a game frame; a repeat, a blank frame or a frame
    // without motion inputs is not eligible, so a pause turns the mode off after off_after.
    Step Update(const Inputs& in) {
        if (m_failed) {
            return {false, false, m_fail_reason};
        }
        if (m_config.hold) {
            return {false, false, "hold"};
        }
        const bool eligible = in.tags_ready && in.window_ok;
        const std::string_view why = eligible ? "eligible" : (!in.window_ok ? "window" : "no-tags");
        if (eligible) {
            m_ineligible = 0;
            if (!m_on && ++m_eligible >= m_config.on_after) {
                m_on = true;
                m_eligible = 0;
                return {true, true, why};
            }
        } else {
            m_eligible = 0;
            if (m_on && ++m_ineligible >= m_config.off_after) {
                m_on = false;
                m_ineligible = 0;
                return {false, true, why};
            }
        }
        return {m_on, false, why};
    }
    // Returns true when this turned the mode off. The first reason is kept. reason must outlive
    // the gate (a literal).
    bool Fail(std::string_view reason) {
        const bool was_on = m_on;
        if (!m_failed) {
            m_fail_reason = reason;
        }
        m_failed = true;
        m_on = false;
        return was_on;
    }
    [[nodiscard]] bool On() const {
        return m_on;
    }
    [[nodiscard]] bool Failed() const {
        return m_failed;
    }

private:
    Config m_config{};
    bool m_on = false;
    bool m_failed = false;
    uint32_t m_eligible = 0;
    uint32_t m_ineligible = 0;
    std::string_view m_fail_reason;
};

// Which frames carry tags. A frame is numbered by its guest flip (1, 2, ...); the number rides with
// the flip to the GPU thread and with the presenter's Frame to its present, so Streamline sees the
// simulation markers, tags, constants and present of one frame under one frame token, also while
// the guest runs frames ahead of the GPU. Callers hold one mutex around these calls.
class FrameLedger {
public:
    void Tagged(uint64_t frame) {
        m_ring[frame % m_ring.size()] = frame;
        m_last_tagged = std::max(m_last_tagged, frame);
    }
    // True when `frame` (not 0) was tagged among the last 64 frames.
    [[nodiscard]] bool WasTagged(uint64_t frame) const {
        return frame != 0 && m_ring[frame % m_ring.size()] == frame;
    }
    [[nodiscard]] uint64_t LastTagged() const {
        return m_last_tagged;
    }

private:
    std::array<uint64_t, 64> m_ring{};
    uint64_t m_last_tagged = 0;
};

// Turning frame generation on and off around swapchain recreates (present thread). On: recreate in
// a frame-generation present mode first, then switch Streamline's mode on. Off: switch the mode off
// (it takes effect at the next present), present once, then recreate in the requested mode. A
// recreate for any other reason (resize, out of date, HDR) turns it off first and waits `cooldown`
// presents before turning on again (DLSS-G guide: off before window changes).
class ModeSequencer {
public:
    enum class State : uint8_t { Off, WaitRecreateOn, On, WaitPresentOff, WaitRecreateOff };
    struct Output {
        bool set_options = false; // call slDLSSGSetOptions with options_on
        bool options_on = false;
        bool request_recreate = false; // recreate the swapchain before presenting
    };

    explicit ModeSequencer(uint32_t cooldown = 30) : m_cooldown(cooldown) {}

    // Before each present.
    Output BeforePresent(bool gate_on) {
        if (m_wait > 0) {
            m_wait--;
        }
        switch (m_state) {
        case State::Off:
            if (gate_on && m_wait == 0) {
                m_state = State::WaitRecreateOn;
                return {false, false, true};
            }
            return {};
        case State::WaitRecreateOn:
            return {}; // the recreate is pending
        case State::On:
            if (!gate_on) {
                m_state = State::WaitPresentOff;
                return {true, false, false};
            }
            return {};
        case State::WaitPresentOff:
            m_state = State::WaitRecreateOff;
            return {false, false, true};
        case State::WaitRecreateOff:
            return {};
        }
        return {};
    }
    // After a swapchain recreate this sequencer asked for. Returns the options call to make.
    Output Recreated(bool gate_on) {
        if (m_state == State::WaitRecreateOn) {
            if (gate_on) {
                m_state = State::On;
                return {true, true, false};
            }
            // The gate went off while waiting: back to the plain mode.
            m_state = State::WaitRecreateOff;
            return {false, false, true};
        }
        if (m_state == State::WaitRecreateOff) {
            m_state = State::Off;
        }
        return {};
    }
    // Before a recreate for another reason. Returns the options call to make first (mode off).
    Output OutsideRecreate() {
        const bool was_on = m_state == State::On || m_state == State::WaitPresentOff;
        m_state = State::Off;
        m_wait = m_cooldown;
        return was_on ? Output{true, false, false} : Output{};
    }
    // The present mode the next swapchain is created with: a frame-generation mode from the moment
    // turning on is decided until turning off is complete.
    [[nodiscard]] bool FrameGenPresentMode() const {
        return m_state == State::WaitRecreateOn || m_state == State::On ||
               m_state == State::WaitPresentOff;
    }
    [[nodiscard]] State Current() const {
        return m_state;
    }

private:
    State m_state = State::Off;
    uint32_t m_cooldown;
    uint32_t m_wait = 0;
};

// Frame generation on Vulkan does not run with FIFO (VSync): mailbox when the surface has it,
// otherwise immediate. Returns the requested mode when frame generation is off.
enum class PresentMode : uint8_t { Fifo, FifoRelaxed, Mailbox, Immediate };
[[nodiscard]] constexpr PresentMode PresentModeFor(bool frame_gen_on, PresentMode requested,
                                                   bool mailbox_supported,
                                                   bool immediate_supported) {
    if (!frame_gen_on || requested == PresentMode::Mailbox || requested == PresentMode::Immediate) {
        return requested;
    }
    if (mailbox_supported) {
        return PresentMode::Mailbox;
    }
    return immediate_supported ? PresentMode::Immediate : requested;
}

} // namespace Vulkan::HostPasses::FrameGen
