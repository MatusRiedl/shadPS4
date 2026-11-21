// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>
#include "core/emulator_settings.h"

// Overtime Native v3 picture options (the "GPU" section; Overtime Roster writes them). Each is off by
// default and changes only the picture: the guest's command stream and timing stay as they are.
namespace Vulkan::HostPasses::Picture {

/// "" (follow fsr_enabled) | "off" | "fsr1" | "dlss"
inline std::string Upscaler() {
    return EmulatorSettings.GetUpscaler();
}

/// FSR 1 at the upscale slot: the old switch while no upscaler is chosen.
inline bool Fsr1() {
    const auto upscaler = Upscaler();
    return upscaler.empty() ? EmulatorSettings.IsFsrEnabled() : upscaler == "fsr1";
}

/// DLSS Super Resolution at the upscale slot.
inline bool DlssSr() {
    return Upscaler() == "dlss";
}

/// DLSS 5 (NVIDIA's Neural Rendering model, the player's own nvngx_dlssnr.dll) over the frame.
inline bool DlssNr() {
    return EmulatorSettings.IsDlssNr();
}

/// DLSS Frame Generation at present (experimental).
inline bool DlssFg() {
    return EmulatorSettings.GetFrameGen() == "dlss";
}

/// The transform-feedback capture, velocity, depth and camera passes run only for a feature that
/// reads them, or for the motion vector debug view.
inline bool MotionDebug() {
    return EmulatorSettings.IsMotionDebug();
}

inline bool MotionInputs() {
    return DlssSr() || DlssNr() || DlssFg() || EmulatorSettings.IsMotionDebug();
}

} // namespace Vulkan::HostPasses::Picture
