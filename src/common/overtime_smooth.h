// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Overtime lab switches for frame pacing. Each is on unless its environment variable is "0":
//   OVERTIME_SMOOTH_PRIORITY  process above normal, power throttling off, GPU/present threads highest
//   OVERTIME_SMOOTH_FLIP      present a finished frame at once instead of at the next vblank tick
//                             (at most one per vblank period, so the frame cap stays the vblank rate)

#pragma once

#include <cstdlib>

namespace Overtime {

inline bool SwitchOn(const char* name) {
    const char* v = std::getenv(name);
    return !(v && v[0] == '0');
}

inline bool SmoothPriority() {
    static const bool on = SwitchOn("OVERTIME_SMOOTH_PRIORITY");
    return on;
}

inline bool SmoothFlip() {
    static const bool on = SwitchOn("OVERTIME_SMOOTH_FLIP");
    return on;
}

} // namespace Overtime
