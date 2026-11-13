// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Overtime switches. Each is on unless its environment variable is "0":
//   OVERTIME_LAZY_COMMIT  (Windows) guest memory backing reserved, committed range by range when
//                         first mapped, instead of the whole backing committed at start
//   OVERTIME_FLIP_TOLERANT  a flip IRQ whose buffer label was already reset logs a warning and
//                         flips, instead of the "Out of order flip IRQ" assert (crash)
//   OVERTIME_POLL_THREAD  controllers (4 ms) and mouse (32 ms) polled from a sleeping thread
//                         instead of SDL timers (SDL's timer thread spins before each tick)

#pragma once

#include <cstdlib>

namespace Overtime {

inline bool SwitchOn(const char* name) {
    const char* v = std::getenv(name);
    return !(v && v[0] == '0');
}

} // namespace Overtime
