// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Overtime switches. Each is on unless its environment variable is "0":
//   OVERTIME_LAZY_COMMIT  (Windows) guest memory backing reserved, committed range by range when
//                         first mapped, instead of the whole backing committed at start

#pragma once

#include <cstdlib>

namespace Overtime {

inline bool SwitchOn(const char* name) {
    const char* v = std::getenv(name);
    return !(v && v[0] == '0');
}

} // namespace Overtime
